# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Dual-model PPO Trainer with Ray-based single controller.

- DOES NOT inherit RayPPOTrainer.
- Reuses/copies most of the single-model dataflow.
- Runs per-iteration pipeline as:
    rollout(A) -> reward/adv -> train critic -> train actor(A) -> sync(A)
    rollout(B) -> reward/adv -> train critic -> train actor(B) -> sync(B)

Assumptions:
- You introduced Role.ActorRolloutA / Role.ActorRolloutB in verl.trainer.ppo.utils.Role (or ray_trainer.Role).
- TaskRunner filled:
    self.role_worker_mapping[Role.ActorRolloutA] = ray.remote(ActorRolloutRefWorker)
    self.role_worker_mapping[Role.ActorRolloutB] = ray.remote(ActorRolloutRefWorker)
  and both mapped to "global_pool".
- Config contains:
    config.actor_rollout_ref_a
    config.actor_rollout_ref_b
  Each matches the original config.actor_rollout_ref schema.

Important fixes vs your draft:
1) AgentLoopManager must receive a FULL config (top-level), not a sub-tree.
   We return a deep-copied global config whose config.actor_rollout_ref is swapped to A/B.
2) Removed buggy _strip_keys usage.
3) balance_batch dp_size lookup uses the correct role key (Role.ActorRolloutA/B), not "actor_a/b".
"""

from __future__ import annotations

import json
import os
import uuid
from collections import defaultdict
from copy import deepcopy
from pprint import pprint
from typing import Optional

import numpy as np
import ray
import torch
from omegaconf import OmegaConf, open_dict
from torch.utils.data import Dataset, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm

from verl import DataProto
from verl.checkpoint_engine import CheckpointEngineManager
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.ray import RayClassWithInitArgs, RayWorkerGroup, ResourcePoolManager
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import AdvantageEstimator, agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    compute_variance_proxy_metrics,
    process_validation_metrics,
)
from verl.trainer.ppo.reward import compute_reward, compute_reward_async
from verl.trainer.ppo.utils import Role, WorkerType, need_critic, need_reward_model
from verl.utils import tensordict_utils as tu
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path, should_save_ckpt_esi
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.debug import marked_timer
from verl.utils.import_utils import load_class_from_fqn
from verl.utils.metric import reduce_metrics
from verl.utils.model import compute_position_id_with_mask
from verl.utils.py_functional import rename_dict
from verl.utils.rollout_skip import RolloutSkip
from verl.utils.seqlen_balancing import calculate_workload, get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.torch_functional import masked_mean
from verl.utils.tracking import ValidationGenerationsLogger
from verl.workers.config import FSDPEngineConfig
from verl.workers.utils.padding import left_right_2_no_padding, no_padding_2_padding

from dataclasses import dataclass, field
from collections import defaultdict, deque
from typing import Dict, Optional
import time
from tensordict import NonTensorData
from concurrent.futures import ThreadPoolExecutor
import sys, time

def rtprint(msg: str):
    print(msg, flush=True)

def attach_global_token_num(td):
    if "attention_mask" in td.keys():
        # attention_mask: [B, T]，每行 sum 就是 token 数
        token_num = td["attention_mask"].sum(dim=-1).to(torch.int).tolist()
    else:
        # fallback（需要你 pad_id 正确）
        pad_id = 0
        token_num = (td["input_ids"] != pad_id).sum(dim=-1).to(torch.int).tolist()

    tu.assign_non_tensor(td, global_token_num=NonTensorData(token_num))
    return td

def rollout_token_stats_from_batch(batch: DataProto, tokenizer=None):
    """
    Return dict of stats for rollout token lengths in current batch.
    Prefer non_tensor 'global_token_num' (you attached in compute_log_prob path),
    else fallback to attention_mask.
    """
    lens = None

    # 1) prefer NonTensorData attached in tensordict path
    # After DataProto conversions, this may live in batch.non_tensor_batch or batch.meta_info depending on your path.
    if hasattr(batch, "non_tensor_batch") and "global_token_num" in batch.non_tensor_batch:
        lens = batch.non_tensor_batch["global_token_num"]
    elif hasattr(batch, "meta_info") and "global_token_num" in batch.meta_info:
        lens = batch.meta_info["global_token_num"]
    elif "attention_mask" in batch.batch:
        am = batch.batch["attention_mask"]
        lens = am.sum(dim=-1).to(torch.int).cpu().tolist()
    else:
        # last resort
        pad_id = getattr(tokenizer, "pad_token_id", 0) if tokenizer is not None else 0
        if "input_ids" in batch.batch:
            lens = (batch.batch["input_ids"] != pad_id).sum(dim=-1).to(torch.int).cpu().tolist()

    if lens is None:
        return {}

    # normalize to python list[int]
    if isinstance(lens, np.ndarray):
        lens = lens.tolist()
    if isinstance(lens, torch.Tensor):
        lens = lens.cpu().tolist()

    if len(lens) == 0:
        return {}

    arr = np.asarray(lens, dtype=np.int64)
    return {
        "rollout_tokens/min": int(arr.min()),
        "rollout_tokens/max": int(arr.max()),
        "rollout_tokens/mean": float(arr.mean()),
        "rollout_tokens/p50": float(np.percentile(arr, 50)),
        "rollout_tokens/p90": float(np.percentile(arr, 90)),
        "rollout_tokens/p99": float(np.percentile(arr, 99)),
        "rollout_tokens/n": int(arr.size),
    }

@dataclass
class StepProfiler:
    enabled: bool = True
    verbose: bool = True                  # 是否每步/每rollout打印
    print_every: int = 1                  # 每多少 step 打一次汇总
    window: int = 1.                      # 滑动窗口大小
    print_rollout_each: bool = True       # 是否每次 rollout 都打印一次（会很吵）

    _hist: Dict[str, deque] = field(default_factory=lambda: defaultdict(deque))
    _sum: Dict[str, float] = field(default_factory=lambda: defaultdict(float))
    _cnt: int = 0

    def update(self, timing_raw: Dict[str, float], step: int, prefix: str = ""):
        if not self.enabled:
            return
        self._cnt += 1

        # timing_raw 里通常是 {name: seconds}
        for k, v in timing_raw.items():
            if v is None:
                continue
            key = f"{prefix}{k}" if prefix else k
            self._sum[key] += float(v)
            dq = self._hist[key]
            dq.append(float(v))
            if len(dq) > self.window:
                dq.popleft()

        # 打印策略
        if self.verbose and self.print_every > 0 and (step % self.print_every == 0):
            print(self.format_report(step=step))

    def mean(self, key: str) -> float:
        dq = self._hist.get(key)
        if not dq:
            return 0.0
        return sum(dq) / len(dq)

    def total_mean(self, key: str) -> float:
        if self._cnt == 0:
            return 0.0
        return self._sum.get(key, 0.0) / self._cnt

    def format_report(self, step: int) -> str:
        # 你关心的主路径：rollout(两边) + train(update actor/critic) + ref + reward 等
        keys = [
            "step",
            "gen_a", "a_train_wall", "gen_b", "b_train_wall",
            "reward_a", "old_log_prob_a", "ref_log_prob_a", "values_a", "adv_a",
            "reward_b", "old_log_prob_b", "ref_log_prob_b", "values_b", "adv_b",
            "update_critic_a", "update_critic_b",
            "update_actor_a", "update_actor_b",
            "save_checkpoint", "update_weights_a", "update_weights_b",
            "testing",
        ]

        present = [k for k in keys if k in self._hist]
        parts = [f"[profile] step={step} window={self.window}"]

        # a_train = reward_a + old_log_prob_a + ref_log_prob_a + adv_a + update_critic_a + update_actor_a
        a_train_keys = ["reward_a", "old_log_prob_a", "ref_log_prob_a", "adv_a", "update_critic_a", "update_actor_a"]
        if any(k in self._hist for k in a_train_keys):
            a_sum = sum(self.mean(k) for k in a_train_keys if k in self._hist)
            parts.append(f"a_train_sum={a_sum:.2f}s")
        if "a_train_wall" in self._hist:
            parts.append(f"a_train_wall={self.mean('a_train_wall'):.2f}s")
        # b_train = step_b + update_critic_b + update_actor_b (= reward_b + old_log_prob_b + ref_log_prob_b + adv_b + ...)
        b_train_keys = ["reward_b", "old_log_prob_b", "ref_log_prob_b", "adv_b", "update_critic_b", "update_actor_b"]
        if any(k in self._hist for k in b_train_keys):
            b_sum = sum(self.mean(k) for k in b_train_keys if k in self._hist)
            parts.append(f"b_train_sum={b_sum:.2f}s")
        if "b_train_wall" in self._hist:
            parts.append(f"b_train_wall={self.mean('b_train_wall'):.2f}s")

        for k in present:
            if k not in ("a_train_wall", "b_train_wall"):
                parts.append(f"{k}: {self.mean(k):.2f}s")
        return " | ".join(parts)

# -------------------------
# Small shared helpers
# -------------------------
def compute_response_mask(data: DataProto):
    responses = data.batch["responses"]
    response_length = responses.size(1)
    attention_mask = data.batch["attention_mask"]
    return attention_mask[:, -response_length:]


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty="kl"):
    response_mask = data.batch["response_mask"]
    token_level_scores = data.batch["token_level_scores"]
    batch_size = data.batch.batch_size[0]

    kld = core_algos.kl_penalty(data.batch["old_log_probs"], data.batch["ref_log_prob"], kl_penalty=kl_penalty)
    kld = kld * response_mask
    beta = kl_ctrl.value
    token_level_rewards = token_level_scores - beta * kld

    current_kl = masked_mean(kld, mask=response_mask, axis=-1)
    current_kl = torch.mean(current_kl, dim=0).item()

    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch["token_level_rewards"] = token_level_rewards

    metrics = {"actor/reward_kl_penalty": current_kl, "actor/reward_kl_penalty_coeff": beta}
    return data, metrics


def compute_advantage(
    data: DataProto,
    adv_estimator: AdvantageEstimator,
    gamma: float = 1.0,
    lam: float = 1.0,
    num_repeat: int = 1,
    norm_adv_by_std_in_grpo: bool = True,
    config=None,
) -> DataProto:
    if "response_mask" not in data.batch.keys():
        data.batch["response_mask"] = compute_response_mask(data)

    if adv_estimator == AdvantageEstimator.GAE:
        advantages, returns = core_algos.compute_gae_advantage_return(
            token_level_rewards=data.batch["token_level_rewards"],
            values=data.batch["values"],
            response_mask=data.batch["response_mask"],
            gamma=gamma,
            lam=lam,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.GRPO:
        advantages, returns = core_algos.compute_grpo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=data.batch["response_mask"],
            index=data.non_tensor_batch["uid"],
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    else:
        adv_estimator_fn = core_algos.get_adv_estimator_fn(adv_estimator)
        adv_kwargs = {
            "token_level_rewards": data.batch["token_level_rewards"],
            "response_mask": data.batch["response_mask"],
            "config": config,
        }
        if "uid" in data.non_tensor_batch:
            adv_kwargs["index"] = data.non_tensor_batch["uid"]
        if "reward_baselines" in data.batch:
            adv_kwargs["reward_baselines"] = data.batch["reward_baselines"]

        advantages, returns = adv_estimator_fn(**adv_kwargs)
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    return data


# -------------------------
# Dual trainer
# -------------------------
class DualRayPPOTrainer:
    """
    Clean dual-model trainer. No inheritance.

    Maintains two actor contexts (A and B):
    - actor rollout WG
    - async rollout manager
    - checkpoint manager
    - actor config (actor_rollout_ref_a/b)
    """

    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: type[RayWorkerGroup] = RayWorkerGroup,
        processor=None,
        reward_fn=None,
        val_reward_fn=None,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        collate_fn=None,
        train_sampler: Optional[Sampler] = None,
        device_name=None,
    ):
        self.config = config
        self.tokenizer = tokenizer
        self.processor = processor
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        assert hasattr(config, "actor_rollout_ref_a") and hasattr(config, "actor_rollout_ref_b"), (
            "Dual trainer requires config.actor_rollout_ref_a and config.actor_rollout_ref_b"
        )
        self.cfg_a = config.actor_rollout_ref_a
        self.cfg_b = config.actor_rollout_ref_b

        self.use_legacy_worker_impl = config.trainer.get("use_legacy_worker_impl", "auto")
        assert self.use_legacy_worker_impl == "disable", "Dual trainer assumes new engine (use_legacy_worker_impl=disable)."

        assert self.cfg_a.model.get("lora", {}).get("rank", 0) <= 0
        assert self.cfg_b.model.get("lora", {}).get("rank", 0) <= 0

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.ray_worker_group_cls = ray_worker_group_cls

        self.use_rm = need_reward_model(self.role_worker_mapping)
        self.use_reward_loop = self.config.reward_model.use_reward_loop
        self.use_critic = need_critic(self.config)

        self.device_name = device_name if device_name else self.config.trainer.device
        self.validation_generations_logger = ValidationGenerationsLogger(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
        )

        if self.config.algorithm.use_kl_in_reward:
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(self.config.algorithm.kl_ctrl)

        self.use_prefix_grouper = self.cfg_a.actor.get("use_prefix_grouper", False) or self.cfg_b.actor.get(
            "use_prefix_grouper", False
        )

        self._create_dataloader(train_dataset, val_dataset, collate_fn, train_sampler)

    # -------------------------
    # Dataloader
    # -------------------------
    def _create_dataloader(self, train_dataset, val_dataset, collate_fn, train_sampler: Optional[Sampler]):
        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler

        if train_dataset is None:
            train_dataset = create_rl_dataset(
                self.config.data.train_files,
                self.config.data,
                self.tokenizer,
                self.processor,
                max_samples=self.config.data.get("train_max_samples", -1),
            )
        if val_dataset is None:
            val_dataset = create_rl_dataset(
                self.config.data.val_files,
                self.config.data,
                self.tokenizer,
                self.processor,
                max_samples=self.config.data.get("val_max_samples", -1),
            )

        self.train_dataset, self.val_dataset = train_dataset, val_dataset

        if train_sampler is None:
            train_sampler = create_rl_sampler(self.config.data, self.train_dataset)
        if collate_fn is None:
            from verl.utils.dataset.rl_dataset import collate_fn as default_collate_fn

            collate_fn = default_collate_fn

        num_workers = self.config.data["dataloader_num_workers"]
        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.data.get("gen_batch_size", self.config.data.train_batch_size),
            num_workers=num_workers,
            drop_last=True,
            collate_fn=collate_fn,
            sampler=train_sampler,
        )

        val_batch_size = self.config.data.val_batch_size
        if val_batch_size is None:
            val_batch_size = len(self.val_dataset)

        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            batch_size=val_batch_size,
            num_workers=num_workers,
            shuffle=self.config.data.get("validation_shuffle", True),
            drop_last=False,
            collate_fn=collate_fn,
        )

        assert len(self.train_dataloader) >= 1, "Train dataloader is empty!"
        assert len(self.val_dataloader) >= 1, "Validation dataloader is empty!"

        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs
        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps
        self.total_training_steps = total_training_steps

        # Best-effort set total steps
        try:
            OmegaConf.set_struct(self.config, True)
            with open_dict(self.config):
                if OmegaConf.select(self.config, "critic.optim"):
                    self.config.critic.optim.total_training_steps = total_training_steps
                if OmegaConf.select(self.config, "actor_rollout_ref_a.actor.optim"):
                    self.config.actor_rollout_ref_a.actor.optim.total_training_steps = total_training_steps
                if OmegaConf.select(self.config, "actor_rollout_ref_b.actor.optim"):
                    self.config.actor_rollout_ref_b.actor.optim.total_training_steps = total_training_steps
        except Exception as e:
            print(f"Warning: Could not set total_training_steps in config. Error: {e}")

    # -------------------------
    # Worker init
    # -------------------------
    def init_workers(self):
        self.resource_pool_manager.create_resource_pool()
        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        assert Role.ActorRolloutA in self.role_worker_mapping, "Missing Role.ActorRolloutA mapping"
        assert Role.ActorRolloutB in self.role_worker_mapping, "Missing Role.ActorRolloutB mapping"

        pool_a = self.resource_pool_manager.get_resource_pool(Role.ActorRolloutA)
        pool_b = self.resource_pool_manager.get_resource_pool(Role.ActorRolloutB)
        assert pool_a == pool_b, "Both actors must be in the same pool for colocated spawn."

        actor_a_cls = RayClassWithInitArgs(
            cls=self.role_worker_mapping[Role.ActorRolloutA],
            config=self.cfg_a,
            role=str(Role.ActorRolloutA),
        )
        actor_b_cls = RayClassWithInitArgs(
            cls=self.role_worker_mapping[Role.ActorRolloutB],
            config=self.cfg_b,
            role=str(Role.ActorRolloutB),
        )
        self.resource_pool_to_cls[pool_a][str(Role.ActorRolloutA)] = actor_a_cls
        self.resource_pool_to_cls[pool_b][str(Role.ActorRolloutB)] = actor_b_cls

        # Critic
        if self.use_critic:
            pool_c = self.resource_pool_manager.get_resource_pool(Role.Critic)
            from verl.workers.config import CriticConfig

            critic_cfg: CriticConfig = omega_conf_to_dataclass(self.config.critic)

            from verl.workers.engine_workers import TrainingWorkerConfig

            orig_critic_cfg = critic_cfg
            if orig_critic_cfg.strategy in ("fsdp", "fsdp2"):
                engine_config = orig_critic_cfg.model.fsdp_config
                engine_config.infer_max_token_len_per_gpu = critic_cfg.ppo_infer_max_token_len_per_gpu
                engine_config.max_token_len_per_gpu = critic_cfg.ppo_max_token_len_per_gpu
            elif orig_critic_cfg.strategy == "megatron":
                engine_config = orig_critic_cfg.megatron
                engine_config.infer_max_token_len_per_gpu = critic_cfg.ppo_infer_max_token_len_per_gpu
                engine_config.max_token_len_per_gpu = critic_cfg.ppo_max_token_len_per_gpu
            else:
                raise NotImplementedError(f"Unknown critic strategy {orig_critic_cfg.strategy=}")

            critic_cfg = TrainingWorkerConfig(
                model_type="value_model",
                model_config=orig_critic_cfg.model_config,
                engine_config=engine_config,
                optimizer_config=orig_critic_cfg.optim,
                checkpoint_config=orig_critic_cfg.checkpoint,
            )

            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=critic_cfg)
            self.resource_pool_to_cls[pool_c][str(Role.Critic)] = critic_cls
            self._orig_critic_cfg_for_loss = orig_critic_cfg

        # Reward model WG not supported here; require reward_loop
        if self.use_rm and not self.use_reward_loop:
            raise RuntimeError("Dual trainer requires use_reward_loop=True if reward model is enabled.")

        # Spawn colocated worker groups per pool
        all_wg = {}
        wg_kwargs = {}
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout
        if OmegaConf.select(self.config.global_profiler, "steps") is not None:
            wg_kwargs["profile_steps"] = OmegaConf.select(self.config.global_profiler, "steps")
            if OmegaConf.select(self.config.global_profiler, "tool") == "nsys":
                assert (
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                    is not None
                ), "worker_nsight_options must be set for nsys"
                wg_kwargs["worker_nsight_options"] = OmegaConf.to_container(
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                )
        wg_kwargs["device_name"] = self.device_name

        trainer_mps_pct = OmegaConf.select(self.config.trainer, "mps_active_thread_percentage", default=0)
        if trainer_mps_pct and int(trainer_mps_pct) > 0:
            wg_kwargs["worker_env"] = {
                "CUDA_MPS_PIPE_DIRECTORY": "/tmp/nvidia-mps",
                "CUDA_MPS_LOG_DIRECTORY": "/tmp/nvidia-mps-log",
                "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE": str(trainer_mps_pct),
            }
            print(f"[MPS] Training workers will use CUDA_MPS_ACTIVE_THREAD_PERCENTAGE={trainer_mps_pct}", flush=True)

        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            if not class_dict:
                continue
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(
                resource_pool=resource_pool,
                ray_cls_with_init=worker_dict_cls,
                **wg_kwargs,
            )
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)

        # Bind WGs
        self.actor_rollout_wg_a = all_wg[str(Role.ActorRolloutA)]
        self.actor_rollout_wg_b = all_wg[str(Role.ActorRolloutB)]
        self.actor_rollout_wg_a.init_model()
        self.actor_rollout_wg_b.init_model()

        if self.use_critic:
            self.critic_wg = all_wg[str(Role.Critic)]
            self.critic_wg.reset()
            from functools import partial
            from verl.workers.utils.losses import value_loss

            self.critic_wg.set_loss_fn(partial(value_loss, config=self._orig_critic_cfg_for_loss))

        # Reward loop manager (shared)
        if self.use_reward_loop:
            from verl.experimental.reward_loop import RewardLoopManager

            rm_resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel) if self.use_rm else None
            self.reward_loop_manager = RewardLoopManager(config=self.config, rm_resource_pool=rm_resource_pool)
        else:
            self.reward_loop_manager = None

        # Async rollout managers (one per actor WG)
        self.async_rollout_mode = True

        manager_class_fqn_a = self.cfg_a.rollout.get("agent", {}).get("agent_loop_manager_class")
        manager_class_fqn_b = self.cfg_b.rollout.get("agent", {}).get("agent_loop_manager_class")
        if manager_class_fqn_a or manager_class_fqn_b:
            AgentLoopManagerA = load_class_from_fqn(manager_class_fqn_a, "AgentLoopManager") if manager_class_fqn_a else None
            AgentLoopManagerB = load_class_from_fqn(manager_class_fqn_b, "AgentLoopManager") if manager_class_fqn_b else None
        else:
            AgentLoopManagerA = None
            AgentLoopManagerB = None

        from verl.experimental.agent_loop import AgentLoopManager as DefaultAgentLoopManager

        pool_actor = pool_a

        enable_agent_reward_loop = self.use_reward_loop and ((not self.use_rm) or self.config.reward_model.enable_resource_pool)
        reward_loop_worker_handles = self.reward_loop_manager.reward_loop_workers if enable_agent_reward_loop else None

        # IMPORTANT: pass FULL config (top-level), not sub-tree
        self.async_rollout_manager_a = (AgentLoopManagerA or DefaultAgentLoopManager)(
            config=self._build_actor_scoped_config("a"),
            worker_group=self.actor_rollout_wg_a,
            rollout_resource_pool=pool_actor,
            reward_loop_worker_handles=reward_loop_worker_handles,
        )
        self.async_rollout_manager_b = (AgentLoopManagerB or DefaultAgentLoopManager)(
            config=self._build_actor_scoped_config("b"),
            worker_group=self.actor_rollout_wg_b,
            rollout_resource_pool=pool_actor,
            reward_loop_worker_handles=reward_loop_worker_handles,
        )

        # Checkpoint managers
        self.checkpoint_manager_a = CheckpointEngineManager(
            backend=self.cfg_a.rollout.checkpoint_engine.backend,
            trainer=self.actor_rollout_wg_a,
            replicas=self.async_rollout_manager_a.rollout_replicas,
        )
        self.checkpoint_manager_b = CheckpointEngineManager(
            backend=self.cfg_b.rollout.checkpoint_engine.backend,
            trainer=self.actor_rollout_wg_b,
            replicas=self.async_rollout_manager_b.rollout_replicas,
        )

        self.checkpoint_manager_a.sleep_replicas()
        self.checkpoint_manager_b.sleep_replicas()

    def _build_actor_scoped_config(self, which: str):
        """
        Return a FULL config for AgentLoopManager, keeping single-model schema,
        but swapping config.actor_rollout_ref to A/B.
        """
        assert which in ("a", "b")
        cfg = deepcopy(self.config)

        actor_sub = deepcopy(getattr(self.config, f"actor_rollout_ref_{which}", None))
        if actor_sub is None:
            actor_sub = deepcopy(self.config.actor_rollout_ref)

        # Normalize optional path_a/path_b and n_a/n_b if they are still present
        with open_dict(actor_sub):
            path_key = f"path_{which}"
            if OmegaConf.select(actor_sub, f"model.{path_key}") is not None:
                actor_sub.model.path = OmegaConf.select(actor_sub, f"model.{path_key}")
                if OmegaConf.select(actor_sub, "model.path_a") is not None:
                    del actor_sub.model["path_a"]
                if OmegaConf.select(actor_sub, "model.path_b") is not None:
                    del actor_sub.model["path_b"]

            n_key = f"n_{which}"
            if OmegaConf.select(actor_sub, f"rollout.{n_key}") is not None:
                actor_sub.rollout.n = OmegaConf.select(actor_sub, f"rollout.{n_key}")
                if OmegaConf.select(actor_sub, "rollout.n_a") is not None:
                    del actor_sub.rollout["n_a"]
                if OmegaConf.select(actor_sub, "rollout.n_b") is not None:
                    del actor_sub.rollout["n_b"]

        with open_dict(cfg):
            cfg.actor_rollout_ref = actor_sub
            # keep these too (for your own access)
            cfg.actor_rollout_ref_a = deepcopy(self.config.actor_rollout_ref_a)
            cfg.actor_rollout_ref_b = deepcopy(self.config.actor_rollout_ref_b)

        return cfg

    def _actor_cfg(self, which: str):
        return self.cfg_a if which == "a" else self.cfg_b

    def _actor_wg(self, which: str):
        return self.actor_rollout_wg_a if which == "a" else self.actor_rollout_wg_b

    def _async_mgr(self, which: str):
        return self.async_rollout_manager_a if which == "a" else self.async_rollout_manager_b

    def _ckpt_mgr(self, which: str):
        return self.checkpoint_manager_a if which == "a" else self.checkpoint_manager_b

    # -------------------------
    # Data shaping utils
    # -------------------------
    def _get_gen_batch(self, batch: DataProto) -> DataProto:
        reward_model_keys = set({"data_source", "reward_model", "extra_info", "uid"}) & batch.non_tensor_batch.keys()
        non_tensor_batch_keys_to_pop = set(batch.non_tensor_batch.keys()) - reward_model_keys
        gen_batch = batch.pop(batch_keys=[], non_tensor_batch_keys=list(non_tensor_batch_keys_to_pop))
        if self.async_rollout_mode:
            gen_batch.non_tensor_batch.update(batch.non_tensor_batch)
        return gen_batch

    def _maybe_log_val_generations(self, inputs, outputs, scores):
        if self.config.trainer.log_val_generations == 0:
            return
        samples = list(zip(inputs, outputs, scores, strict=True))
        samples.sort(key=lambda x: x[0])
        rng = np.random.RandomState(42)
        rng.shuffle(samples)
        samples = samples[: self.config.trainer.log_val_generations]
        self.validation_generations_logger.log(self.config.trainer.logger, samples, self.global_steps)

    def _dump_generations(self, inputs, outputs, gts, scores, reward_extra_infos_dict, dump_path):
        os.makedirs(dump_path, exist_ok=True)
        filename = os.path.join(dump_path, f"{self.global_steps}.jsonl")
        n = len(inputs)
        base_data = {"input": inputs, "output": outputs, "gts": gts, "score": scores, "step": [self.global_steps] * n}
        for k, v in reward_extra_infos_dict.items():
            if len(v) == n:
                base_data[k] = v
        lines = []
        for i in range(n):
            entry = {k: v[i] for k, v in base_data.items()}
            lines.append(json.dumps(entry, ensure_ascii=False))
        with open(filename, "w") as f:
            f.write("\n".join(lines) + "\n")
        print(f"Dumped generations to {filename}")

    # -------------------------
    # Core compute methods (actor-scoped)
    # -------------------------
    def _compute_old_log_prob(self, which: str, batch: DataProto):
        actor_wg = self._actor_wg(which)
        batch_td = batch.to_tensordict()
        batch_td = left_right_2_no_padding(batch_td)
        tu.assign_non_tensor(batch_td, calculate_entropy=True, compute_loss=False)

        batch_td = attach_global_token_num(batch_td)
 
        #debug
        # td = batch_td
        # print("[debug] keys:", list(td.keys()))
        # print("[debug] has global_token_num?", "global_token_num" in td.keys())
        # print("[debug] non_tensor keys:", list(tu.get_non_tensor(td).keys()) if hasattr(tu, "get_non_tensor") else "n/a")

        output = actor_wg.compute_log_prob(batch_td)
        entropy = tu.get(output, "entropy")
        log_probs = tu.get(output, "log_probs")
        # old_log_prob_mfu = tu.get(output, "metrics")["mfu"]
        # metrics = output.get("metrics", {}) or {}
        # old_log_prob_mfu = float(metrics.get("mfu", 0.0))
        metrics = tu.get(output, "metrics", default={})
        old_log_prob_mfu = metrics.get("mfu", None)
        entropy = no_padding_2_padding(entropy, batch_td)
        log_probs = no_padding_2_padding(log_probs, batch_td)
        old_log_prob = tu.get_tensordict({"old_log_probs": log_probs.float(), "entropys": entropy.float()})
        return DataProto.from_tensordict(old_log_prob), old_log_prob_mfu

    def _compute_values(self, batch: DataProto) -> DataProto:
        assert self.use_critic
        batch_td = batch.to_tensordict()
        batch_td = left_right_2_no_padding(batch_td)
        tu.assign_non_tensor(batch_td, compute_loss=False)
        output = self.critic_wg.infer_batch(batch_td).get()
        values = tu.get(output, "values")
        values = no_padding_2_padding(values, batch_td)
        values = tu.get_tensordict({"values": values.float()})
        return DataProto.from_tensordict(values)

    def _compute_ref_log_prob_via_actor(self, which: str, batch: DataProto) -> DataProto:
        actor_wg = self._actor_wg(which)
        batch_td = batch.to_tensordict()
        batch_td = left_right_2_no_padding(batch_td)
        tu.assign_non_tensor(batch_td, calculate_entropy=False, compute_loss=False, no_lora_adapter=True)
        output = actor_wg.compute_log_prob(batch_td)
        log_probs = tu.get(output, "log_probs")
        log_probs = no_padding_2_padding(log_probs, batch_td)
        ref_log_prob = tu.get_tensordict({"ref_log_prob": log_probs.float()})
        return DataProto.from_tensordict(ref_log_prob)

    def _update_actor(self, which: str, batch: DataProto) -> DataProto:
        actor_cfg = self._actor_cfg(which)
        actor_wg = self._actor_wg(which)

        rollout_cfg = actor_cfg.rollout
        batch.meta_info["multi_turn"] = rollout_cfg.multi_turn.enable
        batch.meta_info["temperature"] = rollout_cfg.temperature

        batch_td = batch.to_tensordict()
        batch_td = left_right_2_no_padding(batch_td)

        # print(f"[debug][ppo_loss] missing ref_log_prob. keys={list(batch_td.keys())}")

        calculate_entropy = actor_cfg.actor.entropy_coeff != 0.0
        ppo_mini_batch_size = actor_cfg.actor.ppo_mini_batch_size * actor_cfg.rollout.n
        ppo_epochs = actor_cfg.actor.ppo_epochs
        seed = actor_cfg.actor.data_loader_seed
        shuffle = actor_cfg.actor.shuffle

        tu.assign_non_tensor(
            batch_td,
            calculate_entropy=calculate_entropy,
            global_batch_size=ppo_mini_batch_size,
            mini_batch_size=ppo_mini_batch_size,
            epochs=ppo_epochs,
            seed=seed,
            dataloader_kwargs={"shuffle": shuffle},
        )

        actor_output = actor_wg.update_actor(batch_td)
        actor_output = tu.get(actor_output, "metrics")
        actor_output = rename_dict(actor_output, f"actor_{which}/")
        actor_output[f"perf/mfu/actor_{which}"] = actor_output.pop(f"actor_{which}/mfu")
        return DataProto.from_single_dict(data={}, meta_info={"metrics": actor_output})

    def _update_critic(self, batch: DataProto) -> DataProto:
        assert self.use_critic
        batch_td = batch.to_tensordict()
        batch_td = left_right_2_no_padding(batch_td)

        # NOTE: keep parity with your original: scale by cfg_a.rollout.n
        # If you want strict correctness under n_a != n_b, split critic update per batch size explicitly.
        ppo_mini_batch_size = self.config.critic.ppo_mini_batch_size * self.cfg_a.rollout.n
        ppo_epochs = self.config.critic.ppo_epochs
        seed = self.config.critic.data_loader_seed
        shuffle = self.config.critic.shuffle

        tu.assign_non_tensor(
            batch_td,
            global_batch_size=ppo_mini_batch_size,
            mini_batch_size=ppo_mini_batch_size,
            epochs=ppo_epochs,
            seed=seed,
            dataloader_kwargs={"shuffle": shuffle},
        )
        output = self.critic_wg.train_mini_batch(batch_td).get()
        output = tu.get(output, "metrics")
        output = rename_dict(output, "critic/")
        output["perf/mfu/critic"] = output.pop("critic/mfu")
        return DataProto.from_single_dict(data={}, meta_info={"metrics": output})

    # -------------------------
    # Balance batch
    # -------------------------
    def _get_dp_size(self, worker_group, role: str) -> int:
        if role not in worker_group._dispatch_info:
            dp_rank_mapping = worker_group._query_dispatch_info(role)
            worker_group._dispatch_info[role] = dp_rank_mapping
        else:
            dp_rank_mapping = worker_group._dispatch_info[role]
        return max(dp_rank_mapping) + 1

    def _balance_batch(self, which: str, batch: DataProto, metrics, logging_prefix="global_seqlen", keep_minibatch=False):
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = attention_mask.view(batch_size, -1).sum(-1)
        workload_lst = calculate_workload(global_seqlen_lst)

        # FIX: use correct dispatch key
        role_key = str(Role.ActorRolloutA) if which == "a" else str(Role.ActorRolloutB)
        dp_size = self._get_dp_size(self._actor_wg(which), role_key)

        if getattr(self, "use_prefix_grouper", False) and "uid" in batch.non_tensor_batch:
            from verl.utils.seqlen_balancing import get_group_balanced_partitions

            uid_list = list(batch.non_tensor_batch["uid"])
            seqlen_list = global_seqlen_lst.tolist()
            num_groups = len(set(uid_list))
            if num_groups % dp_size != 0:
                raise ValueError(f"PrefixGrouper requires num_uid_groups % dp_size == 0 ({num_groups}%{dp_size})")
            global_partition_lst = get_group_balanced_partitions(seqlen_list=seqlen_list, uid_list=uid_list, k_partitions=dp_size)
        elif keep_minibatch:
            minibatch_size = self._actor_cfg(which).actor.get("ppo_mini_batch_size")
            minibatch_num = len(workload_lst) // minibatch_size
            global_partition_lst = [[] for _ in range(dp_size)]
            for i in range(minibatch_num):
                rearrange_minibatch_lst = get_seqlen_balanced_partitions(
                    workload_lst[i * minibatch_size : (i + 1) * minibatch_size],
                    k_partitions=dp_size,
                    equal_size=True,
                )
                for j, part in enumerate(rearrange_minibatch_lst):
                    global_partition_lst[j].extend([x + minibatch_size * i for x in part])
        else:
            global_partition_lst = get_seqlen_balanced_partitions(workload_lst, k_partitions=dp_size, equal_size=True)

        if not getattr(self, "use_prefix_grouper", False):
            for idx, partition in enumerate(global_partition_lst):
                partition.sort(key=lambda x: (workload_lst[x], x))
                global_partition_lst[idx] = partition[::2] + partition[1::2][::-1]

        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        stats = log_seqlen_unbalance(
            seqlen_list=global_seqlen_lst.tolist(),
            partitions=global_partition_lst,
            prefix=logging_prefix,
        )
        metrics.update(stats)

    # -------------------------
    # Reward compute/extract
    # -------------------------
    def _compute_or_extract_reward(self, batch: DataProto, reward_fn=None, reward_for_val: bool = False, sum_reward: bool = False):
        if "rm_scores" in batch.batch.keys():
            reward_tensor = batch.batch["rm_scores"]
            if sum_reward:
                reward_tensor = reward_tensor.sum(dim=-1)

            if not reward_for_val and sum_reward:
                return reward_tensor

            reward_extra_keys = batch.meta_info.get("reward_extra_keys", [])
            reward_extra_infos_dict = ({key: batch.non_tensor_batch[key] for key in reward_extra_keys} if reward_extra_keys else {})
            return reward_tensor, reward_extra_infos_dict

        if reward_fn is None:
            raise ValueError("reward_fn must be provided when rm_scores is not available.")

        if reward_for_val:
            result = reward_fn(batch, return_dict=True)
            reward_tensor = result["reward_tensor"]
            if sum_reward:
                reward_tensor = reward_tensor.sum(dim=-1)
            reward_extra_infos_dict = result.get("reward_extra_info", {})
            return reward_tensor, reward_extra_infos_dict

        reward_tensor, reward_extra_infos_dict = compute_reward(batch, reward_fn)
        if sum_reward:
            reward_tensor = reward_tensor.sum(dim=-1)
        return reward_tensor, reward_extra_infos_dict

    # -------------------------
    # Checkpoint save/load
    # -------------------------
    def _save_checkpoint(self):
        from verl.utils.fs import local_mkdir_safe

        local_step_dir = os.path.join(self.config.trainer.default_local_dir, f"global_step_{self.global_steps}")
        local_mkdir_safe(local_step_dir)

        def _save_actor(which: str):
            actor_dir_name = f"actor_{which}"
            actor_local = os.path.join(local_step_dir, actor_dir_name)
            actor_remote = (
                None
                if self.config.trainer.default_hdfs_dir is None
                else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", actor_dir_name)
            )
            max_keep = self.config.trainer.get("max_actor_ckpt_to_keep", None)
            self._actor_wg(which).save_checkpoint(actor_local, actor_remote, self.global_steps, max_ckpt_to_keep=max_keep)

        _save_actor("a")
        _save_actor("b")

        if self.use_critic:
            critic_local = os.path.join(local_step_dir, str(Role.Critic))
            critic_remote = (
                None
                if self.config.trainer.default_hdfs_dir is None
                else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", str(Role.Critic))
            )
            max_keep = self.config.trainer.get("max_critic_ckpt_to_keep", None)
            self.critic_wg.save_checkpoint(critic_local, critic_remote, self.global_steps, max_ckpt_to_keep=max_keep)

        dataloader_local = os.path.join(local_step_dir, "data.pt")
        torch.save(self.train_dataloader.state_dict(), dataloader_local)

        local_latest = os.path.join(self.config.trainer.default_local_dir, "latest_checkpointed_iteration.txt")
        with open(local_latest, "w") as f:
            f.write(str(self.global_steps))

    def _load_checkpoint(self):
        if self.config.trainer.resume_mode == "disable":
            self.global_steps = 0
            return

        if self.config.trainer.default_hdfs_dir is not None:
            raise NotImplementedError("Load from HDFS not implemented in this dual trainer.")
        checkpoint_folder = self.config.trainer.default_local_dir
        if not os.path.isabs(checkpoint_folder):
            checkpoint_folder = os.path.join(os.getcwd(), checkpoint_folder)
        global_step_folder = find_latest_ckpt_path(checkpoint_folder)

        if self.config.trainer.resume_mode == "auto":
            if global_step_folder is None:
                self.global_steps = 0
                return
        elif self.config.trainer.resume_mode == "resume_path":
            global_step_folder = self.config.trainer.resume_from_path
            if not os.path.isabs(global_step_folder):
                global_step_folder = os.path.join(os.getcwd(), global_step_folder)

        self.global_steps = int(global_step_folder.split("global_step_")[-1])
        print(f"Resuming from {global_step_folder}, global_steps={self.global_steps}")

        def _load_actor(which: str):
            actor_path = os.path.join(global_step_folder, f"actor_{which}")
            self._actor_wg(which).load_checkpoint(
                actor_path,
                del_local_after_load=self.config.trainer.del_local_ckpt_after_load,
            )

        _load_actor("a")
        _load_actor("b")

        if self.use_critic:
            critic_path = os.path.join(global_step_folder, str(Role.Critic))
            self.critic_wg.load_checkpoint(critic_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load)

        dataloader_local = os.path.join(global_step_folder, "data.pt")
        if os.path.exists(dataloader_local):
            self.train_dataloader.load_state_dict(torch.load(dataloader_local, weights_only=False))

    # -------------------------
    # Validation
    # -------------------------
    def _validate_one(self, which: str):
        actor_cfg = self._actor_cfg(which)
        async_mgr = self._async_mgr(which)

        data_source_lst = []
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)
        sample_inputs, sample_outputs, sample_gts, sample_scores = [], [], [], []
        sample_turns, sample_uids = [], []

        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)
            if "uid" not in test_batch.non_tensor_batch:
                test_batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(test_batch.batch))], dtype=object
                )

            test_batch = test_batch.repeat(repeat_times=actor_cfg.rollout.val_kwargs.n, interleave=True)

            if self.config.reward_model.enable and test_batch[0].non_tensor_batch["reward_model"]["style"] == "model":
                return {}

            ground_truths = [item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in test_batch]
            sample_gts.extend(ground_truths)

            test_gen_batch = self._get_gen_batch(test_batch)
            test_gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": actor_cfg.rollout.val_kwargs.do_sample,
                "validate": True,
                "global_steps": self.global_steps,
            }

            size_divisor = actor_cfg.rollout.agent.num_workers
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, size_divisor)

            test_output_gen_batch_padded = async_mgr.generate_sequences(test_gen_batch_padded)
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)

            output_ids = test_output_gen_batch.batch["responses"]
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            sample_outputs.extend(output_texts)

            test_batch = test_batch.union(test_output_gen_batch)
            test_batch.meta_info["validate"] = True

            input_ids = test_batch.batch["prompts"]
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            sample_inputs.extend(input_texts)
            sample_uids.extend(test_batch.non_tensor_batch["uid"])

            reward_tensor, reward_extra_info = self._compute_or_extract_reward(
                test_batch, reward_fn=self.val_reward_fn, reward_for_val=True
            )
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)

            reward_extra_infos_dict["reward"].extend(scores)
            for key, values in reward_extra_info.items():
                if isinstance(values, np.ndarray):
                    reward_extra_infos_dict[key].extend(values.tolist())
                else:
                    reward_extra_infos_dict[key].extend(values if isinstance(values, list) else [values])

            if "__num_turns__" in test_batch.non_tensor_batch:
                sample_turns.append(test_batch.non_tensor_batch["__num_turns__"])

            data_source_lst.append(test_batch.non_tensor_batch.get("data_source", ["unknown"] * reward_tensor.shape[0]))

        self._maybe_log_val_generations(sample_inputs, sample_outputs, sample_scores)

        val_data_dir = self.config.trainer.get("validation_data_dir", None)
        if val_data_dir:
            self._dump_generations(sample_inputs, sample_outputs, sample_gts, sample_scores, reward_extra_infos_dict, val_data_dir)

        data_sources = np.concatenate(data_source_lst, axis=0)
        return self._val_metrics_update(data_sources, sample_uids, reward_extra_infos_dict, sample_turns, which)

    def _val_metrics_update(self, data_sources, sample_uids, reward_extra_infos_dict, sample_turns, which: str):
        data_src2var2metric2val = process_validation_metrics(data_sources, sample_uids, reward_extra_infos_dict)
        metric_dict = {}
        for data_source, var2metric2val in data_src2var2metric2val.items():
            core_var = "acc" if "acc" in var2metric2val else "reward"
            for var_name, metric2val in var2metric2val.items():
                n_max = max([int(name.split("@")[-1].split("/")[0]) for name in metric2val.keys()])
                for metric_name, metric_val in metric2val.items():
                    metric_sec = (
                        "val-core"
                        if ((var_name == core_var) and any(metric_name.startswith(pfx) for pfx in ["mean", "maj", "best"]) and (f"@{n_max}" in metric_name))
                        else "val-aux"
                    )
                    metric_dict[f"{metric_sec}/{which}/{data_source}/{var_name}/{metric_name}"] = metric_val

        if len(sample_turns) > 0:
            sample_turns = np.concatenate(sample_turns)
            metric_dict[f"val-aux/{which}/num_turns/min"] = sample_turns.min()
            metric_dict[f"val-aux/{which}/num_turns/max"] = sample_turns.max()
            metric_dict[f"val-aux/{which}/num_turns/mean"] = sample_turns.mean()
        return metric_dict

    # -------------------------
    # Debug: print longest response in a batch (after rollout) for comparing generate length
    # -------------------------
    def _print_longest_response(
        self, label: str, data: DataProto, step: int, head_tokens: int = 20, tail_tokens: int = 20
    ):
        """Print the longest response in data (by response_mask), truncated to head+tail for readability."""
        batch = data.batch
        if "responses" not in batch:
            print(f"[LONGEST_RESP] {label} step={step} (no 'responses' in batch, keys={list(batch.keys())})", flush=True)
            return
        responses = batch["responses"]
        response_mask = batch.get("response_mask")
        pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
        if response_mask is None:
            response_mask = (responses != pad_id).long()
        lengths = response_mask.sum(dim=1)
        if lengths.max().item() == 0:
            print(f"[LONGEST_RESP] {label} step={step} response_len=0 (all empty)", flush=True)
            return
        idx = lengths.argmax().item()
        len_i = int(lengths[idx].item())
        row = responses[idx]
        ids = row[response_mask[idx].bool()].tolist()
        if not ids:
            ids = row[:len_i].tolist()
        if len(ids) <= head_tokens + tail_tokens:
            text = self.tokenizer.decode(ids, skip_special_tokens=True)
        else:
            head_ids = ids[:head_tokens]
            tail_ids = ids[-tail_tokens:]
            head_text = self.tokenizer.decode(head_ids, skip_special_tokens=True)
            tail_text = self.tokenizer.decode(tail_ids, skip_special_tokens=True)
            text = f"{head_text}\n... [truncated, {len(ids) - head_tokens - tail_tokens} tokens] ...\n{tail_text}"
        print(f"[LONGEST_RESP] {label} step={step} response_len={len_i} idx={idx}", flush=True)
        print(f"[LONGEST_RESP] {label} step={step} text:\n{text}", flush=True)
        print(f"[LONGEST_RESP] {label} step={step} --- end ---", flush=True)

    # -------------------------
    # One actor step
    # -------------------------
    def _step_one_actor(self, which: str, batch: DataProto, timing_raw: dict, metrics: dict, gen_batch_output_override: Optional[DataProto] = None):
        actor_cfg = self._actor_cfg(which)
        async_mgr = self._async_mgr(which)
        ckpt_mgr = self._ckpt_mgr(which)

        if gen_batch_output_override is None:
            batch.meta_info["temperature"] = actor_cfg.rollout.temperature
            batch.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object)

            gen_batch = self._get_gen_batch(batch)
            gen_batch.meta_info["global_steps"] = self.global_steps
            gen_batch_output = gen_batch.repeat(repeat_times=actor_cfg.rollout.n, interleave=True)

            with marked_timer(f"gen_{which}", timing_raw, color="red"):
                t_rollout_start = time.perf_counter()
                gen_batch_output = async_mgr.generate_sequences(gen_batch_output)
                t_rollout_end = time.perf_counter()
                ckpt_mgr.sleep_replicas()
                timing_raw.update(gen_batch_output.meta_info.get("timing", {}))
                gen_batch_output.meta_info.pop("timing", None)

            print(f"[ROLLOUT] wall_time={t_rollout_end - t_rollout_start:.3f}s which={which} step={self.global_steps}", flush=True)
        else:
            t0 = time.time()
            gen_batch_output = gen_batch_output_override
            ckpt_mgr.sleep_replicas()
            timing_raw[f"gen_{which}"] = time.time() - t0
            timing_raw.update(gen_batch_output.meta_info.get("timing", {}))
            gen_batch_output.meta_info.pop("timing", None)

            rtprint(f"[rt] step={self.global_steps} gen_{which} override-done: {timing_raw[f'gen_{which}']:.2f}s")

        if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
            if gen_batch_output_override is not None:
                raise NotImplementedError("REMAX with gen_batch_output_override is not supported")
            if self.reward_fn is None:
                raise ValueError("A reward_fn is required for REMAX advantage estimation.")
            with marked_timer(f"gen_max_{which}", timing_raw, color="purple"):
                gen_baseline_batch = deepcopy(gen_batch)
                gen_baseline_batch.meta_info["do_sample"] = False
                gen_baseline_output = async_mgr.generate_sequences(gen_baseline_batch)
                ckpt_mgr.sleep_replicas()
                batch = batch.union(gen_baseline_output)

                rm_scores = None
                if self.use_rm and "rm_scores" not in batch.batch.keys():
                    assert self.reward_loop_manager is not None, "RewardLoopManager is None"
                    rm_scores = self.reward_loop_manager.compute_rm_score(batch)
                    batch = batch.union(rm_scores)

                reward_baseline_tensor = self._compute_or_extract_reward(batch, reward_fn=self.reward_fn, sum_reward=True)

                keys_to_pop = set(gen_baseline_output.batch.keys())
                if rm_scores is not None:
                    keys_to_pop.update(rm_scores.batch.keys())
                batch.pop(batch_keys=list(keys_to_pop))
                batch.batch["reward_baselines"] = reward_baseline_tensor

        batch = batch.repeat(repeat_times=actor_cfg.rollout.n, interleave=True)
        # In override mode, generation output may carry stale non-tensor keys (e.g. uid)
        # from an external pre-gen path. Keep batch-owned keys authoritative to avoid
        # DataProto.union equality assertion on duplicated keys.
        if gen_batch_output_override is not None and gen_batch_output.non_tensor_batch:
            overlap_non_tensor_keys = set(batch.non_tensor_batch.keys()) & set(gen_batch_output.non_tensor_batch.keys())
            for k in overlap_non_tensor_keys:
                gen_batch_output.non_tensor_batch.pop(k, None)
        batch = batch.union(gen_batch_output)

        if "response_mask" not in batch.batch.keys():
            batch.batch["response_mask"] = compute_response_mask(batch)

        # 方案 B: 每个 step 将本 batch 的 decoding length 追加到 CSV（multi turn / dual 通用）
        # multi-turn 时若 batch 带 __num_turns__，则多写一列 turn
        decoding_length_csv_dir = self.config.trainer.get("decoding_length_csv_dir", None)
        if decoding_length_csv_dir is not None:
            os.makedirs(decoding_length_csv_dir, exist_ok=True)
            lengths = batch.batch["response_mask"].sum(dim=-1).cpu().tolist()
            exp_name = getattr(self.config.trainer, "experiment_name", "run")
            safe_name = "".join(c if c.isalnum() or c in "._-" else "_" for c in str(exp_name))
            csv_path = os.path.join(decoding_length_csv_dir, f"lengths_{safe_name}_{which}.csv")
            num_turns = batch.non_tensor_batch.get("__num_turns__", None)
            write_header = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0
            with open(csv_path, "a", newline="") as f:
                if num_turns is not None:
                    turns = np.asarray(num_turns).ravel()
                    if write_header:
                        f.write("length,turn\n")
                    for L, t in zip(lengths, turns, strict=True):
                        f.write(f"{L},{int(t)}\n")
                else:
                    for L in lengths:
                        f.write(f"{L}\n")

        assert not self.config.trainer.balance_batch

        reward_extra_infos_dict = {}
        with marked_timer(f"reward_{which}", timing_raw, color="yellow"):
            if self.use_rm and "rm_scores" not in batch.batch.keys():
                assert self.reward_loop_manager is not None, "RewardLoopManager is None"
                reward_tensor = self.reward_loop_manager.compute_rm_score(batch)
                batch = batch.union(reward_tensor)

            if self.config.reward_model.launch_reward_fn_async:
                future_reward = compute_reward_async.remote(data=batch, config=self.config, tokenizer=self.tokenizer)
            else:
                reward_tensor, reward_extra_infos_dict = self._compute_or_extract_reward(batch, reward_fn=self.reward_fn, reward_for_val=False)
                future_reward = None

        with marked_timer(f"old_log_prob_{which}", timing_raw, color="blue"):
            old_log_prob, old_log_prob_mfu = self._compute_old_log_prob(which, batch)
            entropys = old_log_prob.batch["entropys"]
            response_masks = batch.batch["response_mask"]
            entropy_agg = agg_loss(
                loss_mat=entropys,
                loss_mask=response_masks,
                loss_agg_mode=actor_cfg.actor.loss_agg_mode,
                loss_scale_factor=actor_cfg.actor.loss_scale_factor,
            )
            metrics.update(
                {
                    f"actor_{which}/entropy": float(entropy_agg.detach().item()),
                    f"perf/mfu/actor_{which}_infer": float(old_log_prob_mfu),
                    # f"perf/mfu/actor_{which}_infer": float(old_log_prob_mfu or 0.0),
                }
            )
            old_log_prob.batch.pop("entropys")
            batch = batch.union(old_log_prob)

        if self.config.algorithm.use_kl_in_reward:
            with marked_timer(f"ref_log_prob_{which}", timing_raw, color="olive"):
                ref_log_prob = self._compute_ref_log_prob_via_actor(which, batch)
                batch = batch.union(ref_log_prob)

        if self.use_critic:
            with marked_timer(f"values_{which}", timing_raw, color="cyan"):
                values = self._compute_values(batch)
                batch = batch.union(values)

        with marked_timer(f"adv_{which}", timing_raw, color="brown"):
            if future_reward is not None:
                reward_tensor, reward_extra_infos_dict = ray.get(future_reward)

            batch.batch["token_level_scores"] = reward_tensor
            if reward_extra_infos_dict:
                batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

            if self.config.algorithm.use_kl_in_reward:
                batch, kl_metrics = apply_kl_penalty(batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty)
                metrics.update({f"{k}_{which}": v for k, v in kl_metrics.items()})
            else:
                batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

            norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)
            batch = compute_advantage(
                batch,
                adv_estimator=self.config.algorithm.adv_estimator,
                gamma=self.config.algorithm.gamma,
                lam=self.config.algorithm.lam,
                num_repeat=actor_cfg.rollout.n,
                norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                config=self.config.algorithm,
            )

        # 在 _step_one_actor 末尾 return 前加：
        if getattr(self, "profiler", None) is not None and self.profiler.enabled:
            if self.profiler.verbose and self.profiler.print_rollout_each:
                print(f"Rollout for {which} successfully finished")
                # 只打印该 actor 本次 rollout 的关键段
                keys = [f"gen_{which}", f"reward_{which}", f"old_log_prob_{which}",
                        f"ref_log_prob_{which}", f"values_{which}", f"adv_{which}"]
                msg = [f"[rollout-profile] which={which} global_step={self.global_steps}"]
                for k in keys:
                    if k in timing_raw:
                        msg.append(f"{k}={timing_raw[k]*1000:.1f}ms")
                print(" | ".join(msg))

        # ----- rollout length stats (min/max/mean) -----
        stats = rollout_token_stats_from_batch(batch, tokenizer=self.tokenizer)
        print("[debug stats]", stats)
        if stats:
            # 写到 metrics 里（带 actor 前缀）
            for k, v in stats.items():
                metrics[f"{which}/{k}"] = v   # 例如 a/rollout_tokens/min

            # 同时按需打印（跟 profiler 行一起看很直观）
            if getattr(self, "profiler", None) is not None and self.profiler.enabled:
                if self.profiler.verbose and self.profiler.print_rollout_each:
                    print(
                        f"[rollout-len] {which} step={self.global_steps} "
                        f"n={stats['rollout_tokens/n']} "
                        f"min={stats['rollout_tokens/min']} "
                        f"mean={stats['rollout_tokens/mean']:.1f} "
                        f"p90={stats['rollout_tokens/p90']:.1f} "
                        f"max={stats['rollout_tokens/max']}"
                    )
    
        # ensure global_token_num exists for throughput metrics
        if "global_token_num" not in batch.meta_info:
            if "attention_mask" in batch.batch:
                batch.meta_info["global_token_num"] = batch.batch["attention_mask"].sum(dim=-1).to(torch.int).tolist()
            else:
                pad_id = self.tokenizer.pad_token_id or 0
                batch.meta_info["global_token_num"] = (batch.batch["input_ids"] != pad_id).sum(dim=-1).to(torch.int).tolist()

        return batch, reward_extra_infos_dict

    # -------------------------
    # Fit
    # -------------------------
    def fit(self):
        from verl.utils.tracking import Tracking
        _ts = lambda: time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )
        fit_wall_start = time.perf_counter()
        print(f"[FIT_DUAL] wall_start={_ts()}", flush=True)

        self.global_steps = 0
        self._load_checkpoint()

        self.checkpoint_manager_a.update_weights()
        self.checkpoint_manager_b.update_weights()

        if self.cfg_a.rollout.get("skip_rollout", False):
            RolloutSkip(self._build_actor_scoped_config("a"), self.actor_rollout_wg_a).wrap_generate_sequences()
        if self.cfg_b.rollout.get("skip_rollout", False):
            RolloutSkip(self._build_actor_scoped_config("b"), self.actor_rollout_wg_b).wrap_generate_sequences()

        current_epoch = self.global_steps // len(self.train_dataloader) if len(self.train_dataloader) > 0 else 0

        # if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
        #     val_a = self._validate_one("a")
        #     val_b = self._validate_one("b")
        #     # logger.log(data=val_a, step=self.global_steps)
        #     # logger.log(data=val_b, step=self.global_steps)
        #     if self.config.trainer.get("val_only", False):
        #         return

        # 从 config 里读（你也可以放 trainer.profile.*）
        prof_cfg = self.config.trainer.get("profile", {})
        # print(f"[debug profiler] {prof_cfg.get("enabled", True)}, {prof_cfg.get("verbose", True)}, {prof_cfg.get("print_every", 1)}")
        self.profiler = StepProfiler(
            enabled=prof_cfg.get("enabled", True),
            verbose=prof_cfg.get("verbose", True),
            print_every=prof_cfg.get("print_every", 1),
            window=prof_cfg.get("window", 1),
            print_rollout_each=prof_cfg.get("print_rollout_each", False),
        )

        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Dual Training Progress")

        self.global_steps += 1
        last_val_metrics = None
        self.max_steps_duration = 0

        for epoch in range(current_epoch, self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                rtprint(f"[{_ts()}] [FIT] epoch={epoch} step={self.global_steps} begin")
                metrics = {}
                timing_raw = {}

                is_last_step = self.global_steps >= self.total_training_steps

                with marked_timer("step", timing_raw):
                    batch_a, batch_b = DataProto.from_single_dict_two_copies(batch_dict)
                    t_rollout_a0 = time.perf_counter()
                    print(f"[FIT] rollout_a walltime_before={t_rollout_a0:.6f} step={self.global_steps}", flush=True)
                    batch_a, _ = self._step_one_actor("a", batch_a, timing_raw, metrics)
                    t_rollout_a1 = time.perf_counter()
                    print(f"[FIT] rollout_a walltime_after={t_rollout_a1:.6f} duration={t_rollout_a1 - t_rollout_a0:.3f}s step={self.global_steps}", flush=True)
                    self._print_longest_response("FIT_rollout_a", batch_a, self.global_steps)
                    t_rollout_b0 = time.perf_counter()
                    print(f"[FIT] rollout_b walltime_before={t_rollout_b0:.6f} step={self.global_steps}", flush=True)
                    batch_b, _ = self._step_one_actor("b", batch_b, timing_raw, metrics)
                    t_rollout_b1 = time.perf_counter()
                    print(f"[FIT] rollout_b walltime_after={t_rollout_b1:.6f} duration={t_rollout_b1 - t_rollout_b0:.3f}s step={self.global_steps}", flush=True)
                    self._print_longest_response("FIT_rollout_b", batch_b, self.global_steps)

                    if self.use_critic:
                        with marked_timer("update_critic_a", timing_raw, color="pink"):
                            critic_out_a = self._update_critic(batch_a)
                        metrics.update(reduce_metrics(critic_out_a.meta_info["metrics"]))

                        with marked_timer("update_critic_b", timing_raw, color="pink"):
                            critic_out_b = self._update_critic(batch_b)
                        metrics.update(reduce_metrics(critic_out_b.meta_info["metrics"]))

                    if self.config.trainer.critic_warmup <= self.global_steps:
                        with marked_timer("update_actor_a", timing_raw, color="red"):
                            actor_out_a = self._update_actor("a", batch_a)
                        metrics.update(reduce_metrics(actor_out_a.meta_info["metrics"]))

                        with marked_timer("update_actor_b", timing_raw, color="red"):
                            actor_out_b = self._update_actor("b", batch_b)
                        metrics.update(reduce_metrics(actor_out_b.meta_info["metrics"]))

                        esi_close_to_expiration = should_save_ckpt_esi(
                            max_steps_duration=self.max_steps_duration,
                            redundant_time=self.config.trainer.esi_redundant_time,
                        )
                        if self.config.trainer.save_freq > 0 and (
                            self.global_steps % self.config.trainer.save_freq == 0
                            or esi_close_to_expiration
                        ):
                            with marked_timer("save_checkpoint", timing_raw, color="green"):
                                self._save_checkpoint()

                        with marked_timer("update_weights_a", timing_raw, color="red"):
                            self.checkpoint_manager_a.update_weights()
                        with marked_timer("update_weights_b", timing_raw, color="red"):
                            self.checkpoint_manager_b.update_weights()

                if (
                    self.val_reward_fn is not None
                    and self.config.trainer.test_freq > 0
                    and (self.global_steps % self.config.trainer.test_freq == 0)
                ):
                    with marked_timer("testing", timing_raw, color="green"):
                        val_a = self._validate_one("a")
                        val_b = self._validate_one("b")
                        metrics.update(val_a)
                        metrics.update(val_b)
                        if is_last_step:
                            last_val_metrics = {"a": val_a, "b": val_b}

                steps_duration = timing_raw.get("step", 0.0)
                self.max_steps_duration = max(self.max_steps_duration, steps_duration)

                metrics.update({"training/global_step": self.global_steps, "training/epoch": epoch})

                metrics.update(rename_dict(compute_data_metrics(batch=batch_a, use_critic=self.use_critic), "data_a/"))
                metrics.update(rename_dict(compute_data_metrics(batch=batch_b, use_critic=self.use_critic), "data_b/"))

                metrics.update(compute_timing_metrics(batch=batch_a, timing_raw=timing_raw))
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch_a, timing_raw=timing_raw, n_gpus=n_gpus))

                gradient_norm = metrics.get("actor_a/grad_norm", None) or metrics.get("actor_b/grad_norm", None)
                metrics.update(compute_variance_proxy_metrics(batch=batch_a, gradient_norm=gradient_norm))

                # logger.log(data=metrics, step=self.global_steps)

                if getattr(self, "profiler", None) is not None:
                    rtprint(f"[{_ts()}] [FIT] before profiler.update step={self.global_steps}")
                    self.profiler.update(timing_raw=timing_raw, step=self.global_steps)
                    rtprint(f"[{_ts()}] [FIT] after profiler.update step={self.global_steps}")

                progress_bar.update(1)
                self.global_steps += 1

                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    print(f"[FIT_DUAL] wall_total={time.perf_counter() - fit_wall_start:.3f}s (completed)", flush=True)
                    return
        print(f"[FIT_DUAL] wall_total={time.perf_counter() - fit_wall_start:.3f}s (epoch_end)", flush=True)

    def fit_naive_concurrent_rollout(self):
        from verl.utils.tracking import Tracking
        _ts = lambda: time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0
        self._load_checkpoint()

        self.checkpoint_manager_a.update_weights()
        self.checkpoint_manager_b.update_weights()

        if self.cfg_a.rollout.get("skip_rollout", False):
            RolloutSkip(self._build_actor_scoped_config("a"), self.actor_rollout_wg_a).wrap_generate_sequences()
        if self.cfg_b.rollout.get("skip_rollout", False):
            RolloutSkip(self._build_actor_scoped_config("b"), self.actor_rollout_wg_b).wrap_generate_sequences()

        current_epoch = self.global_steps // len(self.train_dataloader) if len(self.train_dataloader) > 0 else 0

        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_a = self._validate_one("a")
            val_b = self._validate_one("b")
            # logger.log(data=val_a, step=self.global_steps)
            # logger.log(data=val_b, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        # 从 config 里读（你也可以放 trainer.profile.*）
        prof_cfg = self.config.trainer.get("profile", {})
        self.profiler = StepProfiler(
            enabled=prof_cfg.get("enabled", True),
            verbose=prof_cfg.get("verbose", True),
            print_every=prof_cfg.get("print_every", 1),
            window=prof_cfg.get("window", 1),
            print_rollout_each=prof_cfg.get("print_rollout_each", False),
        )

        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Dual Training Progress")

        self.global_steps += 1
        last_val_metrics = None
        self.max_steps_duration = 0

        for epoch in range(current_epoch, self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                rtprint(f"[{_ts()}] [FIT_NAIVE] epoch={epoch} step={self.global_steps} begin")
                metrics = {}
                timing_raw = {}

                is_last_step = self.global_steps >= self.total_training_steps

                with marked_timer("step", timing_raw):
                    batch_a, batch_b = DataProto.from_single_dict_two_copies(batch_dict)

                    # -------- 准备 A 的 gen 输入（复制 _step_one_actor 里 gen 前的准备逻辑）--------
                    actor_cfg_a = self._actor_cfg("a")
                    batch_a.meta_info["temperature"] = actor_cfg_a.rollout.temperature
                    batch_a.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(batch_a.batch))], dtype=object)

                    gen_a = self._get_gen_batch(batch_a)
                    gen_a.meta_info["global_steps"] = self.global_steps
                    gen_a = gen_a.repeat(repeat_times=actor_cfg_a.rollout.n, interleave=True)

                    # -------- 准备 B 的 gen 输入 --------
                    actor_cfg_b = self._actor_cfg("b")
                    batch_b.meta_info["temperature"] = actor_cfg_b.rollout.temperature
                    batch_b.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(batch_b.batch))], dtype=object)

                    gen_b = self._get_gen_batch(batch_b)
                    gen_b.meta_info["global_steps"] = self.global_steps
                    gen_b = gen_b.repeat(repeat_times=actor_cfg_b.rollout.n, interleave=True)

                    # -------- 并行提交 generate_sequences，并统一等待 --------
                    rtprint(f"[rt] step={self.global_steps} submit gen_a/gen_b ...")
                    t_gen0 = time.time()

                    with ThreadPoolExecutor(max_workers=2) as ex:
                        fut_a = ex.submit(self._async_mgr("a").generate_sequences, gen_a)
                        fut_b = ex.submit(self._async_mgr("b").generate_sequences, gen_b)
                        gen_out_a = fut_a.result()
                        gen_out_b = fut_b.result()
                    t_gen1 = time.time()

                    timing_raw["gen_parallel_wall"] = t_gen1 - t_gen0
                    print(f"[ROLLOUT] wall_time={t_gen1 - t_gen0:.3f}s (a+b parallel) step={self.global_steps}", flush=True)

                    # 把 vLLM timing 合进 timing_raw（否则你后面看不到 engine 内部细分）
                    timing_raw.update(gen_out_a.meta_info.get("timing", {}))
                    timing_raw.update(gen_out_b.meta_info.get("timing", {}))
                    gen_out_a.meta_info.pop("timing", None)
                    gen_out_b.meta_info.pop("timing", None)

                    # 关键：并发收益估算（需要 gen_a/gen_b 也被计到 timing_raw）
                    ga = timing_raw.get("gen_a", None)
                    gb = timing_raw.get("gen_b", None)
                    if ga is not None and gb is not None:
                        speedup = (ga + gb) / max(timing_raw["gen_parallel_wall"], 1e-9)
                        rtprint(f"[rt] step={self.global_steps} parallel_speedup≈{speedup:.2f}x "
                                f"(gen_a+gen_b={(ga+gb)*1000:.1f}ms vs wall={timing_raw['gen_parallel_wall']*1000:.1f}ms)")

                    if "uid" in batch_a.non_tensor_batch:
                        gen_out_a.non_tensor_batch["uid"] = batch_a.non_tensor_batch["uid"]
                    if "uid" in batch_b.non_tensor_batch:
                        gen_out_b.non_tensor_batch["uid"] = batch_b.non_tensor_batch["uid"]

                    # -------- 继续走 _step_one_actor 后半段（reward/logprob/values/adv），但跳过内部 gen --------
                    batch_a, _ = self._step_one_actor("a", batch_a, timing_raw, metrics, gen_batch_output_override=gen_out_a)
                    batch_b, _ = self._step_one_actor("b", batch_b, timing_raw, metrics, gen_batch_output_override=gen_out_b)

                    if self.use_critic:
                        with marked_timer("update_critic_a", timing_raw, color="pink"):
                            critic_out_a = self._update_critic(batch_a)
                        metrics.update(reduce_metrics(critic_out_a.meta_info["metrics"]))

                        with marked_timer("update_critic_b", timing_raw, color="pink"):
                            critic_out_b = self._update_critic(batch_b)
                        metrics.update(reduce_metrics(critic_out_b.meta_info["metrics"]))

                    if self.config.trainer.critic_warmup <= self.global_steps:
                        with marked_timer("update_actor_a", timing_raw, color="red"):
                            actor_out_a = self._update_actor("a", batch_a)
                        metrics.update(reduce_metrics(actor_out_a.meta_info["metrics"]))

                        with marked_timer("update_actor_b", timing_raw, color="red"):
                            actor_out_b = self._update_actor("b", batch_b)
                        metrics.update(reduce_metrics(actor_out_b.meta_info["metrics"]))

                        esi_close_to_expiration = should_save_ckpt_esi(
                            max_steps_duration=self.max_steps_duration,
                            redundant_time=self.config.trainer.esi_redundant_time,
                        )
                        if self.config.trainer.save_freq > 0 and (
                            self.global_steps % self.config.trainer.save_freq == 0
                            or esi_close_to_expiration
                        ):
                            with marked_timer("save_checkpoint", timing_raw, color="green"):
                                self._save_checkpoint()

                        with marked_timer("update_weights_a", timing_raw, color="red"):
                            self.checkpoint_manager_a.update_weights()
                        with marked_timer("update_weights_b", timing_raw, color="red"):
                            self.checkpoint_manager_b.update_weights()

                if (
                    self.val_reward_fn is not None
                    and self.config.trainer.test_freq > 0
                    and (self.global_steps % self.config.trainer.test_freq == 0)
                ):
                    with marked_timer("testing", timing_raw, color="green"):
                        val_a = self._validate_one("a")
                        val_b = self._validate_one("b")
                        metrics.update(val_a)
                        metrics.update(val_b)
                        if is_last_step:
                            last_val_metrics = {"a": val_a, "b": val_b}

                steps_duration = timing_raw.get("step", 0.0)
                self.max_steps_duration = max(self.max_steps_duration, steps_duration)

                metrics.update({"training/global_step": self.global_steps, "training/epoch": epoch})

                metrics.update(rename_dict(compute_data_metrics(batch=batch_a, use_critic=self.use_critic), "data_a/"))
                metrics.update(rename_dict(compute_data_metrics(batch=batch_b, use_critic=self.use_critic), "data_b/"))

                metrics.update(compute_timing_metrics(batch=batch_a, timing_raw=timing_raw))
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch_a, timing_raw=timing_raw, n_gpus=n_gpus))

                gradient_norm = metrics.get("actor_a/grad_norm", None) or metrics.get("actor_b/grad_norm", None)
                metrics.update(compute_variance_proxy_metrics(batch=batch_a, gradient_norm=gradient_norm))

                # logger.log(data=metrics, step=self.global_steps)

                if getattr(self, "profiler", None) is not None:
                    rtprint(f"[{_ts()}] [FIT_NAIVE] before profiler.update step={self.global_steps}")
                    self.profiler.update(timing_raw=timing_raw, step=self.global_steps)
                    rtprint(f"[{_ts()}] [FIT_NAIVE] after profiler.update step={self.global_steps}")

                progress_bar.update(1)
                self.global_steps += 1

                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

    def fit_overlap_decode(self, start_b_when_active_a_leq: int = 0, poll_timeout_ms: int = 20):
        import time
        import uuid
        import numpy as np
        import torch
        from tqdm import tqdm
        from tensordict import TensorDict
        from omegaconf import OmegaConf
        from pprint import pprint
        from verl import DataProto

        print("\n========== ENTER fit_overlap_decode ==========", flush=True)
        def _ts():
            return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

        self.global_steps = 0
        self._load_checkpoint()
        print("[INIT] checkpoint loaded", flush=True)

        self.checkpoint_manager_a.update_weights()
        self.checkpoint_manager_b.update_weights()
        print("[INIT] weights synced", flush=True)

        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps)

        self.global_steps += 1

        # ----------------------------
        # helper: trim padding
        # ----------------------------
        def _trim(ids, attn):
            if attn is not None:
                valid_len = int(attn.sum().item())
                if valid_len > 0:
                    return ids[-valid_len:]
            while len(ids) > 0 and ids[0] == 0:
                ids = ids[1:]
            return ids

        tool_schemas = None
        tool_config_path = self.config.data.get("tool_config_path", None)
        if tool_config_path:
            try:
                from verl.tools.utils.tool_registry import initialize_tools_from_config

                tool_list = initialize_tools_from_config(tool_config_path)
                tool_schemas = [tool.tool_schema.model_dump(exclude_unset=True, exclude_none=True) for tool in tool_list]
            except Exception:
                tool_schemas = None

        def _messages_to_prompt_ids(messages):
            apply_kwargs = dict(self.config.data.get("apply_chat_template_kwargs", {}))
            if self.processor is not None:
                raw_prompt = self.processor.apply_chat_template(
                    messages, tools=tool_schemas, add_generation_prompt=True, tokenize=False, **apply_kwargs
                )
                model_inputs = self.processor(text=[raw_prompt], return_tensors="pt")
                return model_inputs["input_ids"].squeeze(0).tolist()
            return self.tokenizer.apply_chat_template(
                messages, tools=tool_schemas, add_generation_prompt=True, tokenize=True, **apply_kwargs
            )

        def _get_prompt_ids(gen_in: DataProto, idx: int):
            nt = gen_in.non_tensor_batch
            if "raw_prompt" in nt:
                messages = list(nt["raw_prompt"][idx])
                return _messages_to_prompt_ids(messages)

            if "prompt" in nt:
                prompt_obj = nt["prompt"][idx]
                if isinstance(prompt_obj, str):
                    return self.tokenizer(prompt_obj, add_special_tokens=False)["input_ids"]
                if isinstance(prompt_obj, (list, tuple)):
                    if len(prompt_obj) > 0 and isinstance(prompt_obj[0], dict):
                        return _messages_to_prompt_ids(list(prompt_obj))
                    if len(prompt_obj) == 0 or isinstance(prompt_obj[0], (int, np.integer)):
                        return [int(x) for x in prompt_obj]

            raise KeyError(f"cannot build prompt ids, non_tensor keys={list(nt.keys())}")

        def _stream_sampling_params(actor_cfg, prompt_ids):
            rollout_cfg = actor_cfg.rollout
            sampling_params = {
                "temperature": float(rollout_cfg.temperature),
                "top_p": float(getattr(rollout_cfg, "top_p", 1.0)),
                "top_k": int(getattr(rollout_cfg, "top_k", -1)),
                "repetition_penalty": float(getattr(rollout_cfg, "repetition_penalty", 1.0)),
                "logprobs": bool(getattr(rollout_cfg, "calculate_log_probs", False)),
            }

            max_model_len = int(getattr(rollout_cfg, "max_model_len", rollout_cfg.prompt_length + rollout_cfg.response_length))
            max_possible_tokens = max(0, max_model_len - len(prompt_ids))
            rollout_name = str(getattr(rollout_cfg, "name", "")).lower()

            # Match the non-streaming defaults as closely as possible so overlap/non-overlap
            # runs use the same generation budget.
            if rollout_name == "sglang":
                target_max_tokens = int(rollout_cfg.response_length + rollout_cfg.prompt_length - len(prompt_ids))
            else:
                target_max_tokens = int(rollout_cfg.response_length)

            sampling_params["max_tokens"] = max(0, min(target_max_tokens, max_possible_tokens))
            return sampling_params

        # ----------------------------
        # helper: build DataProto from token buffers
        # ----------------------------
        def _build_output(gen_in, prompt_ids_by_handle, result_by_handle, handles, prompt_length, response_length):
            print("[BUILD] assembling DataProto", flush=True)
            bsz = len(handles)
            prompt_len = int(prompt_length)
            pad_id = self.tokenizer.pad_token_id or 0

            prompts = torch.full((bsz, prompt_len), int(pad_id), dtype=torch.long)
            prompt_attn = torch.zeros((bsz, prompt_len), dtype=torch.long)
            resp = torch.zeros((bsz, response_length), dtype=torch.long)
            resp_mask = torch.zeros((bsz, response_length), dtype=torch.long)

            for i, h in enumerate(handles):
                pids = prompt_ids_by_handle.get(h, [])
                if pids:
                    pids = pids[-prompt_len:]
                    lp = len(pids)
                    prompts[i, prompt_len - lp : prompt_len] = torch.tensor(pids, dtype=torch.long)
                    prompt_attn[i, prompt_len - lp : prompt_len] = 1

                toks = result_by_handle.get(h, {}).get("output_token_ids", [])
                L = min(len(toks), response_length)
                if L > 0:
                    resp[i, :L] = torch.tensor(toks[:L], dtype=torch.long)
                    resp_mask[i, :L] = 1

            attention_mask = torch.cat([prompt_attn, resp_mask], dim=1)
            input_ids = torch.cat([prompts, resp], dim=1)
            position_ids = compute_position_id_with_mask(attention_mask)

            batch = TensorDict(
                {
                    "prompts": prompts,
                    "responses": resp,
                    "response_mask": resp_mask,
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "position_ids": position_ids,
                },
                batch_size=bsz,
            )
            return DataProto(batch=batch, non_tensor_batch=gen_in.non_tensor_batch, meta_info={"timing": {}})

        # ============================
        # main loop
        # ============================
        for epoch in range(self.config.trainer.total_epochs):
            print(f"\n[LOOP] epoch={epoch}", flush=True)

            for batch_dict in self.train_dataloader:
                print(f"\n[{_ts()}] [STEP] global_step={self.global_steps}", flush=True)
                t_step0 = time.perf_counter()

                batch_a, batch_b = DataProto.from_single_dict_two_copies(batch_dict)

                batch_a.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(batch_a.batch))], dtype=object)
                batch_b.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(batch_b.batch))], dtype=object)

                # Keep gen input preparation consistent with _step_one_actor/fit_naive_concurrent_rollout.
                actor_cfg_a = self._actor_cfg("a")
                actor_cfg_b = self._actor_cfg("b")
                batch_a.meta_info["temperature"] = actor_cfg_a.rollout.temperature
                batch_b.meta_info["temperature"] = actor_cfg_b.rollout.temperature
                batch_a.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(batch_a.batch))], dtype=object)
                batch_b.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(batch_b.batch))], dtype=object)

                gen_a = self._get_gen_batch(batch_a)
                gen_a.meta_info["global_steps"] = self.global_steps
                gen_a = gen_a.repeat(repeat_times=actor_cfg_a.rollout.n, interleave=True)

                gen_b = self._get_gen_batch(batch_b)
                gen_b.meta_info["global_steps"] = self.global_steps
                gen_b = gen_b.repeat(repeat_times=actor_cfg_b.rollout.n, interleave=True)

                mgr_a = self._async_mgr("a")
                mgr_b = self._async_mgr("b")

                # ---------- start A ----------
                print(f"[{_ts()}] [A] starting streaming...", flush=True)
                t_rollout_a_wall0 = time.perf_counter()
                t_a_start = time.perf_counter()

                handles_a = []
                result_meta_a = {}
                prompt_ids_a_by_handle = {}
                for i in range(len(gen_a)):
                    ids = _get_prompt_ids(gen_a, i)
                    sampling_a = _stream_sampling_params(actor_cfg_a, ids)

                    rid = str(uuid.uuid4())
                    ret = mgr_a.start_generate_stream(
                        prompt_ids=ids,
                        sampling_params=sampling_a,
                        request_id=rid,
                        emit_token_deltas=False,
                        training_global_step=self.global_steps,
                    )
                    h = ret["handle"]
                    handles_a.append(h)
                    prompt_ids_a_by_handle[h] = ids

                active_a_count = len(handles_a)
                group_a = mgr_a.register_generate_stream_group(handles_a)["group_id"]
                print(f"[{_ts()}] [A] started {active_a_count} streams", flush=True)

                started_b = False
                active_b_count = 0
                group_b = None
                result_meta_b = {}
                handles_b = []
                prompt_ids_b_by_handle = {}
                t_b_trigger = None
                t_b_start = None
                t_a_done = None
                t_b_done = None
                _hl_threshold = 128
                high_low_a = {"triggered": False, "start_ts": None, "elapsed_s": None}
                high_low_b = {"triggered": False, "start_ts": None, "elapsed_s": None}

                poll_iter = 0

                # ---------- status loop ----------
                print(f"[{_ts()}] [STATUS] enter polling loop", flush=True)

                while True:
                    poll_iter += 1

                    if active_a_count > 0:
                        group_status_a = mgr_a.get_generate_stream_group_status(group_a)
                        active_a_count = int(group_status_a.get("active_count", active_a_count))
                        if active_a_count == 0 and t_a_done is None:
                            t_a_done = time.perf_counter()
                            print(f"[ROLLOUT] wall_time={t_a_done - t_rollout_a_wall0:.3f}s which=a step={self.global_steps}", flush=True)

                    if (not started_b) and (active_a_count <= start_b_when_active_a_leq):
                        print(f"[{_ts()}] [TRIGGER] start B (active_a={active_a_count})", flush=True)
                        started_b = True
                        t_rollout_b_wall0 = time.perf_counter()
                        t_b_trigger = time.perf_counter()
                        t_b_start = t_b_trigger

                        for i in range(len(gen_b)):
                            ids = _get_prompt_ids(gen_b, i)
                            sampling_b = _stream_sampling_params(actor_cfg_b, ids)

                            rid = str(uuid.uuid4())
                            ret = mgr_b.start_generate_stream(
                                prompt_ids=ids,
                                sampling_params=sampling_b,
                                request_id=rid,
                                emit_token_deltas=False,
                                training_global_step=self.global_steps,
                            )
                            h = ret["handle"]
                            handles_b.append(h)
                            prompt_ids_b_by_handle[h] = ids

                        active_b_count = len(handles_b)
                        group_b = mgr_b.register_generate_stream_group(handles_b)["group_id"]
                        print(f"[{_ts()}] [B] started {active_b_count} streams", flush=True)

                    if started_b and active_b_count > 0:
                        group_status_b = mgr_b.get_generate_stream_group_status(group_b)
                        active_b_count = int(group_status_b.get("active_count", active_b_count))
                        if active_b_count == 0 and t_b_done is None:
                            t_b_done = time.perf_counter()
                            print(f"[ROLLOUT] wall_time={t_b_done - t_rollout_b_wall0:.3f}s which=b step={self.global_steps}", flush=True)

                    if not high_low_a["triggered"] and 0 < active_a_count < _hl_threshold:
                        high_low_a["triggered"] = True
                        high_low_a["start_ts"] = time.perf_counter()
                    if not high_low_b["triggered"] and started_b and 0 < active_b_count < _hl_threshold:
                        high_low_b["triggered"] = True
                        high_low_b["start_ts"] = time.perf_counter()

                    if poll_iter % 50 == 0:
                        print(
                            f"[{_ts()}] [STATUS] iter={poll_iter} active_a={active_a_count} active_b={active_b_count}",
                            flush=True,
                        )

                    if active_a_count == 0 and (not started_b or active_b_count == 0):
                        print(f"[{_ts()}] [STATUS] exit loop", flush=True)
                        break

                    time.sleep(max(float(poll_timeout_ms) / 1000.0, 0.001))

                # ---------- finalize ----------
                print(f"[{_ts()}] [FINALIZE] A", flush=True)
                for h in handles_a:
                    result_meta_a[h] = mgr_a.finalize_generate_stream(h)
                mgr_a.clear_generate_stream_group(group_a)

                if started_b:
                    print(f"[{_ts()}] [FINALIZE] B", flush=True)
                    for h in handles_b:
                        result_meta_b[h] = mgr_b.finalize_generate_stream(h)
                    if group_b is not None:
                        mgr_b.clear_generate_stream_group(group_b)
                for _label, _hl, _t_done in [("A", high_low_a, t_a_done), ("B", high_low_b, t_b_done)]:
                    if _hl["triggered"] and _hl["start_ts"] is not None:
                        _hl["elapsed_s"] = (_t_done if _t_done else time.perf_counter()) - _hl["start_ts"]
                        print(
                            f"[{_ts()}] [HIGH_LOW_{_label}] threshold={_hl_threshold} "
                            f"tail_to_done={_hl['elapsed_s']:.3f}s",
                            flush=True,
                        )

                def _fmt_sec(v):
                    return f"{v:.3f}s" if v is not None else "NA"

                def _dist_fmt(mn, mx, avg, p50):
                    if mn is None:
                        return "NA"
                    return f"min={mn:.3f}s max={mx:.3f}s avg={avg:.3f}s p50={p50:.3f}s"

                ttft_a = (None, None, None, None)
                ttft_b = (None, None, None, None)
                prefill_phase_a = None
                prefill_phase_b = None
                print(f"[{_ts()}] [TTFT_A] n=0 {_dist_fmt(*ttft_a)}", flush=True)
                print(f"[{_ts()}] [TTFT_B] n=0 {_dist_fmt(*ttft_b)}", flush=True)
                print(f"[{_ts()}] [SERVER_PREFILL_A] n=0 prefill_phase=NA prefill_compute=NA queue_time=NA launch_delay=NA", flush=True)
                print(f"[{_ts()}] [SERVER_PREFILL_B] n=0 prefill_phase=NA prefill_compute=NA queue_time=NA launch_delay=NA", flush=True)

                # ---------- build outputs ----------
                gen_out_a = _build_output(
                    gen_a,
                    prompt_ids_a_by_handle,
                    result_meta_a,
                    handles_a,
                    int(actor_cfg_a.rollout.prompt_length),
                    int(actor_cfg_a.rollout.response_length),
                )

                if started_b:
                    gen_out_b = _build_output(
                        gen_b,
                        prompt_ids_b_by_handle,
                        result_meta_b,
                        handles_b,
                        int(actor_cfg_b.rollout.prompt_length),
                        int(actor_cfg_b.rollout.response_length),
                    )
                else:
                    print("[WARNING] B never started — fallback sync generate", flush=True)
                    gen_out_b = mgr_b.generate_sequences(gen_b)

                self._print_longest_response("fit_overlap_decode_rollout_a", gen_out_a, self.global_steps)
                self._print_longest_response("fit_overlap_decode_rollout_b", gen_out_b, self.global_steps)

                print(f"[{_ts()}] [STEP] before _step_one_actor", flush=True)
                t_train0 = time.perf_counter()

                batch_a, _ = self._step_one_actor(
                    "a", batch_a, {}, {}, gen_batch_output_override=gen_out_a
                )
                batch_b, _ = self._step_one_actor(
                    "b", batch_b, {}, {}, gen_batch_output_override=gen_out_b
                )

                print(f"[{_ts()}] [STEP] update_actor / update_critic", flush=True)

                self._update_actor("a", batch_a)
                self._update_actor("b", batch_b)
                t_train1 = time.perf_counter()

                a_to_threshold = (t_b_trigger - t_a_start) if t_b_trigger is not None else None
                a_rollout_total = (t_a_done - t_a_start) if t_a_done is not None else None
                b_rollout_total = (t_b_done - t_b_start) if (t_b_start is not None and t_b_done is not None) else None
                train_time = t_train1 - t_train0
                step_total = t_train1 - t_step0

                print(
                    f"[{_ts()}] [WALL] step={self.global_steps} "
                    f"a_to_threshold={_fmt_sec(a_to_threshold)} "
                    f"a_rollout_total={_fmt_sec(a_rollout_total)} "
                    f"b_rollout_total={_fmt_sec(b_rollout_total)} "
                    f"hl_tail_a={_fmt_sec(high_low_a['elapsed_s'])} "
                    f"hl_tail_b={_fmt_sec(high_low_b['elapsed_s'])} "
                    f"prefill_phase_a={_fmt_sec(prefill_phase_a)} "
                    f"prefill_phase_b={_fmt_sec(prefill_phase_b)} "
                    f"ttft_a_avg={_fmt_sec(ttft_a[2])} "
                    f"ttft_b_avg={_fmt_sec(ttft_b[2])} "
                    f"train_time={_fmt_sec(train_time)} "
                    f"step_total={_fmt_sec(step_total)}",
                    flush=True,
                )

                progress_bar.update(1)
                self.global_steps += 1

                if self.global_steps >= self.total_training_steps:
                    print("========== EXIT fit_overlap_decode ==========", flush=True)
                    progress_bar.close()
                    return

    # =========================================================================
    # fit_overlap_b_rollout_a_train
    #   A rollout (sync) -> B rollout (streaming) || A train -> B train
    # =========================================================================
    def fit_overlap_b_rollout_a_train(self):
        from concurrent.futures import ThreadPoolExecutor
        import time
        import uuid
        import numpy as np
        import torch
        from tensordict import TensorDict
        from omegaconf import OmegaConf
        from pprint import pprint

        from verl import DataProto
        from verl.utils.tracking import Tracking

        _ts = lambda: time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        _fmt = lambda v: f"{v:.3f}s" if v is not None else "NA"

        print("\n========== ENTER fit_overlap_b_rollout_a_train ==========", flush=True)

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0
        self._load_checkpoint()

        self.checkpoint_manager_a.update_weights()
        self.checkpoint_manager_b.update_weights()

        if self.cfg_a.rollout.get("skip_rollout", False):
            RolloutSkip(self._build_actor_scoped_config("a"), self.actor_rollout_wg_a).wrap_generate_sequences()
        if self.cfg_b.rollout.get("skip_rollout", False):
            RolloutSkip(self._build_actor_scoped_config("b"), self.actor_rollout_wg_b).wrap_generate_sequences()

        current_epoch = self.global_steps // len(self.train_dataloader) if len(self.train_dataloader) > 0 else 0

        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            self._validate_one("a")
            self._validate_one("b")
            if self.config.trainer.get("val_only", False):
                return

        prof_cfg = self.config.trainer.get("profile", {})
        self.profiler = StepProfiler(
            enabled=prof_cfg.get("enabled", True),
            verbose=prof_cfg.get("verbose", True),
            print_every=prof_cfg.get("print_every", 1),
            window=prof_cfg.get("window", 1),
            print_rollout_each=prof_cfg.get("print_rollout_each", False),
        )

        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Dual Training Progress")
        self.global_steps += 1
        last_val_metrics = None
        self.max_steps_duration = 0

        # Threshold: start A train only when active B streams are no larger than this.
        start_a_when_active_b_leq = getattr(
            self.config.trainer, "start_a_when_active_b_leq", 1024
        )
        poll_timeout_ms = getattr(self.config.trainer, "overlap_poll_timeout_ms", 20)

        # Helpers to build prompt ids for streaming rollout (borrowed from fit_overlap_decode)
        tool_schemas = None
        tool_config_path = self.config.data.get("tool_config_path", None)
        if tool_config_path:
            try:
                from verl.tools.utils.tool_registry import initialize_tools_from_config

                tool_list = initialize_tools_from_config(tool_config_path)
                tool_schemas = [
                    tool.tool_schema.model_dump(exclude_unset=True, exclude_none=True) for tool in tool_list
                ]
            except Exception:
                tool_schemas = None

        def _messages_to_prompt_ids(messages):
            apply_kwargs = dict(self.config.data.get("apply_chat_template_kwargs", {}))
            if self.processor is not None:
                raw_prompt = self.processor.apply_chat_template(
                    messages, tools=tool_schemas, add_generation_prompt=True, tokenize=False, **apply_kwargs
                )
                model_inputs = self.processor(text=[raw_prompt], return_tensors="pt")
                return model_inputs["input_ids"].squeeze(0).tolist()
            return self.tokenizer.apply_chat_template(
                messages, tools=tool_schemas, add_generation_prompt=True, tokenize=True, **apply_kwargs
            )

        def _get_prompt_ids(gen_in: DataProto, idx: int):
            nt = gen_in.non_tensor_batch
            if "raw_prompt" in nt:
                messages = list(nt["raw_prompt"][idx])
                return _messages_to_prompt_ids(messages)

            if "prompt" in nt:
                prompt_obj = nt["prompt"][idx]
                if isinstance(prompt_obj, str):
                    return self.tokenizer(prompt_obj, add_special_tokens=False)["input_ids"]
                if isinstance(prompt_obj, (list, tuple)):
                    if len(prompt_obj) > 0 and isinstance(prompt_obj[0], dict):
                        return _messages_to_prompt_ids(list(prompt_obj))
                    if len(prompt_obj) == 0 or isinstance(prompt_obj[0], (int, np.integer)):
                        return [int(x) for x in prompt_obj]

            raise KeyError(f"cannot build prompt ids, non_tensor keys={list(nt.keys())}")

        def _build_output(gen_in, prompt_ids_by_handle, tokbuf, handles, prompt_length, response_length):
            bsz = len(handles)
            prompt_len = int(prompt_length)
            pad_id = self.tokenizer.pad_token_id or 0

            prompts = torch.full((bsz, prompt_len), int(pad_id), dtype=torch.long)
            prompt_attn = torch.zeros((bsz, prompt_len), dtype=torch.long)
            resp = torch.zeros((bsz, response_length), dtype=torch.long)
            resp_mask = torch.zeros((bsz, response_length), dtype=torch.long)

            for i, h in enumerate(handles):
                pids = prompt_ids_by_handle.get(h, [])
                if pids:
                    pids = pids[-prompt_len:]
                    lp = len(pids)
                    prompts[i, prompt_len - lp : prompt_len] = torch.tensor(pids, dtype=torch.long)
                    prompt_attn[i, prompt_len - lp : prompt_len] = 1

                toks = tokbuf.get(h, [])
                L = min(len(toks), response_length)
                if L > 0:
                    resp[i, :L] = torch.tensor(toks[:L], dtype=torch.long)
                    resp_mask[i, :L] = 1

            attention_mask = torch.cat([prompt_attn, resp_mask], dim=1)
            input_ids = torch.cat([prompts, resp], dim=1)
            position_ids = compute_position_id_with_mask(attention_mask)

            batch = TensorDict(
                {
                    "prompts": prompts,
                    "responses": resp,
                    "response_mask": resp_mask,
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "position_ids": position_ids,
                },
                batch_size=bsz,
            )
            return DataProto(batch=batch, non_tensor_batch=gen_in.non_tensor_batch, meta_info={"timing": {}})

        for epoch in range(current_epoch, self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                rtprint(f"[{_ts()}] [FIT_OVERLAP] epoch={epoch} step={self.global_steps} begin")
                metrics = {}
                timing_raw = {}
                is_last_step = self.global_steps >= self.total_training_steps

                with marked_timer("step", timing_raw):
                    batch_a, batch_b = DataProto.from_single_dict_two_copies(batch_dict)

                    actor_cfg_a = self._actor_cfg("a")
                    batch_a.meta_info["temperature"] = actor_cfg_a.rollout.temperature
                    batch_a.non_tensor_batch["uid"] = np.array(
                        [str(uuid.uuid4()) for _ in range(len(batch_a.batch))], dtype=object
                    )
                    gen_a = self._get_gen_batch(batch_a)
                    gen_a.meta_info["global_steps"] = self.global_steps
                    gen_a = gen_a.repeat(repeat_times=actor_cfg_a.rollout.n, interleave=True)

                    actor_cfg_b = self._actor_cfg("b")
                    batch_b.meta_info["temperature"] = actor_cfg_b.rollout.temperature
                    batch_b.non_tensor_batch["uid"] = np.array(
                        [str(uuid.uuid4()) for _ in range(len(batch_b.batch))], dtype=object
                    )
                    gen_b = self._get_gen_batch(batch_b)
                    gen_b.meta_info["global_steps"] = self.global_steps
                    gen_b = gen_b.repeat(repeat_times=actor_cfg_b.rollout.n, interleave=True)

                    mgr_a = self._async_mgr("a")
                    mgr_b = self._async_mgr("b")

                    # ── Phase 1: A rollout (sync, blocking) ──
                    t_a_roll0 = time.perf_counter()
                    with marked_timer("gen_a", timing_raw):
                        gen_out_a = mgr_a.generate_sequences(gen_a)
                    t_a_roll1 = time.perf_counter()
                    print(f"[ROLLOUT] wall_time={t_a_roll1 - t_a_roll0:.3f}s which=a step={self.global_steps}", flush=True)
                    self._print_longest_response("fit_overlap_b_rollout_a_train_rollout_a", gen_out_a, self.global_steps)

                    timing_raw.update(gen_out_a.meta_info.get("timing", {}))
                    gen_out_a.meta_info.pop("timing", None)
                    if "uid" in batch_a.non_tensor_batch:
                        gen_out_a.non_tensor_batch["uid"] = batch_a.non_tensor_batch["uid"]

                    # ── Phase 2: start B rollout (streaming) ──
                    sampling_b = {
                        "temperature": float(actor_cfg_b.rollout.temperature),
                        "max_tokens": int(actor_cfg_b.rollout.response_length),
                    }
                    handles_b = []
                    active_b = set()
                    tokbuf_b = {}
                    prompt_ids_b_by_handle = {}

                    for i in range(len(gen_b)):
                        ids = _get_prompt_ids(gen_b, i)
                        rid = str(uuid.uuid4())
                        ret = mgr_b.start_generate_stream(
                            prompt_ids=ids,
                            sampling_params=sampling_b,
                            request_id=rid,
                            training_global_step=self.global_steps,
                        )
                        h = ret["handle"]
                        handles_b.append(h)
                        active_b.add(h)
                        tokbuf_b[h] = []
                        prompt_ids_b_by_handle[h] = ids

                    rtprint(
                        f"[{_ts()}] [B_STREAM] started {len(active_b)} streams, "
                        f"start_a_when_active_b_leq={start_a_when_active_b_leq}",
                    )

                    # ── Phase 3: poll B; when active_b <= threshold, start A train in a thread ──
                    overlap_time = 0.0
                    b_extra_after_train = 0.0
                    t_overlap0 = time.perf_counter()

                    started_a_train = False
                    a_train_done = False

                    def _run_a_train():
                        nonlocal batch_a, a_train_done
                        rtprint(f"[{_ts()}] [A_TRAIN] start (overlap with B)")
                        t_train_a0 = time.perf_counter()

                        local_timing = {}
                        local_metrics = {}
                        batch_a_local, _ = self._step_one_actor(
                            "a", batch_a, local_timing, local_metrics, gen_batch_output_override=gen_out_a
                        )
                        batch_a = batch_a_local

                        if self.use_critic:
                            with marked_timer("update_critic_a", local_timing, color="pink"):
                                critic_out_a = self._update_critic(batch_a)

                        if self.config.trainer.critic_warmup <= self.global_steps:
                            with marked_timer("update_actor_a", local_timing, color="red"):
                                self._update_actor("a", batch_a)

                        t_train_a1 = time.perf_counter()
                        local_timing["a_train_wall"] = t_train_a1 - t_train_a0
                        rtprint(
                            f"[{_ts()}] [A_TRAIN] done, wall={_fmt(local_timing['a_train_wall'])}",
                        )
                        a_train_done = True
                        return local_timing

                    with ThreadPoolExecutor(max_workers=1) as ex:
                        fut_a = None
                        t_a_train_start = None
                        t_a_train_end = None
                        poll_iter = 0

                        while True:
                            poll_iter += 1

                            if active_b:
                                poll_b = mgr_b.poll_generate_stream_many(list(active_b), timeout_ms=poll_timeout_ms)
                                if poll_b:
                                    for item in poll_b:
                                        h = item["handle"]
                                        ev = item.get("event", item)
                                        typ = ev.get("type")
                                        if typ == "delta":
                                            tokbuf_b[h].extend(ev.get("token_ids", []))
                                        elif typ in ("done", "error"):
                                            active_b.discard(h)

                            if (not started_a_train) and (len(active_b) <= start_a_when_active_b_leq):
                                rtprint(
                                    f"[{_ts()}] [TRIGGER_A_TRAIN] active_b={len(active_b)} "
                                    f"<= {start_a_when_active_b_leq}",
                                )
                                started_a_train = True
                                t_a_train_start = time.perf_counter()
                                fut_a = ex.submit(_run_a_train)

                            # Exit polling as soon as all B streams finish.
                            # A 训练线程在循环外用 fut_a.result() 等待即可。
                            if not active_b:
                                rtprint(
                                    f"[{_ts()}] [POLL_B] exit loop "
                                    f"(active_b={len(active_b)}, started_a_train={started_a_train}, "
                                    f"a_train_done={a_train_done})",
                                )
                                break

                            if poll_iter % 50 == 0:
                                rtprint(
                                    f"[{_ts()}] [POLL_B] iter={poll_iter} active_b={len(active_b)}"
                                )

                        if fut_a is not None:
                            local_timing = fut_a.result()
                            t_a_train_end = time.perf_counter()
                            timing_raw.update(local_timing)
                        else:
                            t_a_train_start = t_a_train_end = None

                    t_overlap1 = time.perf_counter()

                    # Finalize B streams
                    for h in handles_b:
                        mgr_b.finalize_generate_stream(h)

                    # Build B rollout outputs from streamed tokens
                    gen_out_b = _build_output(
                        gen_b,
                        prompt_ids_b_by_handle,
                        tokbuf_b,
                        handles_b,
                        int(actor_cfg_b.rollout.prompt_length),
                        int(actor_cfg_b.rollout.response_length),
                    )
                    self._print_longest_response("fit_overlap_b_rollout_a_train_rollout_b", gen_out_b, self.global_steps)

                    # Overlap statistics (approximate)
                    if t_a_train_start is not None and t_a_train_end is not None:
                        a_train_wall = t_a_train_end - t_a_train_start
                    else:
                        a_train_wall = 0.0
                    b_roll_wall = t_overlap1 - t_overlap0
                    overlap_time = min(a_train_wall, b_roll_wall) if a_train_wall > 0 else 0.0
                    b_extra_after_train = max(0.0, b_roll_wall - a_train_wall)

                    rtprint(
                        f"[{_ts()}] [OVERLAP] a_train={_fmt(a_train_wall)} "
                        f"b_roll={_fmt(b_roll_wall)} overlap={_fmt(overlap_time)} "
                        f"b_extra={_fmt(b_extra_after_train)}",
                    )

                    # ── Phase 4: B train ──
                    t_b_train0 = time.perf_counter()
                    with marked_timer("step_b", timing_raw):
                        batch_b, _ = self._step_one_actor(
                            "b", batch_b, timing_raw, metrics, gen_batch_output_override=gen_out_b
                        )

                    if self.use_critic:
                        with marked_timer("update_critic_b", timing_raw, color="pink"):
                            critic_out_b = self._update_critic(batch_b)
                        metrics.update(reduce_metrics(critic_out_b.meta_info["metrics"]))

                    if self.config.trainer.critic_warmup <= self.global_steps:
                        with marked_timer("update_actor_b", timing_raw, color="red"):
                            actor_out_b = self._update_actor("b", batch_b)
                        metrics.update(reduce_metrics(actor_out_b.meta_info["metrics"]))
                        t_b_train1 = time.perf_counter()
                        timing_raw["b_train_wall"] = t_b_train1 - t_b_train0
                        rtprint(f"[{_ts()}] [B_TRAIN] done {_fmt(timing_raw['b_train_wall'])}")

                        esi_close_to_expiration = should_save_ckpt_esi(
                            max_steps_duration=self.max_steps_duration,
                            redundant_time=self.config.trainer.esi_redundant_time,
                        )
                        if self.config.trainer.save_freq > 0 and (
                            self.global_steps % self.config.trainer.save_freq == 0
                            or esi_close_to_expiration
                        ):
                            with marked_timer("save_checkpoint", timing_raw, color="green"):
                                self._save_checkpoint()

                        with marked_timer("update_weights_a", timing_raw, color="red"):
                            self.checkpoint_manager_a.update_weights()
                        with marked_timer("update_weights_b", timing_raw, color="red"):
                            self.checkpoint_manager_b.update_weights()

                if (
                    self.val_reward_fn is not None
                    and self.config.trainer.test_freq > 0
                    and (self.global_steps % self.config.trainer.test_freq == 0)
                ):
                    with marked_timer("testing", timing_raw, color="green"):
                        val_a = self._validate_one("a")
                        val_b = self._validate_one("b")
                        metrics.update(val_a)
                        metrics.update(val_b)
                        if is_last_step:
                            last_val_metrics = {"a": val_a, "b": val_b}

                steps_duration = timing_raw.get("step", 0.0)
                self.max_steps_duration = max(self.max_steps_duration, steps_duration)

                metrics.update({"training/global_step": self.global_steps, "training/epoch": epoch})
                metrics.update(rename_dict(compute_data_metrics(batch=batch_a, use_critic=self.use_critic), "data_a/"))
                metrics.update(rename_dict(compute_data_metrics(batch=batch_b, use_critic=self.use_critic), "data_b/"))
                metrics.update(compute_timing_metrics(batch=batch_a, timing_raw=timing_raw))
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch_a, timing_raw=timing_raw, n_gpus=n_gpus))
                gradient_norm = metrics.get("actor_a/grad_norm", None) or metrics.get("actor_b/grad_norm", None)
                metrics.update(compute_variance_proxy_metrics(batch=batch_a, gradient_norm=gradient_norm))

                timing_raw["overlap"] = overlap_time
                timing_raw["b_extra_after_train"] = b_extra_after_train

                if getattr(self, "profiler", None) is not None:
                    rtprint(f"[{_ts()}] [FIT_OVERLAP] before profiler.update step={self.global_steps}")
                    self.profiler.update(timing_raw=timing_raw, step=self.global_steps)
                    rtprint(f"[{_ts()}] [FIT_OVERLAP] after profiler.update step={self.global_steps}")

                progress_bar.update(1)
                self.global_steps += 1

                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    print("========== EXIT fit_overlap_b_rollout_a_train ==========", flush=True)
                    return

    # =========================================================================
    # Profiling: measure sub-stage latencies at different Green Context SM levels
    # =========================================================================

    def _profile_get_sm_count(self) -> int:
        """Query SM count from actor worker (driver has no CUDA)."""
        rets = self.actor_rollout_wg_a.get_sm_count()
        if isinstance(rets, (list, tuple)) and rets:
            return int(rets[0])
        return int(rets)

    @staticmethod
    def _profile_compute_sm_levels(total_sms: int, n_levels: int = 5) -> list:
        """Return SM counts spaced across [20%, ~90%], aligned to 4, plus None for 100%.

        None means "skip green context, use default stream (all SMs)".
        FlashInfer's split_device_green_ctx rounds up, so requesting total_sms
        will exceed device capacity — we cap partitioned levels at ~90%.
        """
        max_pct = 90
        pcts = np.linspace(20, max_pct, max(1, n_levels - 1))
        seen, levels = set(), []
        for p in pcts:
            raw = int(total_sms * p / 100)
            aligned = max(4, (raw // 4) * 4)  # round DOWN to mult-of-4
            aligned = min(aligned, total_sms - 4)  # leave room for remainder
            if aligned not in seen:
                seen.add(aligned)
                levels.append(aligned)
        levels.append(None)  # 100% = default stream, no green context
        return levels

    def _profile_make_batch(self, batch_size: int, prompt_len: int, response_len: int) -> DataProto:
        """Build a minimal synthetic DataProto suitable for compute_log_prob / update_actor."""
        vocab = getattr(self.tokenizer, "vocab_size", None) or 32000
        total_len = prompt_len + response_len

        input_ids = torch.randint(1, vocab, (batch_size, total_len))
        attention_mask = torch.ones(batch_size, total_len, dtype=torch.long)
        position_ids = torch.arange(total_len).unsqueeze(0).expand(batch_size, -1).clone()
        responses = input_ids[:, prompt_len:]
        response_mask = torch.ones(batch_size, response_len, dtype=torch.long)
        prompts = input_ids[:, :prompt_len]

        from tensordict import TensorDict as TD
        td = TD(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "responses": responses,
                "response_mask": response_mask,
                "prompts": prompts,
            },
            batch_size=batch_size,
        )
        temperature = float(OmegaConf.select(self.cfg_a, "rollout.temperature", default=1.0))
        return DataProto(batch=td, meta_info={"temperature": temperature})

    @staticmethod
    def _profile_timed(fn, iters: int = 3, warmup: int = 1) -> dict:
        _nan_result = lambda e: {"median_ms": float("nan"), "mean_ms": float("nan"),
                                 "min_ms": float("nan"), "max_ms": float("nan"),
                                 "times_ms": [], "error": str(e)}
        for _ in range(warmup):
            try:
                fn()
            except Exception as e:
                rtprint(f"  [_profile_timed] warmup exception: {e}")
                return _nan_result(e)
        times = []
        for _ in range(iters):
            try:
                t0 = time.time()
                fn()
                times.append((time.time() - t0) * 1000)
            except Exception as e:
                rtprint(f"  [_profile_timed] iter exception: {e}")
                return _nan_result(e)
        arr = np.array(times)
        return {
            "median_ms": float(np.median(arr)),
            "mean_ms": float(np.mean(arr)),
            "min_ms": float(np.min(arr)),
            "max_ms": float(np.max(arr)),
            "times_ms": times,
        }

    @staticmethod
    def _profile_linear_fit(xs, ys):
        """Least-squares fit y = alpha * x + beta.  Returns (alpha, beta, r2)."""
        xs, ys = np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.float64)
        if len(xs) < 2:
            return (float("nan"),) * 3
        A = np.vstack([xs, np.ones_like(xs)]).T
        result = np.linalg.lstsq(A, ys, rcond=None)
        alpha, beta = result[0]
        ss_res = np.sum((ys - (alpha * xs + beta)) ** 2)
        ss_tot = np.sum((ys - np.mean(ys)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        return float(alpha), float(beta), float(r2)

    # ---- main profiling entry point ----

    def profile_substages(
        self,
        n_sm_levels: int = 5,
        iters: int = 3,
        warmup: int = 1,
        response_len: int = 128,
        prefill_prompt_lens: list[int] | None = None,
        decode_large_batch_sizes: list[int] | None = None,
        decode_small_batch_sizes: list[int] | None = None,
    ) -> dict:
        """Profile each sub-stage at different Green-Context SM partition levels.

        Sub-stages
        ----------
        1. **Prefill proxy** – ``compute_log_prob`` with varying total tokens
           (large prompt, minimum batch).
        2. **Decode-large proxy** – ``compute_log_prob`` with batch 128-1024,
           short sequence.
        3. **Decode-small proxy** – ``compute_log_prob`` with batch < 128,
           short sequence.
        4. **compute_ref_log_prob** – reference-model forward.
        5. **update_actor** – full backward + optimizer step (WARNING: mutates
           weights; restored via ``update_weights`` afterward).

        For each sub-stage × each SM level the median latency (ms) is recorded.
        A linear fit ``time = α·tokens + β`` is computed for the prefill proxy.

        Returns
        -------
        dict   Nested results suitable for further analysis / plotting.
        """
        rtprint("[profile_substages] ENTER")
        sys.stdout.flush()

        _ts = lambda: time.strftime("%H:%M:%S")

        rtprint("[profile_substages] querying SM count from workers ...")
        sys.stdout.flush()
        total_sms = self._profile_get_sm_count()
        rtprint(f"[profile_substages] total_sms={total_sms}")
        sys.stdout.flush()
        sm_levels = self._profile_compute_sm_levels(total_sms, n_sm_levels)

        dp_size = self._get_dp_size(self.actor_rollout_wg_a, "actor")
        rtprint(f"[profile_substages] dp_size={dp_size}")
        sys.stdout.flush()

        micro_bs = OmegaConf.select(self.cfg_a, "rollout.log_prob_micro_batch_size_per_gpu", default=None)
        if micro_bs is None:
            micro_bs = OmegaConf.select(self.cfg_a, "actor.ppo_micro_batch_size_per_gpu", default=4)
        micro_bs = int(micro_bs)
        min_batch = dp_size * micro_bs  # minimum valid global batch

        if prefill_prompt_lens is None:
            prefill_prompt_lens = [256, 512, 1024, 2048, 4096]
        prefill_bs = min_batch

        def _align_up(bs):
            """Round up to nearest multiple of min_batch."""
            return max(min_batch, ((bs + min_batch - 1) // min_batch) * min_batch)

        if decode_large_batch_sizes is None:
            decode_large_batch_sizes = sorted(set(_align_up(bs) for bs in [128, 256, 512, 1024]))
        if decode_small_batch_sizes is None:
            candidates = sorted(set(_align_up(bs) for bs in [16, 32, 64, 96]))
            candidates = [bs for bs in candidates if bs < 128]
            decode_small_batch_sizes = candidates if candidates else [min_batch]
        decode_seq_len = 128

        rtprint(f"\n{'='*80}")
        rtprint(f"[{_ts()}] PROFILING START")
        rtprint(f"  GPU SMs          : {total_sms}")
        rtprint(f"  SM levels        : {sm_levels}")
        rtprint(f"  dp_size          : {dp_size}")
        rtprint(f"  micro_bs/gpu     : {micro_bs}")
        rtprint(f"  min_batch(global): {min_batch}  (dp_size * micro_bs)")
        rtprint(f"  prefill_bs       : {prefill_bs}")
        rtprint(f"  prefill_prompts  : {prefill_prompt_lens}")
        rtprint(f"  decode_large_bs  : {decode_large_batch_sizes}")
        rtprint(f"  decode_small_bs  : {decode_small_batch_sizes}")
        rtprint(f"  response_len     : {response_len}")
        rtprint(f"  iters/warmup     : {iters}/{warmup}")
        rtprint(f"{'='*80}\n")

        results: dict = {
            "prefill_proxy": {},
            "decode_large_proxy": {},
            "decode_small_proxy": {},
            "ref_log_prob": {},
            "update_actor": {},
            "fits": {},
        }

        def _print_phase_summary(phase_name, data):
            """Print a one-line summary right after a phase completes."""
            if not data:
                rtprint(f"  [{phase_name}] (no data)")
                sys.stdout.flush()
                return
            medians = [v.get("median_ms", float("nan")) for v in data.values()]
            valid = [m for m in medians if m == m]  # filter nan
            if valid:
                rtprint(f"  [{phase_name}] DONE  n={len(medians)}  median_range=[{min(valid):.1f}, {max(valid):.1f}] ms")
            else:
                rtprint(f"  [{phase_name}] DONE  n={len(medians)}  (all nan)")
            sys.stdout.flush()

        def _sm_label(sm):
            return total_sms if sm is None else sm

        def _set_gc(sm):
            if sm is not None:
                self.actor_rollout_wg_a.set_green_context(sm)

        def _clear_gc(sm):
            if sm is not None:
                self.actor_rollout_wg_a.clear_green_context()

        # ------------------------------------------------------------------
        # Phase 1 – Forward pass profiling  (prefill proxy)
        # ------------------------------------------------------------------
        rtprint(f"[{_ts()}] === Phase 1: prefill proxy (compute_log_prob, vary total tokens) ===")
        sys.stdout.flush()
        for sm in sm_levels:
            lbl = _sm_label(sm)
            rtprint(f"  [P1] setting green_context sm={lbl} (None=default) ...")
            sys.stdout.flush()
            _set_gc(sm)
            for plen in prefill_prompt_lens:
                batch = self._profile_make_batch(prefill_bs, plen, response_len)
                total_tok = prefill_bs * (plen + response_len)

                t = self._profile_timed(
                    lambda b=batch: self._compute_old_log_prob("a", deepcopy(b)),
                    iters=iters, warmup=warmup,
                )
                results["prefill_proxy"][(total_tok, lbl)] = t
                rtprint(f"  SM={lbl:>4d}  tokens={total_tok:>8d}  median={t['median_ms']:>8.1f}ms")
                sys.stdout.flush()
            _clear_gc(sm)
        _print_phase_summary("Phase 1 prefill_proxy", results["prefill_proxy"])

        # ------------------------------------------------------------------
        # Phase 2 – Decode-large proxy  (batch 128-1024, short seq)
        # ------------------------------------------------------------------
        rtprint(f"\n[{_ts()}] === Phase 2: decode-large proxy (batch 128-1024) ===")
        sys.stdout.flush()
        for sm in sm_levels:
            lbl = _sm_label(sm)
            rtprint(f"  [P2] setting green_context sm={lbl} ...")
            sys.stdout.flush()
            _set_gc(sm)
            for bs in decode_large_batch_sizes:
                batch = self._profile_make_batch(bs, decode_seq_len, response_len)

                t = self._profile_timed(
                    lambda b=batch: self._compute_old_log_prob("a", deepcopy(b)),
                    iters=iters, warmup=warmup,
                )
                results["decode_large_proxy"][(bs, lbl)] = t
                rtprint(f"  SM={lbl:>4d}  bs={bs:>5d}  median={t['median_ms']:>8.1f}ms")
                sys.stdout.flush()
            _clear_gc(sm)
        _print_phase_summary("Phase 2 decode_large_proxy", results["decode_large_proxy"])

        # ------------------------------------------------------------------
        # Phase 3 – Decode-small proxy  (batch < 128, short seq)
        # ------------------------------------------------------------------
        rtprint(f"\n[{_ts()}] === Phase 3: decode-small proxy (batch < 128) ===")
        sys.stdout.flush()
        for sm in sm_levels:
            lbl = _sm_label(sm)
            rtprint(f"  [P3] setting green_context sm={lbl} ...")
            sys.stdout.flush()
            _set_gc(sm)
            for bs in decode_small_batch_sizes:
                batch = self._profile_make_batch(bs, decode_seq_len, response_len)

                t = self._profile_timed(
                    lambda b=batch: self._compute_old_log_prob("a", deepcopy(b)),
                    iters=iters, warmup=warmup,
                )
                results["decode_small_proxy"][(bs, lbl)] = t
                rtprint(f"  SM={lbl:>4d}  bs={bs:>5d}  median={t['median_ms']:>8.1f}ms")
                sys.stdout.flush()
            _clear_gc(sm)
        _print_phase_summary("Phase 3 decode_small_proxy", results["decode_small_proxy"])

        # ------------------------------------------------------------------
        # Phase 4 – compute_ref_log_prob
        # ------------------------------------------------------------------
        rtprint(f"\n[{_ts()}] === Phase 4: compute_ref_log_prob ===")
        sys.stdout.flush()
        ref_batch = self._profile_make_batch(prefill_bs, 512, response_len)
        for sm in sm_levels:
            lbl = _sm_label(sm)
            rtprint(f"  [P4] setting green_context sm={lbl} ...")
            sys.stdout.flush()
            _set_gc(sm)
            t = self._profile_timed(
                lambda: self._compute_ref_log_prob_via_actor("a", deepcopy(ref_batch)),
                iters=iters, warmup=warmup,
            )
            results["ref_log_prob"][lbl] = t
            rtprint(f"  SM={lbl:>4d}  median={t['median_ms']:>8.1f}ms")
            sys.stdout.flush()
            _clear_gc(sm)
        _print_phase_summary("Phase 4 ref_log_prob", results["ref_log_prob"])

        # ------------------------------------------------------------------
        # Phase 5 – update_actor  (modifies weights!)
        # ------------------------------------------------------------------
        rtprint(f"\n[{_ts()}] === Phase 5: update_actor (will restore weights afterward) ===")
        sys.stdout.flush()

        # update_actor requires batch_size % (ppo_mini_batch_size // dp_size) == 0 per GPU.
        # ppo_mini_batch_size = actor.ppo_mini_batch_size * rollout.n
        ppo_mini = int(OmegaConf.select(self.cfg_a, "actor.ppo_mini_batch_size", default=256))
        rollout_n = int(OmegaConf.select(self.cfg_a, "rollout.n", default=4))
        ppo_mini_batch_size = ppo_mini * rollout_n
        update_actor_bs = max(min_batch, ppo_mini_batch_size)
        update_actor_bs = _align_up(update_actor_bs)  # ensure divisible by dp_size
        rtprint(f"  [P5] update_actor_bs={update_actor_bs} (ppo_mini_batch_size={ppo_mini_batch_size})")
        sys.stdout.flush()

        actor_batch = self._profile_make_batch(update_actor_bs, 512, response_len)
        self.actor_rollout_wg_a.clear_green_context()
        rtprint(f"  [P5] computing old_log_prob for actor_batch ...")
        sys.stdout.flush()
        old_lp, _ = self._compute_old_log_prob("a", deepcopy(actor_batch))
        actor_batch.batch["old_log_probs"] = old_lp.batch["old_log_probs"]
        actor_batch.batch["advantages"] = torch.randn(
            update_actor_bs, response_len, dtype=torch.float32
        )

        if self.config.algorithm.get("use_kl_loss", False):
            ref_lp = self._compute_ref_log_prob_via_actor("a", deepcopy(actor_batch))
            actor_batch.batch["ref_log_prob"] = ref_lp.batch["ref_log_prob"]

        for sm in sm_levels:
            lbl = _sm_label(sm)
            rtprint(f"  [P5] setting green_context sm={lbl} ...")
            sys.stdout.flush()
            _set_gc(sm)
            t = self._profile_timed(
                lambda: self._update_actor("a", deepcopy(actor_batch)),
                iters=iters, warmup=warmup,
            )
            results["update_actor"][lbl] = t
            rtprint(f"  SM={lbl:>4d}  median={t['median_ms']:>8.1f}ms")
            sys.stdout.flush()
            _clear_gc(sm)
        _print_phase_summary("Phase 5 update_actor", results["update_actor"])

        rtprint(f"\n[{_ts()}] Restoring weights after update_actor profiling ...")
        sys.stdout.flush()
        self.checkpoint_manager_a.update_weights()
        rtprint(f"[{_ts()}] Weights restored.")
        sys.stdout.flush()

        # ------------------------------------------------------------------
        # Phase 6 – Linear fits for prefill proxy
        # ------------------------------------------------------------------
        rtprint(f"\n[{_ts()}] === Computing linear fits ===")
        sys.stdout.flush()
        for sm in sm_levels:
            lbl = _sm_label(sm)
            tok_list, time_list = [], []
            for (tok, s), v in results["prefill_proxy"].items():
                if s == lbl:
                    tok_list.append(tok)
                    time_list.append(v["median_ms"])
            alpha, beta, r2 = self._profile_linear_fit(tok_list, time_list)
            results["fits"][lbl] = {"alpha_ms_per_token": alpha, "beta_ms": beta, "r2": r2}
            rtprint(f"  SM={lbl:>4d}  α={alpha:.6f} ms/tok  β={beta:.2f} ms  R²={r2:.4f}")

        # ------------------------------------------------------------------
        # Phase 7 – Summary table
        # ------------------------------------------------------------------
        sm_labels = [_sm_label(s) for s in sm_levels]
        self._profile_print_report(results, total_sms, sm_labels,
                                   prefill_prompt_lens, prefill_bs, response_len,
                                   decode_large_batch_sizes, decode_small_batch_sizes,
                                   decode_seq_len)

        rtprint(f"\n[{_ts()}] PROFILING COMPLETE")
        return results

    def _profile_print_report(
        self, results, total_sms, sm_levels,
        prefill_prompt_lens, prefill_bs, response_len,
        decode_large_bs, decode_small_bs, decode_seq_len,
    ):
        def _v(d, key):
            e = d.get(key)
            if e is None:
                return "    ---"
            return f"{e['median_ms']:>7.1f}"

        rtprint(f"\n{'='*100}")
        rtprint(f"  PROFILING REPORT    GPU SMs={total_sms}")
        rtprint(f"{'='*100}")

        # --- prefill proxy table ---
        tok_labels = [prefill_bs * (pl + response_len) for pl in prefill_prompt_lens]
        header = f"{'SM':>6s}" + "".join(f"{'tok='+str(t):>12s}" for t in tok_labels)
        rtprint(f"\n  Prefill Proxy (compute_log_prob, median ms)")
        rtprint(f"  {header}")
        rtprint(f"  {'-'*len(header)}")
        for sm in sm_levels:
            row = f"{sm:>6d}"
            for tok in tok_labels:
                row += f"{_v(results['prefill_proxy'], (tok, sm)):>12s}"
            rtprint(f"  {row}")

        # --- decode large proxy table ---
        header = f"{'SM':>6s}" + "".join(f"{'bs='+str(b):>10s}" for b in decode_large_bs)
        rtprint(f"\n  Decode-Large Proxy (bs 128-1024, seq={decode_seq_len}, median ms)")
        rtprint(f"  {header}")
        rtprint(f"  {'-'*len(header)}")
        for sm in sm_levels:
            row = f"{sm:>6d}"
            for bs in decode_large_bs:
                row += f"{_v(results['decode_large_proxy'], (bs, sm)):>10s}"
            rtprint(f"  {row}")

        # --- decode small proxy table ---
        header = f"{'SM':>6s}" + "".join(f"{'bs='+str(b):>10s}" for b in decode_small_bs)
        rtprint(f"\n  Decode-Small Proxy (bs < 128, seq={decode_seq_len}, median ms)")
        rtprint(f"  {header}")
        rtprint(f"  {'-'*len(header)}")
        for sm in sm_levels:
            row = f"{sm:>6d}"
            for bs in decode_small_bs:
                row += f"{_v(results['decode_small_proxy'], (bs, sm)):>10s}"
            rtprint(f"  {row}")

        # --- ref_log_prob + update_actor ---
        header = f"{'SM':>6s}{'ref_log_prob':>14s}{'update_actor':>14s}"
        rtprint(f"\n  Training Sub-stages (median ms)")
        rtprint(f"  {header}")
        rtprint(f"  {'-'*len(header)}")
        for sm in sm_levels:
            ref = _v(results["ref_log_prob"], sm)
            upd = _v(results["update_actor"], sm)
            rtprint(f"  {sm:>6d}{ref:>14s}{upd:>14s}")

        # --- linear fits ---
        rtprint(f"\n  Prefill Linear Fit:  time = α · total_tokens + β")
        rtprint(f"  {'SM':>6s}{'α (ms/tok)':>14s}{'β (ms)':>10s}{'R²':>8s}")
        rtprint(f"  {'-'*38}")
        for sm in sm_levels:
            f = results["fits"].get(sm, {})
            a = f.get("alpha_ms_per_token", float("nan"))
            b = f.get("beta_ms", float("nan"))
            r = f.get("r2", float("nan"))
            rtprint(f"  {sm:>6d}{a:>14.6f}{b:>10.2f}{r:>8.4f}")

        rtprint(f"{'='*100}")

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

@dataclass
class StepProfiler:
    enabled: bool = True
    verbose: bool = False                 # 是否每步/每rollout打印
    print_every: int = 10                 # 每多少 step 打一次汇总
    window: int = 50                      # 滑动窗口大小
    print_rollout_each: bool = False      # 是否每次 rollout 都打印一次（会很吵）

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
            "gen_a", "reward_a", "old_log_prob_a", "ref_log_prob_a", "values_a", "adv_a",
            "gen_b", "reward_b", "old_log_prob_b", "ref_log_prob_b", "values_b", "adv_b",
            "update_critic_a", "update_critic_b",
            "update_actor_a", "update_actor_b",
            "save_checkpoint", "update_weights_a", "update_weights_b",
            "testing",
        ]

        # 只打印出现过的 key
        present = [k for k in keys if k in self._hist]

        # 一行：滑动均值（window）+ 总均值
        parts = [f"[profile] step={step} window={self.window}"]
        for k in present:
            parts.append(f"{k}: {self.mean(k)*1000:.1f}ms (avg), {self.total_mean(k)*1000:.1f}ms (all)")
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
            if orig_critic_cfg.strategy == "fsdp":
                engine_config: FSDPEngineConfig = orig_critic_cfg.model.fsdp_config
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
    # One actor step
    # -------------------------
    def _step_one_actor(self, which: str, batch: DataProto, timing_raw: dict, metrics: dict):
        actor_cfg = self._actor_cfg(which)
        async_mgr = self._async_mgr(which)
        ckpt_mgr = self._ckpt_mgr(which)

        batch.meta_info["temperature"] = actor_cfg.rollout.temperature
        batch.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object)

        gen_batch = self._get_gen_batch(batch)
        gen_batch.meta_info["global_steps"] = self.global_steps
        gen_batch_output = gen_batch.repeat(repeat_times=actor_cfg.rollout.n, interleave=True)

        with marked_timer(f"gen_{which}", timing_raw, color="red"):
            gen_batch_output = async_mgr.generate_sequences(gen_batch_output)
            ckpt_mgr.sleep_replicas()
            timing_raw.update(gen_batch_output.meta_info.get("timing", {}))
            gen_batch_output.meta_info.pop("timing", None)

        if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
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
        batch = batch.union(gen_batch_output)

        if "response_mask" not in batch.batch.keys():
            batch.batch["response_mask"] = compute_response_mask(batch)

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
            logger.log(data=val_a, step=self.global_steps)
            logger.log(data=val_b, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        # 从 config 里读（你也可以放 trainer.profile.*）
        prof_cfg = self.config.trainer.get("profile", {})
        self.profiler = StepProfiler(
            enabled=prof_cfg.get("enabled", False),
            verbose=prof_cfg.get("verbose", False),
            print_every=prof_cfg.get("print_every", 10),
            window=prof_cfg.get("window", 50),
            print_rollout_each=prof_cfg.get("print_rollout_each", False),
        )

        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Dual Training Progress")

        self.global_steps += 1
        last_val_metrics = None
        self.max_steps_duration = 0

        for epoch in range(current_epoch, self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}
                timing_raw = {}

                batch_base: DataProto = DataProto.from_single_dict(batch_dict)
                is_last_step = self.global_steps >= self.total_training_steps

                with marked_timer("step", timing_raw):
                    batch_a = deepcopy(batch_base)
                    batch_a, _ = self._step_one_actor("a", batch_a, timing_raw, metrics)

                    batch_b = deepcopy(batch_base)
                    batch_b, _ = self._step_one_actor("b", batch_b, timing_raw, metrics)

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
                            is_last_step
                            or self.global_steps % self.config.trainer.save_freq == 0
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
                    and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
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

                logger.log(data=metrics, step=self.global_steps)

                progress_bar.update(1)
                self.global_steps += 1

                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

                if getattr(self, "profiler", None) is not None:
                    self.profiler.update(timing_raw=timing_raw, step=self.global_steps)
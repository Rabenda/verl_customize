# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
N-model co-located PPO trainer (MultiRayPPOTrainer).

Config:
  config.trainer.num_models = N   (2 .. 8)
  config.actor_rollout_ref_0
  config.actor_rollout_ref_1
  ...
  config.actor_rollout_ref_{N-1}

Training pipeline per step:
  1. Start ALL N rollouts simultaneously (streaming via sglang)
  2. Poll until every rollout completes
  3. For n in 0..N-1 (serial): reward → old_log_prob → ref_log_prob → advantage
  4. For n in 0..N-1 (serial): update_actor (backward+optimizer)
  5. For n in 0..N-1 (serial): update_weights (FSDP→sglang sync)

Usage:
  Set config.trainer.fit_method = "multi_overlap_decode"
       config.trainer.num_models = 3   # or 2/4/5/...
  and provide actor_rollout_ref_0 / _1 / _2 in your yaml.
"""

from __future__ import annotations

import os
import uuid
from copy import deepcopy
from typing import Optional

import numpy as np
import ray
import torch
from omegaconf import OmegaConf, open_dict
from tensordict import TensorDict
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
from verl.trainer.ppo.reward import compute_reward, compute_reward_async
from verl.trainer.ppo.utils import Role, WorkerType, need_critic, need_reward_model
from verl.utils import tensordict_utils as tu
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path
from verl.utils.debug import marked_timer
from verl.utils.import_utils import load_class_from_fqn
from verl.utils.model import compute_position_id_with_mask
from verl.utils.py_functional import rename_dict
from verl.utils.torch_functional import masked_mean
from verl.workers.utils.padding import left_right_2_no_padding, no_padding_2_padding

import time

# ---------------------------------------------------------------------------
# Role list for N-model indexed access (max 8 models)
# ---------------------------------------------------------------------------
_MULTI_ROLES = [
    Role.ActorRolloutM0,
    Role.ActorRolloutM1,
    Role.ActorRolloutM2,
    Role.ActorRolloutM3,
    Role.ActorRolloutM4,
    Role.ActorRolloutM5,
    Role.ActorRolloutM6,
    Role.ActorRolloutM7,
]
_MAX_MODELS = len(_MULTI_ROLES)


# ---------------------------------------------------------------------------
# Helpers (copied from ray_dual_trainer)
# ---------------------------------------------------------------------------
def _compute_response_mask(data: DataProto):
    responses = data.batch["responses"]
    response_length = responses.size(1)
    attention_mask = data.batch["attention_mask"]
    return attention_mask[:, -response_length:]


def _apply_kl_penalty(data: DataProto, kl_ctrl, kl_penalty="kl"):
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


def _compute_advantage(data: DataProto, adv_estimator, gamma, lam, num_repeat, norm_adv_by_std_in_grpo, config):
    if "response_mask" not in data.batch.keys():
        data.batch["response_mask"] = _compute_response_mask(data)

    if adv_estimator == AdvantageEstimator.GAE:
        advantages, returns = core_algos.compute_gae_advantage_return(
            token_level_rewards=data.batch["token_level_rewards"],
            values=data.batch["values"],
            response_mask=data.batch["response_mask"],
            gamma=gamma,
            lam=lam,
        )
    elif adv_estimator == AdvantageEstimator.GRPO:
        advantages, returns = core_algos.compute_grpo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=data.batch["response_mask"],
            index=data.non_tensor_batch["uid"],
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        )
    else:
        adv_estimator_fn = core_algos.get_adv_estimator_fn(adv_estimator)
        adv_kwargs = {
            "token_level_rewards": data.batch["token_level_rewards"],
            "response_mask": data.batch["response_mask"],
            "config": config,
        }
        if "uid" in data.non_tensor_batch:
            adv_kwargs["index"] = data.non_tensor_batch["uid"]
        advantages, returns = adv_estimator_fn(**adv_kwargs)

    data.batch["advantages"] = advantages
    data.batch["returns"] = returns
    return data


# ---------------------------------------------------------------------------
# MultiRayPPOTrainer
# ---------------------------------------------------------------------------
class MultiRayPPOTrainer:
    """
    N-model co-located PPO trainer.

    All N models share the same GPU pool (colocated with sglang).
    Rollout: all N models run simultaneously (parallel streaming).
    Train:   each model trains serially (reward → logprob → adv → update).
    Weights: synced serially after all trains complete.

    Config keys:
      config.trainer.num_models          : int (2..8)
      config.actor_rollout_ref_0         : per-model config (same schema as actor_rollout_ref)
      config.actor_rollout_ref_1         : ...
      ...

    Use fit_method = "multi_overlap_decode" in config.trainer.
    """

    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict,
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: type = RayWorkerGroup,
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

        self.N = int(config.trainer.num_models)
        assert 2 <= self.N <= _MAX_MODELS, f"num_models must be 2..{_MAX_MODELS}, got {self.N}"

        self.cfgs = []
        for i in range(self.N):
            cfg = getattr(config, f"actor_rollout_ref_{i}", None)
            assert cfg is not None, (
                f"config.actor_rollout_ref_{i} not found. "
                f"Provide actor_rollout_ref_0 .. actor_rollout_ref_{self.N - 1}."
            )
            self.cfgs.append(cfg)

        assert config.trainer.get("use_legacy_worker_impl", "auto") == "disable", (
            "MultiRayPPOTrainer requires use_legacy_worker_impl=disable"
        )
        for i in range(self.N):
            assert self.cfgs[i].model.get("lora", {}).get("rank", 0) <= 0, (
                f"lora not supported in multi-model mode (model {i})"
            )

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.ray_worker_group_cls = ray_worker_group_cls
        self.device_name = device_name or config.trainer.device

        self.use_rm = need_reward_model(role_worker_mapping)
        self.use_reward_loop = config.reward_model.use_reward_loop
        self.use_critic = need_critic(config)
        assert not self.use_critic, (
            "MultiRayPPOTrainer currently only supports GRPO (no critic). "
            "Set critic.enable=false."
        )
        if self.use_rm and not self.use_reward_loop:
            raise RuntimeError("Multi trainer requires use_reward_loop=True if reward model is enabled.")

        if config.algorithm.use_kl_in_reward:
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(config.algorithm.kl_ctrl)

        self.async_rollout_mode = True

        self._create_dataloader(train_dataset, val_dataset, collate_fn, train_sampler)

    # -------------------------------------------------------------------------
    # Dataloader
    # -------------------------------------------------------------------------
    def _create_dataloader(self, train_dataset, val_dataset, collate_fn, train_sampler):
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

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

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

        assert len(self.train_dataloader) >= 1
        assert len(self.val_dataloader) >= 1

        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs
        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps
        self.total_training_steps = total_training_steps

        try:
            OmegaConf.set_struct(self.config, True)
            with open_dict(self.config):
                for i in range(self.N):
                    key = f"actor_rollout_ref_{i}"
                    if OmegaConf.select(self.config, f"{key}.actor.optim"):
                        getattr(self.config, key).actor.optim.total_training_steps = total_training_steps
        except Exception as e:
            print(f"Warning: could not set total_training_steps: {e}")

    # -------------------------------------------------------------------------
    # Worker init
    # -------------------------------------------------------------------------
    def init_workers(self):
        self.resource_pool_manager.create_resource_pool()
        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # Verify all N roles are registered
        for i in range(self.N):
            role = _MULTI_ROLES[i]
            assert role in self.role_worker_mapping, (
                f"Missing role {role} in role_worker_mapping. "
                f"Call add_actor_rollout_workers_multi(config) to register all {self.N} roles."
            )

        # All N models must be in the same pool (colocated)
        pool = self.resource_pool_manager.get_resource_pool(_MULTI_ROLES[0])
        for i in range(1, self.N):
            pool_i = self.resource_pool_manager.get_resource_pool(_MULTI_ROLES[i])
            assert pool_i == pool, f"All N models must be in the same resource pool (model {i} is in a different pool)"

        for i in range(self.N):
            role = _MULTI_ROLES[i]
            actor_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[role],
                config=self.cfgs[i],
                role=str(role),
            )
            self.resource_pool_to_cls[pool][str(role)] = actor_cls

        # Spawn colocated worker groups
        wg_kwargs = {}
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout
        wg_kwargs["device_name"] = self.device_name

        all_wg = {}
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

        self.actor_rollout_wgs = [all_wg[str(_MULTI_ROLES[i])] for i in range(self.N)]
        for wg in self.actor_rollout_wgs:
            wg.init_model()

        # Reward loop manager (shared)
        if self.use_reward_loop:
            from verl.experimental.reward_loop import RewardLoopManager
            rm_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel) if self.use_rm else None
            self.reward_loop_manager = RewardLoopManager(config=self.config, rm_resource_pool=rm_pool)
        else:
            self.reward_loop_manager = None

        enable_agent_reward_loop = (
            self.use_reward_loop and ((not self.use_rm) or self.config.reward_model.enable_resource_pool)
        )
        reward_loop_handles = (
            self.reward_loop_manager.reward_loop_workers if enable_agent_reward_loop else None
        )

        # Async rollout managers (one per model)
        from verl.experimental.agent_loop import AgentLoopManager as DefaultAgentLoopManager

        self.async_rollout_managers = []
        for i in range(self.N):
            manager_fqn = self.cfgs[i].rollout.get("agent", {}).get("agent_loop_manager_class")
            ManagerCls = (
                load_class_from_fqn(manager_fqn, "AgentLoopManager") if manager_fqn
                else DefaultAgentLoopManager
            )
            mgr = ManagerCls(
                config=self._build_actor_scoped_config_n(i),
                worker_group=self.actor_rollout_wgs[i],
                rollout_resource_pool=pool,
                reward_loop_worker_handles=reward_loop_handles,
            )
            self.async_rollout_managers.append(mgr)

        # Checkpoint managers (one per model)
        # Create ALL managers first, then sleep ALL together — same as dual trainer.
        # Never sleep one before the next is created; sequential sleep+init risks
        # the next server claiming the same physical GPU memory the previous one just freed.
        self.checkpoint_managers = []
        for i in range(self.N):
            ckpt = CheckpointEngineManager(
                backend=self.cfgs[i].rollout.checkpoint_engine.backend,
                trainer=self.actor_rollout_wgs[i],
                replicas=self.async_rollout_managers[i].rollout_replicas,
            )
            self.checkpoint_managers.append(ckpt)
        for i in range(self.N):
            self.checkpoint_managers[i].sleep_replicas()

        print(f"[MultiRayPPOTrainer] init_workers done for {self.N} models", flush=True)

    # -------------------------------------------------------------------------
    # Helpers: indexed access
    # -------------------------------------------------------------------------
    def _cfg(self, n: int):
        return self.cfgs[n]

    def _wg(self, n: int):
        return self.actor_rollout_wgs[n]

    def _mgr(self, n: int):
        return self.async_rollout_managers[n]

    def _ckpt(self, n: int):
        return self.checkpoint_managers[n]

    def _build_actor_scoped_config_n(self, n: int):
        """Build a full top-level config with actor_rollout_ref swapped to model n."""
        cfg = deepcopy(self.config)
        actor_sub = deepcopy(self.cfgs[n])
        with open_dict(cfg):
            cfg.actor_rollout_ref = actor_sub
        return cfg

    # -------------------------------------------------------------------------
    # Data helpers
    # -------------------------------------------------------------------------
    def _get_gen_batch(self, batch: DataProto) -> DataProto:
        reward_model_keys = set({"data_source", "reward_model", "extra_info", "uid"}) & batch.non_tensor_batch.keys()
        non_tensor_batch_keys_to_pop = set(batch.non_tensor_batch.keys()) - reward_model_keys
        gen_batch = batch.pop(batch_keys=[], non_tensor_batch_keys=list(non_tensor_batch_keys_to_pop))
        if self.async_rollout_mode:
            gen_batch.non_tensor_batch.update(batch.non_tensor_batch)
        return gen_batch

    # -------------------------------------------------------------------------
    # Compute methods (indexed by model n)
    # -------------------------------------------------------------------------
    def _compute_old_log_prob_n(self, n: int, batch: DataProto):
        actor_wg = self._wg(n)
        batch_td = batch.to_tensordict()
        batch_td = left_right_2_no_padding(batch_td)
        tu.assign_non_tensor(batch_td, calculate_entropy=True, compute_loss=False)
        output = actor_wg.compute_log_prob(batch_td)
        entropy = tu.get(output, "entropy")
        log_probs = tu.get(output, "log_probs")
        metrics = tu.get(output, "metrics", default={})
        old_log_prob_mfu = metrics.get("mfu", None)
        entropy = no_padding_2_padding(entropy, batch_td)
        log_probs = no_padding_2_padding(log_probs, batch_td)
        old_log_prob = tu.get_tensordict({"old_log_probs": log_probs.float(), "entropys": entropy.float()})
        return DataProto.from_tensordict(old_log_prob), old_log_prob_mfu

    def _compute_ref_log_prob_via_actor_n(self, n: int, batch: DataProto) -> DataProto:
        actor_wg = self._wg(n)
        batch_td = batch.to_tensordict()
        batch_td = left_right_2_no_padding(batch_td)
        tu.assign_non_tensor(batch_td, calculate_entropy=False, compute_loss=False, no_lora_adapter=True)
        output = actor_wg.compute_log_prob(batch_td)
        log_probs = tu.get(output, "log_probs")
        log_probs = no_padding_2_padding(log_probs, batch_td)
        ref_log_prob = tu.get_tensordict({"ref_log_prob": log_probs.float()})
        return DataProto.from_tensordict(ref_log_prob)

    def _update_actor_n(self, n: int, batch: DataProto) -> DataProto:
        actor_cfg = self._cfg(n)
        actor_wg = self._wg(n)

        rollout_cfg = actor_cfg.rollout
        batch.meta_info["multi_turn"] = rollout_cfg.multi_turn.enable
        batch.meta_info["temperature"] = rollout_cfg.temperature

        batch_td = batch.to_tensordict()
        batch_td = left_right_2_no_padding(batch_td)

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
        actor_output = rename_dict(actor_output, f"actor_{n}/")
        if f"actor_{n}/mfu" in actor_output:
            actor_output[f"perf/mfu/actor_{n}"] = actor_output.pop(f"actor_{n}/mfu")
        return DataProto.from_single_dict(data={}, meta_info={"metrics": actor_output})

    def _compute_or_extract_reward(self, batch: DataProto, reward_fn=None, reward_for_val=False, sum_reward=False):
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

    # -------------------------------------------------------------------------
    # One actor step: reward + logprob + advantage
    # Returns (batch_with_advantages, reward_extra_infos_dict)
    # -------------------------------------------------------------------------
    def _step_one_actor_n(
        self,
        n: int,
        batch: DataProto,
        timing_raw: dict,
        metrics: dict,
        gen_batch_output_override: Optional[DataProto] = None,
        sglang_already_slept: bool = False,
    ):
        actor_cfg = self._cfg(n)
        ckpt_mgr = self._ckpt(n)
        label = str(n)

        if gen_batch_output_override is None:
            # Normal (non-overlap) rollout path
            batch.meta_info["temperature"] = actor_cfg.rollout.temperature
            batch.non_tensor_batch["uid"] = np.array(
                [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
            )
            gen_batch = self._get_gen_batch(batch)
            gen_batch.meta_info["global_steps"] = self.global_steps
            gen_batch = gen_batch.repeat(repeat_times=actor_cfg.rollout.n, interleave=True)
            t0 = time.perf_counter()
            gen_batch_output = self._mgr(n).generate_sequences(gen_batch)
            ckpt_mgr.sleep_replicas()
            timing_raw[f"puzzrl_rollout_{n}"] = time.perf_counter() - t0
            timing_raw.update(gen_batch_output.meta_info.get("timing", {}))
            gen_batch_output.meta_info.pop("timing", None)
        else:
            gen_batch_output = gen_batch_output_override
            if sglang_already_slept:
                print(f"[{label}] step={self.global_steps} skip sleep_replicas (already slept in rollout)", flush=True)
            else:
                print(f"[{label}] step={self.global_steps} sleep_replicas start", flush=True)
                ckpt_mgr.sleep_replicas()
                print(f"[{label}] step={self.global_steps} sleep_replicas done", flush=True)
            timing_raw.update(gen_batch_output.meta_info.get("timing", {}))
            gen_batch_output.meta_info.pop("timing", None)

        # Repeat batch to match rollout.n
        batch = batch.repeat(repeat_times=actor_cfg.rollout.n, interleave=True)
        if gen_batch_output_override is not None and gen_batch_output.non_tensor_batch:
            overlap_keys = set(batch.non_tensor_batch.keys()) & set(gen_batch_output.non_tensor_batch.keys())
            for k in overlap_keys:
                gen_batch_output.non_tensor_batch.pop(k, None)
        batch = batch.union(gen_batch_output)

        if "response_mask" not in batch.batch.keys():
            batch.batch["response_mask"] = _compute_response_mask(batch)

        # --- Reward ---
        reward_extra_infos_dict = {}
        print(f"[{label}] step={self.global_steps} reward start", flush=True)
        with marked_timer(f"reward_{n}", timing_raw, color="yellow"):
            if self.use_rm and "rm_scores" not in batch.batch.keys():
                assert self.reward_loop_manager is not None
                reward_tensor_proto = self.reward_loop_manager.compute_rm_score(batch)
                batch = batch.union(reward_tensor_proto)

            if self.config.reward_model.launch_reward_fn_async:
                future_reward = compute_reward_async.remote(data=batch, config=self.config, tokenizer=self.tokenizer)
            else:
                reward_tensor, reward_extra_infos_dict = self._compute_or_extract_reward(
                    batch, reward_fn=self.reward_fn, reward_for_val=False
                )
                future_reward = None
        print(f"[{label}] step={self.global_steps} reward done", flush=True)

        # --- Old log prob ---
        with marked_timer(f"old_log_prob_{n}", timing_raw, color="blue"):
            old_log_prob, old_log_prob_mfu = self._compute_old_log_prob_n(n, batch)
            entropys = old_log_prob.batch["entropys"]
            response_masks = batch.batch["response_mask"]
            entropy_agg = agg_loss(
                loss_mat=entropys,
                loss_mask=response_masks,
                loss_agg_mode=actor_cfg.actor.loss_agg_mode,
                loss_scale_factor=actor_cfg.actor.loss_scale_factor,
            )
            metrics.update({
                f"actor_{n}/entropy": float(entropy_agg.detach().item()),
            })
            if old_log_prob_mfu is not None:
                metrics[f"perf/mfu/actor_{n}_infer"] = float(old_log_prob_mfu)
            old_log_prob.batch.pop("entropys")
            batch = batch.union(old_log_prob)

        # --- Ref log prob (if KL) ---
        if self.config.algorithm.use_kl_in_reward:
            with marked_timer(f"ref_log_prob_{n}", timing_raw, color="olive"):
                ref_log_prob = self._compute_ref_log_prob_via_actor_n(n, batch)
                batch = batch.union(ref_log_prob)

        # --- Advantage ---
        with marked_timer(f"adv_{n}", timing_raw, color="brown"):
            if future_reward is not None:
                reward_tensor, reward_extra_infos_dict = ray.get(future_reward)

            batch.batch["token_level_scores"] = reward_tensor
            if reward_extra_infos_dict:
                batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

            if self.config.algorithm.use_kl_in_reward:
                batch, kl_metrics = _apply_kl_penalty(
                    batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                )
                metrics.update({f"{k}_{n}": v for k, v in kl_metrics.items()})
            else:
                batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

            norm_adv = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)
            batch = _compute_advantage(
                batch,
                adv_estimator=self.config.algorithm.adv_estimator,
                gamma=self.config.algorithm.gamma,
                lam=self.config.algorithm.lam,
                num_repeat=actor_cfg.rollout.n,
                norm_adv_by_std_in_grpo=norm_adv,
                config=self.config.algorithm,
            )

        return batch, reward_extra_infos_dict

    # -------------------------------------------------------------------------
    # Checkpoint
    # -------------------------------------------------------------------------
    def _save_checkpoint(self):
        from verl.utils.fs import local_mkdir_safe
        local_step_dir = os.path.join(self.config.trainer.default_local_dir, f"global_step_{self.global_steps}")
        local_mkdir_safe(local_step_dir)

        for i in range(self.N):
            actor_dir = os.path.join(local_step_dir, f"actor_{i}")
            actor_remote = (
                None if self.config.trainer.default_hdfs_dir is None
                else os.path.join(
                    self.config.trainer.default_hdfs_dir,
                    f"global_step_{self.global_steps}",
                    f"actor_{i}",
                )
            )
            max_keep = self.config.trainer.get("max_actor_ckpt_to_keep", None)
            self._wg(i).save_checkpoint(actor_dir, actor_remote, self.global_steps, max_ckpt_to_keep=max_keep)

        dataloader_local = os.path.join(local_step_dir, "data.pt")
        torch.save(self.train_dataloader.state_dict(), dataloader_local)

        local_latest = os.path.join(self.config.trainer.default_local_dir, "latest_checkpointed_iteration.txt")
        with open(local_latest, "w") as f:
            f.write(str(self.global_steps))

    def _load_checkpoint(self):
        if self.config.trainer.resume_mode == "disable":
            self.global_steps = 0
            return

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

        for i in range(self.N):
            actor_path = os.path.join(global_step_folder, f"actor_{i}")
            self._wg(i).load_checkpoint(
                actor_path,
                del_local_after_load=self.config.trainer.del_local_ckpt_after_load,
            )

        dataloader_local = os.path.join(global_step_folder, "data.pt")
        if os.path.exists(dataloader_local):
            self.train_dataloader.load_state_dict(torch.load(dataloader_local, weights_only=False))

    # -------------------------------------------------------------------------
    # Entry point
    # -------------------------------------------------------------------------
    def fit(self):
        self.fit_multi_overlap_decode()

    # -------------------------------------------------------------------------
    # fit_multi_overlap_decode
    #
    # All N rollouts start simultaneously (true parallel streaming).
    # Training is serial: model 0 → model 1 → ... → model N-1.
    # Weight sync is serial after all trains.
    # -------------------------------------------------------------------------
    def fit_multi_overlap_decode(self, poll_timeout_ms: int = 20):
        _ts = lambda: time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

        poll_timeout_ms = int(getattr(self.config.trainer, "overlap_poll_timeout_ms", poll_timeout_ms))

        print(f"\n========== ENTER fit_multi_overlap_decode (N={self.N}) ==========", flush=True)

        self.global_steps = 0
        self._load_checkpoint()
        print("[INIT] checkpoint loaded", flush=True)

        max_cr_cfg = int(getattr(self.config.trainer, "max_concurrent_rollout", 3))
        _rollout_batches = [
            list(range(s, min(s + max_cr_cfg, self.N)))
            for s in range(0, self.N, max_cr_cfg)
        ]
        # Only wake the first rollout batch; subsequent batches are woken
        # just before their rollout via update_weights below.
        for i in _rollout_batches[0]:
            self.checkpoint_managers[i].update_weights()
        print(f"[INIT] weights synced for first batch {_rollout_batches[0]}", flush=True)

        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps)
        self.global_steps += 1

        # -------------------------------------------------------------------
        # helpers: prompt ids for streaming
        # -------------------------------------------------------------------
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
                return _messages_to_prompt_ids(list(nt["raw_prompt"][idx]))
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
            if rollout_name == "sglang":
                target_max_tokens = int(rollout_cfg.response_length + rollout_cfg.prompt_length - len(prompt_ids))
            else:
                target_max_tokens = int(rollout_cfg.response_length)
            sampling_params["max_tokens"] = max(0, min(target_max_tokens, max_possible_tokens))
            return sampling_params

        def _build_output(gen_in, prompt_ids_by_handle, result_by_handle, handles, prompt_length, response_length):
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
                    prompts[i, prompt_len - lp: prompt_len] = torch.tensor(pids, dtype=torch.long)
                    prompt_attn[i, prompt_len - lp: prompt_len] = 1
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

        # -------------------------------------------------------------------
        # main loop
        # -------------------------------------------------------------------
        for epoch in range(self.config.trainer.total_epochs):
            print(f"\n[LOOP] epoch={epoch}", flush=True)

            for batch_dict in self.train_dataloader:
                t_step0 = time.perf_counter()
                print(f"\n[{_ts()}] [STEP] global_step={self.global_steps}", flush=True)

                # Create N independent copies of the batch
                base_batch = DataProto.from_single_dict(batch_dict)
                batches = [base_batch] + [deepcopy(base_batch) for _ in range(self.N - 1)]

                # -----------------------------------------------------------
                # STREAMING ROLLOUT: all N models start simultaneously.
                # Uses start_generate_stream (non-blocking) + polling loop,
                # exactly like fit_overlap_decode in ray_dual_trainer.py.
                # -----------------------------------------------------------
                gen_batches = []
                for i in range(self.N):
                    actor_cfg = self._cfg(i)
                    batches[i].meta_info["temperature"] = actor_cfg.rollout.temperature
                    batches[i].non_tensor_batch["uid"] = np.array(
                        [str(uuid.uuid4()) for _ in range(len(batches[i].batch))], dtype=object
                    )
                    gen = self._get_gen_batch(batches[i])
                    gen.meta_info["global_steps"] = self.global_steps
                    gen = gen.repeat(repeat_times=actor_cfg.rollout.n, interleave=True)
                    gen_batches.append(gen)

                # Per-model state
                handles        = [[] for _ in range(self.N)]
                pid_by_handle  = [{} for _ in range(self.N)]
                result_by_hand = [{} for _ in range(self.N)]
                group_ids      = [None] * self.N
                active         = [0] * self.N
                t_starts       = [None] * self.N
                t_dones        = [None] * self.N

                # Rollout batches: at most max_concurrent_rollout models active at once.
                # Before each batch: wake that batch (update_weights).
                # After each batch: sleep that batch, then wake the next.
                # After all batches: all models are sleeping.
                for b_idx, batch_ids in enumerate(_rollout_batches):
                    # batch 0 is already awake (from init / end-of-prev-step).
                    # batches 1+ are woken here.
                    if b_idx > 0:
                        for i in batch_ids:
                            print(f"[{_ts()}] [MODEL {i}] update_weights (wake for batch {b_idx})", flush=True)
                            self.checkpoint_managers[i].update_weights()

                    print(f"[{_ts()}] [ROLLOUT BATCH {b_idx}] models={batch_ids}", flush=True)

                    # Submit all streams in this batch simultaneously (non-blocking)
                    for i in batch_ids:
                        mgr = self._mgr(i)
                        actor_cfg = self._cfg(i)
                        gen = gen_batches[i]
                        t_starts[i] = time.perf_counter()
                        print(f"[{_ts()}] [MODEL {i}] streaming start ({len(gen)} seqs)", flush=True)
                        for j in range(len(gen)):
                            ids = _get_prompt_ids(gen, j)
                            sp  = _stream_sampling_params(actor_cfg, ids)
                            rid = str(uuid.uuid4())
                            ret = mgr.start_generate_stream(
                                prompt_ids=ids,
                                sampling_params=sp,
                                request_id=rid,
                                emit_token_deltas=False,
                                training_global_step=self.global_steps,
                            )
                            h = ret["handle"]
                            handles[i].append(h)
                            pid_by_handle[i][h] = ids
                        active[i] = len(handles[i])
                        group_ids[i] = mgr.register_generate_stream_group(handles[i])["group_id"]
                        print(f"[{_ts()}] [MODEL {i}] registered group, active={active[i]}", flush=True)

                    # Poll until all models in this batch finish
                    poll_iter = 0
                    while any(active[i] > 0 for i in batch_ids):
                        poll_iter += 1
                        for i in batch_ids:
                            if active[i] > 0:
                                st = self._mgr(i).get_generate_stream_group_status(group_ids[i])
                                active[i] = int(st.get("active_count", active[i]))
                                if active[i] == 0 and t_dones[i] is None:
                                    t_dones[i] = time.perf_counter()
                                    print(f"[{_ts()}] [MODEL {i}] done  wall={t_dones[i]-t_starts[i]:.3f}s", flush=True)
                        if poll_iter % 50 == 0:
                            print(f"[{_ts()}] [STATUS] batch={b_idx} iter={poll_iter} active={[active[i] for i in batch_ids]}", flush=True)
                        time.sleep(max(float(poll_timeout_ms) / 1000.0, 0.001))

                    print(f"[{_ts()}] [ROLLOUT BATCH {b_idx}] all done, sleeping", flush=True)
                    # Sleep this batch — no active requests, safe to call
                    for i in batch_ids:
                        self.checkpoint_managers[i].sleep_replicas()

                # All models are now sleeping. Finalize and build outputs.
                gen_outputs = [None] * self.N
                for i in range(self.N):
                    mgr = self._mgr(i)
                    actor_cfg = self._cfg(i)
                    for h in handles[i]:
                        result_by_hand[i][h] = mgr.finalize_generate_stream(h)
                    mgr.clear_generate_stream_group(group_ids[i])
                    gen_outputs[i] = _build_output(
                        gen_batches[i],
                        pid_by_handle[i],
                        result_by_hand[i],
                        handles[i],
                        int(actor_cfg.rollout.prompt_length),
                        int(actor_cfg.rollout.response_length),
                    )
                    print(f"[{_ts()}] [MODEL {i}] output built", flush=True)

                # -----------------------------------------------------------
                # SERIAL TRAIN: all models sleeping, sglang_already_slept=True
                # -----------------------------------------------------------
                timing_raw = {}
                metrics = {}

                t_train0 = time.perf_counter()
                for i in range(self.N):
                    print(f"[{_ts()}] [TRAIN {i}] _step_one_actor_n start", flush=True)
                    batches[i], _ = self._step_one_actor_n(
                        i, batches[i], timing_raw, metrics,
                        gen_batch_output_override=gen_outputs[i],
                        sglang_already_slept=True,
                    )
                    print(f"[{_ts()}] [TRAIN {i}] _update_actor_n start", flush=True)
                    with marked_timer(f"update_actor_{i}", timing_raw, color="red"):
                        self._update_actor_n(i, batches[i])
                    print(f"[{_ts()}] [TRAIN {i}] done", flush=True)
                t_train1 = time.perf_counter()
                train_time = t_train1 - t_train0

                # Wake only the first rollout batch for the next step.
                # Later batches are woken at the start of their batch above.
                for i in _rollout_batches[0]:
                    print(f"[{_ts()}] [TRAIN {i}] update_weights start", flush=True)
                    with marked_timer(f"update_weights_{i}", timing_raw, color="red"):
                        self.checkpoint_managers[i].update_weights()
                    print(f"[{_ts()}] [TRAIN {i}] update_weights done", flush=True)

                step_total = time.perf_counter() - t_step0
                print(
                    f"[{_ts()}] [WALL] step={self.global_steps} "
                    f"train_time={train_time:.3f}s step_total={step_total:.3f}s",
                    flush=True,
                )

                # Checkpoint save
                save_freq = self.config.trainer.get("save_freq", 0)
                if save_freq and self.global_steps % save_freq == 0:
                    self._save_checkpoint()

                progress_bar.update(1)
                self.global_steps += 1

                if self.global_steps >= self.total_training_steps:
                    print("========== EXIT fit_multi_overlap_decode ==========", flush=True)
                    progress_bar.close()
                    return

        print("========== EXIT fit_multi_overlap_decode (epoch end) ==========", flush=True)
        progress_bar.close()

    # -------------------------------------------------------------------------
    # fit_multi_pipeline
    #
    # Pipeline with bounded rollout concurrency + overlapped train.
    #
    # Per step, each model runs in its own thread:
    #   1. Acquire rollout_sem (max_concurrent_rollout slots) → do rollout
    #   2. Release rollout_sem → acquire train_lock (serializes GPU train ops)
    #   3. Do train (sleep_replicas + reward + logprob + adv + update_actor +
    #      update_weights) → release train_lock
    #
    # Result for N=4, max_concurrent_rollout=3:
    #   - At most 3 sglang servers decode simultaneously (avoids VRAM OOM)
    #   - As soon as the first model finishes rollout, it trains while the
    #     remaining 3 keep decoding → rollout/train overlap
    #   - Only 1 model trains at a time (FSDP serialization)
    #
    # Config:
    #   trainer.max_concurrent_rollout  (default 3)
    # -------------------------------------------------------------------------
    def fit_multi_pipeline(self, max_concurrent_rollout: int = 3):
        import threading

        _ts = lambda: time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

        max_concurrent_rollout = int(
            getattr(self.config.trainer, "max_concurrent_rollout", max_concurrent_rollout)
        )
        print(
            f"\n========== ENTER fit_multi_pipeline "
            f"(N={self.N}, max_concurrent_rollout={max_concurrent_rollout}) ==========",
            flush=True,
        )

        self.global_steps = 0
        self._load_checkpoint()
        print("[INIT] checkpoint loaded", flush=True)

        for i in range(self.N):
            self.checkpoint_managers[i].update_weights()
        print("[INIT] all weights synced", flush=True)

        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps)
        self.global_steps += 1

        # ------------------------------------------------------------------
        # Reuse prompt-id helpers from fit_multi_overlap_decode
        # (needed only if future callers want streaming; generate_sequences
        #  is used here so we only need the gen-batch prep helpers)
        # ------------------------------------------------------------------

        for epoch in range(self.config.trainer.total_epochs):
            print(f"\n[LOOP] epoch={epoch}", flush=True)

            for batch_dict in self.train_dataloader:
                t_step0 = time.perf_counter()
                print(f"\n[{_ts()}] [STEP] global_step={self.global_steps}", flush=True)

                # Prepare N independent copies + gen batches
                base_batch = DataProto.from_single_dict(batch_dict)
                batches = [base_batch] + [deepcopy(base_batch) for _ in range(self.N - 1)]
                gen_batches = []
                for i in range(self.N):
                    actor_cfg = self._cfg(i)
                    batches[i].meta_info["temperature"] = actor_cfg.rollout.temperature
                    batches[i].non_tensor_batch["uid"] = np.array(
                        [str(uuid.uuid4()) for _ in range(len(batches[i].batch))], dtype=object
                    )
                    gen = self._get_gen_batch(batches[i])
                    gen.meta_info["global_steps"] = self.global_steps
                    gen = gen.repeat(repeat_times=actor_cfg.rollout.n, interleave=True)
                    gen_batches.append(gen)

                # Per-model results / timing
                gen_outputs   = [None] * self.N
                timing_raws   = [{} for _ in range(self.N)]
                metrics_list  = [{} for _ in range(self.N)]
                errors        = [None] * self.N
                t_rollout_starts = [None] * self.N
                t_rollout_dones  = [None] * self.N
                t_train_starts   = [None] * self.N
                t_train_dones    = [None] * self.N

                # Semaphore: limits concurrent sglang decoders
                rollout_sem = threading.Semaphore(max_concurrent_rollout)
                # Lock: serialises GPU train ops (sleep_replicas / FSDP / update_weights)
                train_lock  = threading.Lock()

                def _work(i):
                    try:
                        # ---- ROLLOUT ----
                        rollout_sem.acquire()
                        try:
                            t_rollout_starts[i] = time.perf_counter()
                            print(f"[{_ts()}] [MODEL {i}] rollout start ({len(gen_batches[i])} seqs)", flush=True)
                            gen_outputs[i] = self._mgr(i).generate_sequences(gen_batches[i])
                            t_rollout_dones[i] = time.perf_counter()
                            print(
                                f"[{_ts()}] [MODEL {i}] rollout done "
                                f"wall={t_rollout_dones[i]-t_rollout_starts[i]:.3f}s",
                                flush=True,
                            )
                        finally:
                            rollout_sem.release()

                        # ---- TRAIN (serialized) ----
                        with train_lock:
                            t_train_starts[i] = time.perf_counter()
                            print(f"[{_ts()}] [MODEL {i}] train start", flush=True)

                            batches[i], _ = self._step_one_actor_n(
                                i, batches[i], timing_raws[i], metrics_list[i],
                                gen_batch_output_override=gen_outputs[i],
                            )
                            with marked_timer(f"update_actor_{i}", timing_raws[i], color="red"):
                                self._update_actor_n(i, batches[i])
                            with marked_timer(f"update_weights_{i}", timing_raws[i], color="red"):
                                self.checkpoint_managers[i].update_weights()

                            t_train_dones[i] = time.perf_counter()
                            print(
                                f"[{_ts()}] [MODEL {i}] train done "
                                f"wall={t_train_dones[i]-t_train_starts[i]:.3f}s",
                                flush=True,
                            )
                    except Exception as e:
                        errors[i] = e

                threads = [
                    threading.Thread(target=_work, args=(i,), daemon=True)
                    for i in range(self.N)
                ]
                for t in threads:
                    t.start()
                for t in threads:
                    t.join()

                for i in range(self.N):
                    if errors[i] is not None:
                        raise RuntimeError(f"Model {i} failed: {errors[i]}") from errors[i]

                # ------ timing summary ------
                timing_raw = {}
                for i in range(self.N):
                    for k, v in timing_raws[i].items():
                        timing_raw[f"{k}"] = v
                    rollout_wall = (
                        (t_rollout_dones[i] - t_rollout_starts[i])
                        if t_rollout_starts[i] and t_rollout_dones[i] else None
                    )
                    train_wall = (
                        (t_train_dones[i] - t_train_starts[i])
                        if t_train_starts[i] and t_train_dones[i] else None
                    )
                    if rollout_wall is not None:
                        timing_raw[f"rollout_wall_{i}"] = rollout_wall
                    if train_wall is not None:
                        timing_raw[f"train_wall_{i}"] = train_wall

                step_total = time.perf_counter() - t_step0
                timing_summary = "  ".join(
                    f"m{i}:rollout={timing_raw.get(f'rollout_wall_{i}', float('nan')):.1f}s"
                    f"+train={timing_raw.get(f'train_wall_{i}', float('nan')):.1f}s"
                    for i in range(self.N)
                )
                print(
                    f"[{_ts()}] [WALL] step={self.global_steps} "
                    f"step_total={step_total:.3f}s  {timing_summary}",
                    flush=True,
                )

                save_freq = self.config.trainer.get("save_freq", 0)
                if save_freq and self.global_steps % save_freq == 0:
                    self._save_checkpoint()

                progress_bar.update(1)
                self.global_steps += 1

                if self.global_steps >= self.total_training_steps:
                    print("========== EXIT fit_multi_pipeline ==========", flush=True)
                    progress_bar.close()
                    return

        print("========== EXIT fit_multi_pipeline (epoch end) ==========", flush=True)
        progress_bar.close()

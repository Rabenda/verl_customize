# Copyright 2025 Meituan Ltd. and/or its affiliates
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
Dual async one-step-off-policy trainer.

GPU layout (equal-size pools):
  pool_a  ←  Model-A rollout (SGLang)  +  Model-B actor (FSDP)  +  Model-B ref (FSDP)
  pool_b  ←  Model-B rollout (SGLang)  +  Model-A actor (FSDP)  +  Model-A ref (FSDP)

Per step (async overlap):
  Step N gen:   gen_a on pool_a  ||  gen_b on pool_b          (concurrent asyncio.gather)
  Step N-1 train: train_a on pool_b  ||  train_b on pool_a    (concurrent asyncio.to_thread)
  → 4 workloads running simultaneously (requires MPS on each pool)

Weight sync (sequential, fast ~20ms each):
  Model-A: pool_b (ActorA) → pool_a (RolloutA)   broadcast src_rank = n_pool_a
  Model-B: pool_a (ActorB) → pool_b (RolloutB)   broadcast src_rank = 0
  Both use one shared NCCL group "dual_sync" = pool_a_workers + pool_b_workers.
"""

from __future__ import annotations

import asyncio
import os
import time
import uuid
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
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.ray import RayClassWithInitArgs, RayWorkerGroup, ResourcePoolManager
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
)
from verl.trainer.ppo.ray_trainer import apply_kl_penalty, compute_advantage, compute_response_mask
from verl.trainer.ppo.reward import compute_reward
from verl.trainer.ppo.utils import Role, WorkerType
from verl.utils.checkpoint.checkpoint_manager import should_save_ckpt_esi
from verl.utils.debug import marked_timer
from verl.utils.metric import reduce_metrics
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions


def _need_reference_policy(config) -> bool:
    alg = config.algorithm
    cfg_a = config.actor_rollout_ref_a
    cfg_b = config.actor_rollout_ref_b
    return (
        alg.use_kl_in_reward
        or cfg_a.actor.get("use_kl_loss", False)
        or cfg_b.actor.get("use_kl_loss", False)
    )


class DualOneStepOffRayTrainer:
    """Dual async one-step-off trainer. See module docstring for layout details."""

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
        device_name: Optional[str] = None,
    ):
        assert hasattr(config, "actor_rollout_ref_a") and hasattr(config, "actor_rollout_ref_b"), (
            "config must have actor_rollout_ref_a and actor_rollout_ref_b"
        )
        self.config = config
        self.tokenizer = tokenizer
        self.processor = processor
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn
        self.cfg_a = config.actor_rollout_ref_a
        self.cfg_b = config.actor_rollout_ref_b
        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.ray_worker_group_cls = ray_worker_group_cls
        self.device_name = device_name or config.trainer.device
        self.use_reference_policy = _need_reference_policy(config)

        lora_rank_a = config.actor_rollout_ref_a.model.get("lora", {}).get("rank", 0)
        lora_rank_b = config.actor_rollout_ref_b.model.get("lora", {}).get("rank", 0)
        self.ref_in_actor_a = lora_rank_a > 0
        self.ref_in_actor_b = lora_rank_b > 0

        if config.algorithm.use_kl_in_reward:
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(config.algorithm.kl_ctrl)

        self._create_dataloader(train_dataset, val_dataset, collate_fn, train_sampler)

    # ------------------------------------------------------------------
    # Dataloader (copied pattern from DualRayPPOTrainer)
    # ------------------------------------------------------------------

    def _create_dataloader(self, train_dataset, val_dataset, collate_fn, train_sampler):
        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler

        if train_dataset is None:
            train_dataset = create_rl_dataset(
                self.config.data.train_files, self.config.data, self.tokenizer, self.processor
            )
        if val_dataset is None:
            val_dataset = create_rl_dataset(
                self.config.data.val_files, self.config.data, self.tokenizer, self.processor
            )
        if train_sampler is None:
            train_sampler = create_rl_sampler(self.config.data, train_dataset)

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        self.train_dataloader = StatefulDataLoader(
            dataset=train_dataset,
            batch_size=self.config.data.train_batch_size,
            num_workers=0,
            drop_last=True,
            collate_fn=collate_fn,
            sampler=train_sampler,
        )

        assert len(self.train_dataloader) >= 1, "train dataloader is empty"
        total_steps = len(self.train_dataloader) * self.config.trainer.total_epochs
        if self.config.trainer.total_training_steps is not None:
            total_steps = self.config.trainer.total_training_steps
        self.total_training_steps = total_steps

        try:
            with open_dict(self.config):
                self.config.actor_rollout_ref_a.actor.optim.total_training_steps = total_steps
                self.config.actor_rollout_ref_b.actor.optim.total_training_steps = total_steps
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Worker initialisation
    # ------------------------------------------------------------------

    def init_workers(self):
        self.resource_pool_manager.create_resource_pool()
        self.resource_pool_to_cls = {
            pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()
        }
        self._build_worker_classes()
        self._spawn_worker_groups()
        self._init_models()
        self._create_weight_sync_group()
        self._init_async_rollout_managers()

    def _build_merged_worker_config(self, which: str):
        """Merge cfg_a/cfg_b overrides onto the base actor_rollout_ref schema (which has all defaults).

        cfg_a/cfg_b are sparse DictConfigs built from CLI `+` args and lack many fields
        (e.g. fsdp_size, multi_turn) that exist in the base actor_rollout_ref schema loaded
        from dp_actor.yaml / rollout.yaml etc. Merging ensures workers get all default fields.
        """
        base = deepcopy(self.config.actor_rollout_ref)
        override = self.cfg_a if which == "a" else self.cfg_b
        with open_dict(base):
            merged = OmegaConf.merge(base, override)
        return merged

    def _build_worker_classes(self):
        """Register colocated worker classes for pool_a and pool_b."""
        # Model-A workers (RolloutA on pool_a, ActorA on pool_b, RefA on pool_b) → cfg_a
        # Model-B workers (ActorB on pool_a, RefB on pool_a, RolloutB on pool_b) → cfg_b
        _model_a_roles = {Role.RolloutA, Role.ActorA, Role.RefA}
        # Cache merged configs for reuse in _async_gen_next_batch, _train_one_model, etc.
        self.worker_cfg_a = self._build_merged_worker_config("a")
        self.worker_cfg_b = self._build_merged_worker_config("b")
        worker_cfg_a = self.worker_cfg_a
        worker_cfg_b = self.worker_cfg_b

        def _add(role, cls_key):
            pool = self.resource_pool_manager.get_resource_pool(role)
            cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[role],
                config=worker_cfg_a if role in _model_a_roles else worker_cfg_b,
                role=str(role),
            )
            self.resource_pool_to_cls[pool][cls_key] = cls

        _add(Role.RolloutA, str(Role.RolloutA))  # pool_a: Model-A rollout
        _add(Role.ActorB,   str(Role.ActorB))    # pool_a: Model-B actor
        _add(Role.ActorA,   str(Role.ActorA))    # pool_b: Model-A actor
        _add(Role.RolloutB, str(Role.RolloutB))  # pool_b: Model-B rollout
        if self.use_reference_policy:
            _add(Role.RefB, str(Role.RefB))      # pool_a: Model-B ref
            _add(Role.RefA, str(Role.RefA))      # pool_b: Model-A ref

    def _spawn_worker_groups(self):
        wg_kwargs = {"device_name": self.device_name}
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = (
                self.config.trainer.ray_wait_register_center_timeout
            )
        all_wg = {}
        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            combined_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg = self.ray_worker_group_cls(
                resource_pool=resource_pool,
                ray_cls_with_init=combined_cls,
                **wg_kwargs,
            )
            spawned = wg.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawned)
        self.all_wg = all_wg

    def _init_models(self):
        # Worker groups
        self.actor_a_wg  = self.all_wg[str(Role.ActorA)]
        self.rollout_a_wg = self.all_wg[str(Role.RolloutA)]
        self.actor_b_wg  = self.all_wg[str(Role.ActorB)]
        self.rollout_b_wg = self.all_wg[str(Role.RolloutB)]

        # Init models
        self.actor_a_wg.init_model()
        self.actor_b_wg.init_model()
        self.rollout_a_wg.init_model()
        self.rollout_b_wg.init_model()

        if self.use_reference_policy:
            self.ref_a_wg = self.all_wg[str(Role.RefA)]
            self.ref_b_wg = self.all_wg[str(Role.RefB)]
            self.ref_a_wg.init_model()
            self.ref_b_wg.init_model()

        # Exchange weight shape info (needed for NCCL broadcast loop)
        weights_info_a = self.actor_a_wg.get_actor_weights_info()[0]
        weights_info_b = self.actor_b_wg.get_actor_weights_info()[0]
        self.rollout_a_wg.set_actor_weights_info(weights_info_a)
        self.rollout_b_wg.set_actor_weights_info(weights_info_b)

    def _create_weight_sync_group(self):
        from ray.util.collective import collective
        from verl.utils.device import get_nccl_backend

        # pool_a physical workers == actor_b_wg.workers == rollout_a_wg.workers (colocated)
        # pool_b physical workers == actor_a_wg.workers == rollout_b_wg.workers (colocated)
        pool_a_workers = self.actor_b_wg.workers
        pool_b_workers = self.actor_a_wg.workers
        all_workers = pool_a_workers + pool_b_workers
        n = len(all_workers)
        collective.create_collective_group(
            all_workers, n, list(range(n)),
            backend=get_nccl_backend(),
            group_name="dual_sync",
        )
        self._n_pool_a = len(pool_a_workers)  # broadcast src_rank for Model-A sync

    def _init_async_rollout_managers(self):
        from verl.experimental.one_step_off_policy.agent_loop import OneStepOffAgentLoopManager

        pool_a = self.resource_pool_manager.get_resource_pool(Role.RolloutA)
        pool_b = self.resource_pool_manager.get_resource_pool(Role.RolloutB)

        self.async_rollout_manager_a = OneStepOffAgentLoopManager(
            config=self._build_scoped_config("a"),
            worker_group=self.rollout_a_wg,
            rollout_resource_pool=pool_a,
            reward_loop_worker_handles=None,
        )
        self.async_rollout_manager_b = OneStepOffAgentLoopManager(
            config=self._build_scoped_config("b"),
            worker_group=self.rollout_b_wg,
            rollout_resource_pool=pool_b,
            reward_loop_worker_handles=None,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_scoped_config(self, which: str):
        """Return a copy of the full config with actor_rollout_ref replaced by the merged worker config for model a/b."""
        cfg = deepcopy(self.config)
        worker_cfg = self.worker_cfg_a if which == "a" else self.worker_cfg_b
        with open_dict(cfg):
            cfg.actor_rollout_ref = deepcopy(worker_cfg)
        return cfg

    def _create_continuous_iterator(self):
        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                yield epoch, batch_dict

    def sync_model_a_weights(self):
        """Model-A: pool_b (ActorA) → pool_a (RolloutA). src_rank = n_pool_a."""
        self.actor_a_wg.sync_weights_with_group("dual_sync", self._n_pool_a)       # non-blocking
        ray.get(self.rollout_a_wg.sync_weights_with_group("dual_sync", self._n_pool_a))  # blocking

    def sync_model_b_weights(self):
        """Model-B: pool_a (ActorB) → pool_b (RolloutB). src_rank = 0."""
        self.actor_b_wg.sync_weights_with_group("dual_sync", 0)        # non-blocking
        ray.get(self.rollout_b_wg.sync_weights_with_group("dual_sync", 0))  # blocking

    def _get_gen_batch(self, batch: DataProto) -> DataProto:
        """Extract prompt fields for generation (mirrors RayPPOTrainer._get_gen_batch)."""
        reward_keys = set({"data_source", "reward_model", "extra_info", "uid"}) & batch.non_tensor_batch.keys()
        non_tensor_to_pop = list(set(batch.non_tensor_batch.keys()) - reward_keys)
        gen_batch = batch.pop(batch_keys=[], non_tensor_batch_keys=non_tensor_to_pop)
        gen_batch.non_tensor_batch.update(batch.non_tensor_batch)
        return gen_batch

    def _balance_batch(self, batch: DataProto, metrics: dict):
        """Seqlen-balance across DP ranks (mirrors one_step_off RayTrainer logic)."""
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen = attention_mask.data.sum(-1)
        dp_size = self.actor_a_wg.world_size  # both actors have same DP
        partitions = get_seqlen_balanced_partitions(
            seqlens=global_seqlen.tolist(), n_partitions=dp_size, equal_size=True
        )
        indices = torch.cat([torch.tensor(p) for p in partitions])
        batch.reorder(indices)

    # ------------------------------------------------------------------
    # Async generation (both models, same data batch)
    # ------------------------------------------------------------------

    async def _async_gen_next_batch(self, continuous_iterator):
        try:
            epoch, batch_dict = next(continuous_iterator)
        except StopIteration:
            return None

        batch_a = DataProto.from_single_dict(batch_dict)
        batch_b = DataProto.from_single_dict(batch_dict)  # same data, two models

        batch_a.non_tensor_batch["uid"] = np.array(
            [str(uuid.uuid4()) for _ in range(len(batch_a.batch))], dtype=object
        )
        batch_b.non_tensor_batch["uid"] = np.array(
            [str(uuid.uuid4()) for _ in range(len(batch_b.batch))], dtype=object
        )
        batch_a.meta_info["temperature"] = self.worker_cfg_a.rollout.temperature
        batch_b.meta_info["temperature"] = self.worker_cfg_b.rollout.temperature

        gen_a = self._get_gen_batch(batch_a)
        gen_b = self._get_gen_batch(batch_b)
        gen_a.meta_info["global_steps"] = self.global_steps
        gen_b.meta_info["global_steps"] = self.global_steps
        gen_a = gen_a.repeat(repeat_times=self.worker_cfg_a.rollout.n, interleave=True)
        gen_b = gen_b.repeat(repeat_times=self.worker_cfg_b.rollout.n, interleave=True)

        timing_raw = {}
        metrics = {}

        with marked_timer("generate_async_ab", timing_raw, color="purple"):
            gen_out_a, gen_out_b = await asyncio.gather(
                self.async_rollout_manager_a.generate_sequences_async(gen_a),
                self.async_rollout_manager_b.generate_sequences_async(gen_b),
            )

        timing_raw.update(gen_out_a.meta_info.get("timing", {}))
        gen_out_a.meta_info.pop("timing", None)
        gen_out_b.meta_info.pop("timing", None)

        batch_a = batch_a.repeat(repeat_times=self.worker_cfg_a.rollout.n, interleave=True)
        batch_a = batch_a.union(gen_out_a)
        batch_b = batch_b.repeat(repeat_times=self.worker_cfg_b.rollout.n, interleave=True)
        batch_b = batch_b.union(gen_out_b)

        if "response_mask" not in batch_a.batch:
            batch_a.batch["response_mask"] = compute_response_mask(batch_a)
        if "response_mask" not in batch_b.batch:
            batch_b.batch["response_mask"] = compute_response_mask(batch_b)

        if self.config.trainer.balance_batch:
            self._balance_batch(batch_a, metrics)
            self._balance_batch(batch_b, metrics)

        batch_a.meta_info["global_token_num"] = torch.sum(batch_a.batch["attention_mask"], dim=-1).tolist()
        batch_b.meta_info["global_token_num"] = torch.sum(batch_b.batch["attention_mask"], dim=-1).tolist()

        return epoch, batch_a, batch_b, metrics, timing_raw

    # ------------------------------------------------------------------
    # Training for one model (called from thread via asyncio.to_thread)
    # ------------------------------------------------------------------

    def _train_one_model(self, which: str, batch: DataProto) -> tuple[dict, dict]:
        """Compute reward + ref + adv + update_actor for model `which` ('a' or 'b').
        Runs in a thread (asyncio.to_thread) so it doesn't block the event loop.
        Returns (metrics, timing_raw).
        """
        actor_wg = self.actor_a_wg if which == "a" else self.actor_b_wg
        cfg = self.worker_cfg_a if which == "a" else self.worker_cfg_b  # merged config with all defaults
        alg = self.config.algorithm
        ref_in_actor = self.ref_in_actor_a if which == "a" else self.ref_in_actor_b
        if self.use_reference_policy and not ref_in_actor:
            ref_wg = self.ref_a_wg if which == "a" else self.ref_b_wg
        else:
            ref_wg = None

        metrics = {}
        timing_raw = {}

        # ---- reward ----
        with marked_timer(f"reward_{which}", timing_raw, color="yellow"):
            reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)

        # ---- reference log prob ----
        if self.use_reference_policy:
            with marked_timer(f"ref_{which}", timing_raw, color="olive"):
                if ref_wg is not None:
                    ref_log_prob = ref_wg.compute_ref_log_prob(batch)
                else:
                    ref_log_prob = actor_wg.compute_ref_log_prob(batch)
                batch = batch.union(ref_log_prob)

        # ---- advantage ----
        with marked_timer(f"adv_{which}", timing_raw, color="brown"):
            batch.batch["token_level_scores"] = reward_tensor
            if reward_extra_infos_dict:
                batch.non_tensor_batch.update(
                    {k: np.array(v) for k, v in reward_extra_infos_dict.items()}
                )

            if alg.use_kl_in_reward:
                batch, kl_metrics = apply_kl_penalty(
                    batch,
                    kl_ctrl=self.kl_ctrl_in_reward,
                    kl_penalty=alg.kl_penalty,
                )
                metrics.update(kl_metrics)
            else:
                batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

            norm_adv = alg.get("norm_adv_by_std_in_grpo", True)
            batch = compute_advantage(
                batch,
                adv_estimator=alg.adv_estimator,
                gamma=alg.gamma,
                lam=alg.lam,
                num_repeat=cfg.rollout.n,
                norm_adv_by_std_in_grpo=norm_adv,
                config=alg,
            )

        # ---- old_log_probs (bypass mode: reuse rollout log probs, skip actor fwd pass) ----
        rollout_corr_config = alg.get("rollout_correction", None)
        bypass_mode = rollout_corr_config and rollout_corr_config.get("bypass_mode", False)
        if bypass_mode:
            from verl.trainer.ppo.rollout_corr_helper import apply_bypass_mode
            apply_bypass_mode(
                batch,
                rollout_corr_config=rollout_corr_config,
                policy_loss_config=cfg.actor.policy_loss,
            )
        else:
            with marked_timer(f"old_log_prob_{which}", timing_raw, color="blue"):
                old_log_prob = actor_wg.compute_log_prob(batch)
                batch = batch.union(old_log_prob)

        # ---- update actor ----
        with marked_timer(f"update_actor_{which}", timing_raw, color="red"):
            rollout_cfg = cfg.rollout
            batch.meta_info["multi_turn"] = rollout_cfg.multi_turn.enable
            batch.meta_info["temperature"] = rollout_cfg.temperature
            actor_output = actor_wg.update_actor(batch)

        actor_metrics = reduce_metrics(actor_output.meta_info["metrics"])
        metrics.update({f"{k}_{which}": v for k, v in actor_metrics.items()})

        # data metrics
        metrics.update({
            f"critic_{which}/score/mean": float(reward_tensor.mean()),
            f"critic_{which}/score/max": float(reward_tensor.max()),
            f"critic_{which}/score/min": float(reward_tensor.min()),
        })

        return metrics, timing_raw

    # ------------------------------------------------------------------
    # Checkpoint helpers (minimal: save actor weights only)
    # ------------------------------------------------------------------

    def _save_checkpoint(self):
        from verl.utils.fs import local_mkdir_safe
        folder = os.path.join(
            self.config.trainer.default_local_dir, f"global_step_{self.global_steps}"
        )
        local_mkdir_safe(folder)
        self.actor_a_wg.save_checkpoint(
            os.path.join(folder, "actor_a"), None, self.global_steps
        )
        self.actor_b_wg.save_checkpoint(
            os.path.join(folder, "actor_b"), None, self.global_steps
        )

    def _load_checkpoint(self):
        # Re-use base trainer logic if available; otherwise just log
        pass

    # ------------------------------------------------------------------
    # Timing print
    # ------------------------------------------------------------------

    def _print_unified_timing(self, step: int, timing_raw: dict):
        def _s(v):
            return f"{v:.3f}s" if v is not None else "n/a"

        print(
            f"[TIMING] step={step}"
            f"  gen_async={_s(timing_raw.get('generate_async_ab'))}"
            f"  gen_wait={_s(timing_raw.get('gen'))}"
            f"  sync_weights={_s(timing_raw.get('sync_weights'))}"
            f"  train_concurrent={_s(timing_raw.get('train_concurrent'))}"
            f"  (update_actor_a={_s(timing_raw.get('update_actor_a'))}"
            f"  update_actor_b={_s(timing_raw.get('update_actor_b'))})"
            f"  step_total={_s(timing_raw.get('step'))}",
            flush=True,
        )

    # ------------------------------------------------------------------
    # Main async training loop
    # ------------------------------------------------------------------

    async def fit(self):
        from verl.utils.tracking import Tracking

        _ts = lambda: time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        fit_wall_start = time.perf_counter()
        print(f"[FIT_DUAL_ASYNC] wall_start={_ts()}", flush=True)

        self.global_steps = 0
        self._load_checkpoint()

        # Initial weight sync (before any generation)
        with marked_timer("init_sync", {}):
            self.sync_model_a_weights()
            self.sync_model_b_weights()
        await self.async_rollout_manager_a.clear_kv_cache()
        await self.async_rollout_manager_b.clear_kv_cache()

        progress_bar = tqdm(
            total=self.total_training_steps,
            initial=self.global_steps,
            desc="Dual Async Training",
        )

        self.global_steps += 1
        last_val_metrics = None
        self.max_steps_duration = 0

        continuous_iterator = self._create_continuous_iterator()

        # Kick off first async generation
        gen_future = asyncio.create_task(self._async_gen_next_batch(continuous_iterator))

        while gen_future is not None:
            metrics = {}
            timing_raw = {}
            is_last_step = self.global_steps >= self.total_training_steps
            print(f"[{_ts()}] [FIT_DUAL_ASYNC] step={self.global_steps} begin", flush=True)

            with marked_timer("step", timing_raw):
                # ---- wait for current batch ----
                with marked_timer("gen", timing_raw, color="red"):
                    result = await gen_future
                    if result is None:
                        break
                    epoch, batch_a, batch_b, gen_metrics, gen_timing = result
                    timing_raw.update(gen_timing)
                    metrics.update(gen_metrics)

                # ---- sync weights before next gen ----
                with marked_timer("sync_weights", timing_raw, color="purple"):
                    self.sync_model_a_weights()
                    self.sync_model_b_weights()
                    await self.async_rollout_manager_a.clear_kv_cache()
                    await self.async_rollout_manager_b.clear_kv_cache()

                # ---- launch next gen (overlaps with training below) ----
                if not is_last_step:
                    gen_future = asyncio.create_task(
                        self._async_gen_next_batch(continuous_iterator)
                    )
                    await asyncio.sleep(0)  # yield so gen tasks can start
                else:
                    gen_future = None

                # ---- concurrent training: model_a on pool_b || model_b on pool_a ----
                with marked_timer("train_concurrent", timing_raw, color="red"):
                    (metrics_a, timing_a), (metrics_b, timing_b) = await asyncio.gather(
                        asyncio.to_thread(self._train_one_model, "a", batch_a),
                        asyncio.to_thread(self._train_one_model, "b", batch_b),
                    )
                    metrics.update(metrics_a)
                    metrics.update(metrics_b)
                    timing_raw.update(timing_a)
                    timing_raw.update(timing_b)

            # ---- timing summary ----
            steps_duration = timing_raw["step"]
            self.max_steps_duration = max(self.max_steps_duration, steps_duration)
            self._print_unified_timing(self.global_steps, timing_raw)

            # ---- save checkpoint ----
            esi_close = should_save_ckpt_esi(
                max_steps_duration=self.max_steps_duration,
                redundant_time=self.config.trainer.esi_redundant_time,
            )
            if self.config.trainer.save_freq > 0 and (
                is_last_step
                or self.global_steps % self.config.trainer.save_freq == 0
                or esi_close
            ):
                self._save_checkpoint()

            # ---- collect metrics ----
            metrics.update({
                "training/global_step": self.global_steps,
                "training/epoch": epoch,
            })
            metrics.update(
                {f"data_a/{k}": v for k, v in compute_data_metrics(batch=batch_a, use_critic=False).items()}
            )
            metrics.update(
                {f"data_b/{k}": v for k, v in compute_data_metrics(batch=batch_b, use_critic=False).items()}
            )
            metrics.update(compute_timing_metrics(batch=batch_a, timing_raw=timing_raw))
            n_gpus = self.resource_pool_manager.get_n_gpus()
            metrics.update(compute_throughout_metrics(batch=batch_a, timing_raw=timing_raw, n_gpus=n_gpus))

            logger.log(data=metrics, step=self.global_steps)
            progress_bar.update(1)
            self.global_steps += 1

            if is_last_step:
                pprint(f"Final validation metrics: {last_val_metrics}")
                progress_bar.close()
                print(
                    f"[FIT_DUAL_ASYNC] wall_total={time.perf_counter() - fit_wall_start:.3f}s (completed)",
                    flush=True,
                )
                return

        progress_bar.close()
        print(
            f"[FIT_DUAL_ASYNC] wall_total={time.perf_counter() - fit_wall_start:.3f}s (epoch_end)",
            flush=True,
        )

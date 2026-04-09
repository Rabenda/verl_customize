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
DualFullyAsyncWorker: A single Ray actor that occupies one GPU group and simultaneously:
  - Rolls out model `rollout_which` ("a" or "b") via an async sglang server
  - Trains model `train_which`  ("b" or "a") via FSDP

Cross-connected with the peer DualFullyAsyncWorker:
  - r_mq_client  : produce rollout samples → peer's t_mq_client
  - t_mq_client  : consume training  samples ← peer's r_mq_client
  - t_param_sync : after N train steps, push train_which weights → peer's sglang server
  - (The peer's t_param_sync pushes rollout_which weights → THIS worker's sglang server)

Attribute prefix convention:
  r_* = rollout side  (model rollout_which, sglang)
  t_* = training side (model train_which,   FSDP)
"""

import asyncio
import multiprocessing
import os
import time
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import datetime
from pprint import pformat, pprint

import numpy as np
import ray
import torch
from omegaconf import OmegaConf, open_dict
from ray import ObjectRef

from verl import DataProto
from verl.experimental.fully_async_policy.detach_utils import (
    MetricsAggregator,
    RolloutSample,
    ValidateMetrics,
    assemble_batch_from_rollout_samples,
    prepare_single_generation_data,
)
from verl.experimental.fully_async_policy.message_queue import MessageQueueClient
from verl.experimental.fully_async_policy.ray_trainer import FullyAsyncRayPPOTrainer
from verl.single_controller.ray import RayClassWithInitArgs, RayWorkerGroup
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import AdvantageEstimator, agg_loss
from verl.trainer.ppo.ray_trainer import ResourcePoolManager
from verl.trainer.ppo.reward import compute_reward, compute_reward_async, load_reward_manager
from verl.trainer.ppo.utils import Role, WorkerType
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path, should_save_ckpt_esi
from verl.utils.debug import marked_timer
from verl.utils.metric import reduce_metrics
from verl.utils.tracking import ValidationGenerationsLogger


# ---------------------------------------------------------------------------
# Small helpers (replicate the ones in ray_trainer.py to avoid coupling)
# ---------------------------------------------------------------------------

def _compute_response_mask(data: DataProto):
    responses = data.batch["responses"]
    response_length = responses.size(1)
    attention_mask = data.batch["attention_mask"]
    return attention_mask[:, -response_length:]


def _apply_kl_penalty(data: DataProto, kl_ctrl, kl_penalty="kl"):
    from verl.utils.torch_functional import masked_mean
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
    return data, {"actor/reward_kl_penalty": current_kl, "actor/reward_kl_penalty_coeff": beta}


def _compute_advantage(data: DataProto, adv_estimator, gamma=1.0, lam=1.0,
                        num_repeat=1, norm_adv_by_std_in_grpo=True, config=None):
    if "response_mask" not in data.batch.keys():
        data.batch["response_mask"] = _compute_response_mask(data)
    if adv_estimator == AdvantageEstimator.GAE:
        adv, ret = core_algos.compute_gae_advantage_return(
            token_level_rewards=data.batch["token_level_rewards"],
            values=data.batch["values"],
            response_mask=data.batch["response_mask"],
            gamma=gamma, lam=lam,
        )
    elif adv_estimator == AdvantageEstimator.GRPO:
        adv, ret = core_algos.compute_grpo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=data.batch["response_mask"],
            index=data.non_tensor_batch["uid"],
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        )
    else:
        fn = core_algos.get_adv_estimator_fn(adv_estimator)
        kwargs = dict(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=data.batch["response_mask"],
            config=config,
        )
        if "uid" in data.non_tensor_batch:
            kwargs["index"] = data.non_tensor_batch["uid"]
        if "reward_baselines" in data.batch:
            kwargs["reward_baselines"] = data.batch["reward_baselines"]
        adv, ret = fn(**kwargs)
    data.batch["advantages"] = adv
    data.batch["returns"] = ret
    return data


# ---------------------------------------------------------------------------
# DualFullyAsyncWorker
# ---------------------------------------------------------------------------

@ray.remote(num_cpus=10, max_concurrency=100)
class DualFullyAsyncWorker:
    """
    Occupies one GPU group. Runs rollout for model X and training for model Y
    concurrently (via asyncio.gather). Communicates with its peer worker via
    two MessageQueues and two ParameterSynchronizers.
    """

    def __init__(
        self,
        config,
        rollout_which: str,          # "a" or "b"
        train_which: str,            # "b" or "a"
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: type[RayWorkerGroup] = RayWorkerGroup,
        processor=None,
        device_name: str = None,
    ):
        assert rollout_which in ("a", "b") and train_which in ("a", "b")
        assert rollout_which != train_which

        self.config = config
        self.tokenizer = tokenizer
        self.processor = processor
        self.rollout_which = rollout_which
        self.train_which = train_which
        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.ray_worker_group_cls = ray_worker_group_cls
        self.device_name = device_name or config.trainer.device

        # Per-model configs (actor_rollout_ref_a / actor_rollout_ref_b)
        self.r_model_config = getattr(config, f"actor_rollout_ref_{rollout_which}")
        self.t_model_config = getattr(config, f"actor_rollout_ref_{train_which}")

        # Roles
        self.r_role = Role.RolloutA if rollout_which == "a" else Role.RolloutB
        self.t_role = Role.ActorA if train_which == "a" else Role.ActorB
        ref_role_map = {"a": Role.RefA, "b": Role.RefB}
        self.t_ref_role = ref_role_map[train_which]

        # Reward functions (shared)
        self.reward_fn = load_reward_manager(
            config, tokenizer, num_examine=0, **config.reward_model.get("reward_kwargs", {})
        )
        self.val_reward_fn = load_reward_manager(
            config, tokenizer, num_examine=1, **config.reward_model.get("reward_kwargs", {})
        )

        lora_rank = self.t_model_config.model.get("lora", {}).get("rank", 0)
        self.t_ref_in_actor = lora_rank > 0
        self.t_use_reference_policy = config.algorithm.use_kl_in_reward or config.algorithm.get("kl_penalty", None)
        self.t_use_critic = False  # not supported in dual async mode for now
        self.t_use_rm = False

        if config.algorithm.use_kl_in_reward:
            self.t_kl_ctrl_in_reward = core_algos.get_kl_controller(config.algorithm.kl_ctrl)

        # ---- Rollout-side state ----
        self.r_wg = None
        self.r_async_mgr = None
        self.r_mq_client: MessageQueueClient = None
        self.r_current_param_version = 0
        self.r_paused = False
        self.r_running = False
        self.r_monitor_loop_trigger = True
        self.r_global_steps = 1
        self.r_total_generated_samples = 0
        self.r_dropped_stale_samples = 0
        self.r_processed_sample_count = 0
        self.r_staleness_samples = 0
        self.r_staleness_threshold: float = config.async_training.get("staleness_threshold", 1)
        self.r_max_required_samples = None
        self.r_max_queue_size = None
        self.r_max_concurrent_samples = None
        self.r_idle_start_time = None
        self.r_version_start_time = None
        # asyncio objects (initialized in _init_async_objects)
        self.r_condition = None
        self.r_lock = None
        self.r_pending_queue = asyncio.Queue(maxsize=128)
        self.r_active_tasks: set = set()
        self.r_cancel_queue = asyncio.Queue()
        # validate
        cpu_cores = multiprocessing.cpu_count()
        self.r_validate_executor = ThreadPoolExecutor(max_workers=cpu_cores)
        self.r_parallel_validate_and_rollout = config.async_training.get("parallel_validate_and_rollout", False)
        self.r_validate_task = None
        self.r_validation_generations_logger = ValidationGenerationsLogger(
            project_name=config.trainer.project_name,
            experiment_name=config.trainer.experiment_name,
        )

        # Build rollout dataloader
        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler
        from verl.utils.dataset.rl_dataset import collate_fn

        train_dataset = create_rl_dataset(config.data.train_files, config.data, tokenizer, processor)
        val_dataset = create_rl_dataset(config.data.val_files, config.data, tokenizer, processor)
        train_sampler = create_rl_sampler(config.data, train_dataset)

        self._r_create_dataloader(train_dataset, val_dataset, collate_fn, train_sampler)

        self.r_total_rollout_steps = len(self.r_train_dataloader) * config.rollout.get("total_epochs", config.trainer.total_epochs)
        if config.rollout.total_rollout_steps is not None:
            self.r_total_rollout_steps = min(config.rollout.total_rollout_steps, self.r_total_rollout_steps)
        print(f"[DualWorker-{rollout_which}/{train_which}] total_rollout_steps={self.r_total_rollout_steps}")

        # ---- Training-side state ----
        self.t_wg = None
        self.t_ref_wg = None
        self.t_mq_client: MessageQueueClient = None
        self.t_param_sync = None
        self.t_global_steps = 1
        self.t_local_trigger_step = 1
        self.t_current_param_version = 0
        self.t_trigger_parameter_sync_step = config.async_training.trigger_parameter_sync_step
        self.t_require_batches = config.async_training.require_batches
        self.t_required_samples = (
            self.t_model_config.actor.ppo_mini_batch_size * self.t_require_batches
        )
        self.t_processed_samples = 0
        self.t_stale_samples_processed = 0
        self.t_stale_trajectory_processed = 0
        self.t_max_steps_duration = 0
        self.t_last_ckpt_version = 0
        self.t_logger = None
        total_gpus = (
            config.trainer.nnodes * config.trainer.n_gpus_per_node
            + config.rollout.nnodes * config.rollout.n_gpus_per_node
        )
        self.t_metrics_aggregator = MetricsAggregator(total_gpus=total_gpus)

        print(f"[DualWorker-{rollout_which}/{train_which}] __init__ done")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _r_create_dataloader(self, train_dataset, val_dataset, collate_fn, train_sampler):
        from torchdata.stateful_dataloader import StatefulDataLoader
        num_workers = self.config.data["dataloader_num_workers"]
        self.r_train_dataloader = StatefulDataLoader(
            dataset=train_dataset,
            batch_size=self.r_model_config.rollout.get("gen_batch_size", 1),
            num_workers=num_workers,
            drop_last=True,
            collate_fn=collate_fn,
            sampler=train_sampler,
        )
        val_batch_size = self.config.data.val_batch_size or len(val_dataset)
        self.r_val_dataloader = StatefulDataLoader(
            dataset=val_dataset,
            batch_size=val_batch_size,
            num_workers=num_workers,
            shuffle=self.config.data.get("validation_shuffle", True),
            drop_last=False,
            collate_fn=collate_fn,
        )

    def _build_scoped_config(self, which: str):
        """Return a full config with config.actor_rollout_ref swapped for model `which`."""
        cfg = deepcopy(self.config)
        sub = deepcopy(getattr(self.config, f"actor_rollout_ref_{which}"))
        with open_dict(cfg):
            cfg.actor_rollout_ref = sub
        return cfg

    def _init_async_objects(self):
        self.r_condition = asyncio.Condition()
        self.r_lock = self.r_condition._lock

    # ------------------------------------------------------------------
    # Worker group initialisation
    # ------------------------------------------------------------------

    async def init_workers(self):
        self._init_async_objects()
        self._init_resource_pools()
        self._init_worker_groups()
        self._init_models()
        await self._init_async_rollout_manager()

    def _init_resource_pools(self):
        self.resource_pool_manager.create_resource_pool()
        self.resource_pool_to_cls = {
            pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()
        }

    def _init_worker_groups(self):
        """Colocate rollout-X and actor-Y on the same GPU pool."""
        pool = self.resource_pool_manager.get_resource_pool(self.r_role)
        assert pool == self.resource_pool_manager.get_resource_pool(self.t_role), \
            "Rollout and train roles must map to the same resource pool for DualFullyAsyncWorker"

        r_cls = RayClassWithInitArgs(
            cls=self.role_worker_mapping[self.r_role],
            config=self._build_scoped_config(self.rollout_which),
            role=str(Role.Rollout),
        )
        t_cls = RayClassWithInitArgs(
            cls=self.role_worker_mapping[self.t_role],
            config=self._build_scoped_config(self.train_which),
            role=str(Role.Actor),
        )
        class_dict = {str(self.r_role): r_cls, str(self.t_role): t_cls}

        if self.t_use_reference_policy and not self.t_ref_in_actor:
            ref_pool = self.resource_pool_manager.get_resource_pool(self.t_ref_role)
            assert ref_pool == pool, "Ref policy must be on the same pool"
            ref_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[self.t_ref_role],
                config=self._build_scoped_config(self.train_which),
                role=str(Role.RefPolicy),
            )
            class_dict[str(self.t_ref_role)] = ref_cls

        worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
        wg_kwargs = {"device_name": self.device_name}
        wg_dict = self.ray_worker_group_cls(
            resource_pool=pool,
            ray_cls_with_init=worker_dict_cls,
            **wg_kwargs,
        )
        spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())

        self.r_wg = spawn_wg[str(self.r_role)]
        self.t_wg = spawn_wg[str(self.t_role)]
        if self.t_use_reference_policy and not self.t_ref_in_actor:
            self.t_ref_wg = spawn_wg[str(self.t_ref_role)]

    def _init_models(self):
        self.r_wg.init_model()
        self.t_wg.init_model()
        if self.t_use_reference_policy and not self.t_ref_in_actor and self.t_ref_wg is not None:
            self.t_ref_wg.init_model()

    async def _init_async_rollout_manager(self):
        from verl.experimental.fully_async_policy.agent_loop import FullyAsyncAgentLoopManager
        self.r_async_mgr = await FullyAsyncAgentLoopManager.create(
            config=self._build_scoped_config(self.rollout_which),
            worker_group=self.r_wg,
        )

    # ------------------------------------------------------------------
    # Public setters (called by the main TaskRunner before fit())
    # ------------------------------------------------------------------

    async def set_rollout_mq_client(self, mq_client: MessageQueueClient):
        """Set the MessageQueue where rollout samples are produced."""
        async with self.r_lock:
            self.r_mq_client = mq_client

    def set_train_mq_client(self, mq_client: MessageQueueClient):
        """Set the MessageQueue from which training samples are consumed."""
        self.t_mq_client = mq_client

    def set_param_synchronizer(self, param_sync):
        """Set the ParameterSynchronizer that pushes train_which weights to peer's sglang."""
        self.t_param_sync = param_sync

    async def set_max_required_samples(self):
        async with self.r_lock:
            self.r_max_required_samples = int(
                self.t_required_samples
                * (self.r_staleness_threshold + 1)
                * self.config.async_training.trigger_parameter_sync_step
            )
            total_train_steps = int(
                self.r_total_rollout_steps
                / (self.t_required_samples * self.config.async_training.trigger_parameter_sync_step)
            )
            self.r_total_train_steps = total_train_steps
            self.r_max_concurrent_samples = len(self.r_async_mgr.server_handles) * 16
            self.r_max_concurrent_samples = min(self.r_max_concurrent_samples, self.r_max_required_samples)
            self.r_max_queue_size = self.r_max_required_samples
            print(
                f"[DualWorker-{self.rollout_which}/{self.train_which}][set_max_required_samples] "
                f"required={self.t_required_samples} max_required={self.r_max_required_samples} "
                f"max_queue={self.r_max_queue_size} total_train_steps={total_train_steps}"
            )

    def set_total_train_steps(self, total_train_steps: int):
        self.r_total_train_steps = total_train_steps

    def set_train_logger(self, logger):
        """Inject a Tracking logger for training metrics."""
        self.t_logger = logger

    # ------------------------------------------------------------------
    # Public getters (used by ParameterSynchronizer factory)
    # ------------------------------------------------------------------

    def get_rollout_wg(self):
        """Return the sglang rollout worker group (model rollout_which)."""
        return self.r_wg

    def get_train_wg(self):
        """Return the FSDP training worker group (model train_which)."""
        return self.t_wg

    def get_max_queue_size(self):
        return self.r_max_queue_size

    def get_total_train_steps(self):
        return self.r_total_train_steps

    # ------------------------------------------------------------------
    # ParameterSynchronizer interface (called by PEER's param_sync)
    # i.e. the peer trains model `rollout_which` and pushes weights here
    # ------------------------------------------------------------------

    async def pause_rollout(self):
        """Pause the sglang server (called by peer's ParameterSynchronizer before weight sync)."""
        print(f"[DualWorker-{self.rollout_which}/{self.train_which}][pause_rollout] "
              f"partial_rollout={self.config.async_training.partial_rollout}")
        async with self.r_lock:
            self.r_paused = True
            if self.config.async_training.partial_rollout:
                await self.r_async_mgr.cancel()
            if self.r_active_tasks:
                await asyncio.gather(*self.r_active_tasks, return_exceptions=True)
                self.r_active_tasks.clear()
            await self.r_async_mgr.clear_kv_cache()
            self.r_monitor_loop_trigger = False

    async def resume_rollout(self, dependency_ref: ObjectRef = None):
        """Resume the sglang server after weight sync."""
        if dependency_ref is not None:
            ray.get(dependency_ref)
        print(f"[DualWorker-{self.rollout_which}/{self.train_which}][resume_rollout]")
        async with self.r_lock:
            if self.config.async_training.partial_rollout:
                await self.r_async_mgr.resume()
            self.r_paused = False
            self.r_monitor_loop_trigger = True
            self.r_condition.notify_all()

    async def update_rollout_param_version(
        self, version: int, validate: bool = False, global_steps: int = 0,
        use_trainer_do_validate: bool = False
    ):
        """Update the rollout's current param version (weights were just synced)."""
        async with self.r_lock:
            old = self.r_current_param_version
            self.r_current_param_version = version
            self.r_staleness_samples = (
                len(self.r_active_tasks)
                + self.r_cancel_queue.qsize()
                + await self.r_mq_client.get_queue_size()
            )
            timing_raw = {}
            idle_ratio = None
            if self.r_idle_start_time is not None and self.r_version_start_time is not None:
                active_t = self.r_idle_start_time - self.r_version_start_time
                version_t = time.time() - self.r_version_start_time
                idle_ratio = 1 - active_t / version_t if version_t > 0 else None
                timing_raw["rollouter/active_time"] = active_t
                timing_raw["rollouter/version_time"] = version_t
                timing_raw["rollouter/idle_ratio"] = idle_ratio
                self.r_idle_start_time = None
            print(
                f"[DualWorker-{self.rollout_which}][update_rollout_param_version] "
                f"{old} → {version}, staleness_samples={self.r_staleness_samples}, idle_ratio={idle_ratio}"
            )
            need_validate = (
                self.val_reward_fn is not None
                and self.config.rollout.test_freq > 0
                and version % self.config.rollout.test_freq == 0
                and version > 0
            ) or (validate and self.val_reward_fn is not None)

            if not need_validate:
                data = ValidateMetrics(
                    timing_raw=timing_raw, metrics=None,
                    global_steps=global_steps, param_version=version,
                )
            elif need_validate and not self.r_parallel_validate_and_rollout:
                data = self._r_validate_wrapper(timing_raw, version, global_steps, use_trainer_do_validate)

            if not need_validate or not self.r_parallel_validate_and_rollout:
                await self.r_mq_client.put_validate(ray.cloudpickle.dumps(data))

            self.r_version_start_time = time.time()

        if need_validate and self.r_parallel_validate_and_rollout:
            if self.r_validate_task and not self.r_validate_task.done():
                self.r_validate_task.get()
            self.r_validate_task = asyncio.create_task(
                self._r_do_validate_async(timing_raw, version, global_steps, use_trainer_do_validate)
            )

    # For backward compat with ParameterSynchronizer that calls .pause/.resume/.update_param_version
    async def pause(self):
        await self.pause_rollout()

    async def resume(self, dependency_ref: ObjectRef = None):
        await self.resume_rollout(dependency_ref)

    async def update_param_version(self, version, validate=False, global_steps=0, use_trainer_do_validate=False):
        await self.update_rollout_param_version(version, validate, global_steps, use_trainer_do_validate)

    # ------------------------------------------------------------------
    # Checkpoint (rollout side dataloader, called by ParameterSynchronizer)
    # ------------------------------------------------------------------

    async def save_checkpoint(self, local_global_step_folder: str):
        from verl.utils.fs import local_mkdir_safe
        local_mkdir_safe(local_global_step_folder)
        path = os.path.join(local_global_step_folder, f"data_{self.rollout_which}.pt")
        async with self.r_lock:
            state_dict = self.r_train_dataloader.state_dict()
        torch.save(state_dict, path)
        print(f"[DualWorker-{self.rollout_which}] Saved dataloader to {path}")

    def load_rollout_checkpoint(self):
        if self.config.trainer.resume_mode == "disable":
            return 0
        checkpoint_folder = self.config.trainer.default_local_dir
        if not os.path.isabs(checkpoint_folder):
            checkpoint_folder = os.path.join(os.getcwd(), checkpoint_folder)
        global_step_folder = find_latest_ckpt_path(checkpoint_folder)
        if self.config.trainer.resume_mode == "auto" and global_step_folder is None:
            print(f"[DualWorker-{self.rollout_which}] Training from scratch")
            return 0
        if self.config.trainer.resume_mode == "resume_path":
            global_step_folder = self.config.trainer.resume_from_path
            if not os.path.isabs(global_step_folder):
                global_step_folder = os.path.join(os.getcwd(), global_step_folder)
        trainer_global_steps = int(global_step_folder.split("global_step_")[-1])
        self.r_global_steps = (
            trainer_global_steps * self.t_required_samples
            * self.config.async_training.trigger_parameter_sync_step + 1
        )
        path = os.path.join(global_step_folder, f"data_{self.rollout_which}.pt")
        if os.path.exists(path):
            self.r_train_dataloader.load_state_dict(torch.load(path, weights_only=False))
        return 0

    def load_train_checkpoint(self):
        if self.config.trainer.resume_mode == "disable":
            self.t_wg.load_checkpoint(None)
            return 0
        checkpoint_folder = self.config.trainer.default_local_dir
        if not os.path.isabs(checkpoint_folder):
            checkpoint_folder = os.path.join(os.getcwd(), checkpoint_folder)
        global_step_folder = find_latest_ckpt_path(checkpoint_folder)
        if self.config.trainer.resume_mode == "auto" and global_step_folder is None:
            self.t_wg.load_checkpoint(None)
            return 0
        if self.config.trainer.resume_mode == "resume_path":
            global_step_folder = self.config.trainer.resume_from_path
            if not os.path.isabs(global_step_folder):
                global_step_folder = os.path.join(os.getcwd(), global_step_folder)
        self.t_current_param_version = int(global_step_folder.split("global_step_")[-1])
        self.t_global_steps = (
            self.t_current_param_version * self.t_trigger_parameter_sync_step + 1
        )
        self.t_last_ckpt_version = self.t_current_param_version
        actor_path = os.path.join(global_step_folder, f"actor_{self.train_which}")
        self.t_wg.load_checkpoint(actor_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load)
        return self.t_current_param_version

    # ------------------------------------------------------------------
    # Validation helpers (rollout side)
    # ------------------------------------------------------------------

    def _r_validate(self, use_trainer_do_validate=False):
        from verl.trainer.ppo.metric_utils import compute_data_metrics
        metrics = {}
        for batch_dict in self.r_val_dataloader:
            batch = DataProto.from_single_dict(batch_dict)
            batch = batch.repeat(repeat_times=self.r_model_config.rollout.n, interleave=True)
            gen_out = self.r_wg.generate_sequences(batch)
            batch = batch.union(gen_out)
            reward_tensor, _ = compute_reward(batch, self.val_reward_fn)
            batch.batch["token_level_scores"] = reward_tensor
            metrics.update(compute_data_metrics(batch=batch, use_critic=False))
        return metrics

    def _r_validate_wrapper(self, timing_raw, version, global_steps, use_trainer_do_validate):
        val_metrics = None
        with marked_timer("rollouter/validate_time", timing_raw):
            val_metrics = self._r_validate(use_trainer_do_validate)
        return ValidateMetrics(
            timing_raw=timing_raw, metrics=val_metrics,
            global_steps=global_steps, param_version=version,
        )

    async def _r_do_validate_async(self, timing_raw, version, global_steps, use_trainer_do_validate):
        loop = asyncio.get_running_loop()
        import functools
        data = await loop.run_in_executor(
            self.r_validate_executor,
            functools.partial(
                self._r_validate_wrapper,
                timing_raw=timing_raw, version=version,
                global_steps=global_steps, use_trainer_do_validate=use_trainer_do_validate,
            ),
        )
        await self.r_mq_client.put_validate(ray.cloudpickle.dumps(data))

    # ------------------------------------------------------------------
    # Rollout loop (adapted from FullyAsyncRollouter)
    # ------------------------------------------------------------------

    def _r_create_continuous_iterator(self):
        total_epochs = self.config.rollout.get("total_epochs", self.config.trainer.total_epochs)
        for epoch in range(total_epochs):
            for batch_dict in self.r_train_dataloader:
                yield epoch, batch_dict

    async def _r_feed_samples(self):
        for epoch, batch_dict in self._r_create_continuous_iterator():
            full_batch = prepare_single_generation_data(batch_dict, self.config)
            sample_id = f"sample_{self.rollout_which}_{epoch}_{self.r_global_steps}"
            rollout_sample = RolloutSample(
                full_batch=full_batch,
                agent_loop_output_list=[None] * self.r_model_config.rollout.n,
                sample_id=sample_id,
                epoch=epoch,
                param_version=0,
                param_version_start=[],
                param_version_end=[],
                processing_times=[],
                tool_calls=[],
                rollout_status={},
            )
            await self.r_pending_queue.put(rollout_sample)
            if self.r_global_steps >= self.r_total_rollout_steps:
                print(f"[DualWorker-{self.rollout_which}][Feed] reached total_rollout_steps={self.r_total_rollout_steps}")
                break
            self.r_global_steps += 1
        await self.r_pending_queue.put("DONE")
        print(f"[DualWorker-{self.rollout_which}][Feed] done, {self.r_global_steps} samples")

    async def _r_process_single_sample(self, rollout_sample: RolloutSample):
        rollout_sample.full_batch.non_tensor_batch["param_version"] = [self.r_current_param_version] * len(
            rollout_sample.full_batch
        )
        ret, is_cancel = await self.r_async_mgr.generate_single_sample_async(
            rollout_sample.full_batch, rollout_sample.agent_loop_output_list
        )
        if not is_cancel:
            rollout_sample.full_batch = ret
            rollout_sample.full_batch.non_tensor_batch["uid"] = np.array(
                [f"uid_{rollout_sample.sample_id}"] * len(rollout_sample.full_batch), dtype=object
            )
            rollout_sample.param_version = self.r_current_param_version
            rollout_sample.rollout_status = await self._r_get_statistics()
            rollout_sample.agent_loop_output_list = []
            success = await self.r_mq_client.put_sample(
                sample=ray.cloudpickle.dumps(rollout_sample),
                param_version=rollout_sample.param_version,
            )
            if success:
                self.r_total_generated_samples += 1
            else:
                self.r_dropped_stale_samples += 1
        else:
            rollout_sample.agent_loop_output_list = ret
            await self.r_cancel_queue.put(rollout_sample)
        self.r_processed_sample_count += 1

    async def _r_should_pause(self) -> bool:
        stats = self.r_mq_client.get_statistics_sync()
        if stats["queue_size"] >= self.r_max_queue_size:
            return True
        if self.r_staleness_samples >= self.r_max_required_samples:
            return True
        return False

    async def _r_processor_worker(self):
        while True:
            if self.r_paused or await self._r_should_pause():
                async with self.r_lock:
                    self.r_paused = True
                while self.r_active_tasks:
                    async with self.r_lock:
                        if self.r_active_tasks:
                            done, self.r_active_tasks = await asyncio.wait(
                                self.r_active_tasks, return_when=asyncio.FIRST_COMPLETED
                            )
                        for t in done:
                            await t
                async with self.r_lock:
                    while self.r_paused:
                        self.r_idle_start_time = time.time()
                        await self.r_condition.wait()
                continue

            from_cancel = False
            if not self.r_cancel_queue.empty():
                rollout_sample = await self.r_cancel_queue.get()
                from_cancel = True
            else:
                rollout_sample = await self.r_pending_queue.get()
                self.r_staleness_samples += 1

            if rollout_sample == "DONE":
                while self.r_active_tasks:
                    async with self.r_lock:
                        if self.r_active_tasks:
                            done, self.r_active_tasks = await asyncio.wait(
                                self.r_active_tasks, return_when=asyncio.FIRST_COMPLETED
                            )
                        for t in done:
                            await t
                break

            while len(self.r_active_tasks) >= self.r_max_concurrent_samples:
                async with self.r_lock:
                    if self.r_active_tasks:
                        done, self.r_active_tasks = await asyncio.wait(
                            self.r_active_tasks, return_when=asyncio.FIRST_COMPLETED
                        )
                    for t in done:
                        await t

            async with self.r_lock:
                while self.r_paused:
                    await self.r_condition.wait()
                task = asyncio.create_task(
                    self._r_process_single_sample(rollout_sample),
                    name=rollout_sample.sample_id,
                )
                self.r_active_tasks.add(task)

            if from_cancel:
                self.r_cancel_queue.task_done()
            else:
                self.r_pending_queue.task_done()

    async def _r_streaming_main(self):
        print(f"[DualWorker-{self.rollout_which}][rollout] starting stream, max_concurrent={self.r_max_concurrent_samples}")
        async with self.r_lock:
            self.r_paused = False
            self.r_running = True

        feed_task = asyncio.create_task(self._r_feed_samples())
        proc_task = asyncio.create_task(self._r_processor_worker())

        try:
            done, pending = await asyncio.wait([feed_task, proc_task], return_when=asyncio.FIRST_COMPLETED)
            for t in done:
                if t.exception():
                    raise t.exception()
            if feed_task not in done:
                raise RuntimeError("Processor task exited prematurely in rollout loop")
            print(f"[DualWorker-{self.rollout_which}][rollout] feed done")
            await proc_task
        except Exception as e:
            print(f"[DualWorker-{self.rollout_which}][rollout] exception: {e}")
        finally:
            proc_task.cancel()
            await asyncio.gather(proc_task, return_exceptions=True)

        # Send termination signal
        await self.r_mq_client.put_sample(sample=None, param_version=self.r_current_param_version)
        async with self.r_lock:
            self.r_running = False

    async def _r_monitor_loop(self):
        last_stats = time.time()
        while True:
            async with self.r_lock:
                if not self.r_running:
                    break
            await asyncio.sleep(10.0)
            if time.time() - last_stats >= 60.0:
                stats = await self._r_get_statistics()
                print(f"[DualWorker-{self.rollout_which}][monitor] {pformat(stats)}")
                last_stats = time.time()
            if self.r_monitor_loop_trigger and not await self._r_should_pause():
                async with self.r_lock:
                    self.r_paused = False
                    self.r_condition.notify_all()

    async def _r_get_statistics(self):
        stats = self.r_mq_client.get_statistics_sync()
        return {
            "monitor/active_tasks": len(self.r_active_tasks),
            "monitor/mq_queue_size": stats["queue_size"],
            "count/current_param_version": self.r_current_param_version,
            "count/total_generated": self.r_total_generated_samples,
            "count/staleness_samples": self.r_staleness_samples,
            "count/dropped_stale": self.r_dropped_stale_samples,
        }

    async def _rollout_fit(self):
        print(f"[DualWorker-{self.rollout_which}] _rollout_fit starting")
        if self.r_mq_client is None:
            raise ValueError("r_mq_client not set, call set_rollout_mq_client() first")
        stream_task = asyncio.create_task(self._r_streaming_main())
        monitor_task = asyncio.create_task(self._r_monitor_loop())
        try:
            await asyncio.gather(stream_task, monitor_task, return_exceptions=True)
        finally:
            stream_task.cancel()
            monitor_task.cancel()
            await asyncio.gather(stream_task, monitor_task, return_exceptions=True)
        print(f"[DualWorker-{self.rollout_which}] _rollout_fit done")

    # ------------------------------------------------------------------
    # Training loop (adapted from FullyAsyncTrainer)
    # ------------------------------------------------------------------

    def _t_get_samples_from_queue(self):
        """Pull required_samples from t_mq_client synchronously."""
        print(f"[DualWorker-{self.train_which}][train] requesting {self.t_required_samples} samples")
        start = time.time()
        queue_samples = []
        while len(queue_samples) < self.t_required_samples:
            sample, _ = self.t_mq_client.get_sample_sync()
            if sample is None:
                print(f"[DualWorker-{self.train_which}][train] got termination signal")
                break
            queue_samples.append(sample)

        if not queue_samples or len(queue_samples) < self.t_required_samples:
            return None

        queue_samples = [ray.cloudpickle.loads(x) for x in queue_samples]
        if self.config.trainer.balance_batch:
            batch = assemble_batch_from_rollout_samples(
                queue_samples, self.tokenizer, self.config, self._t_balance_batch
            )
        else:
            batch = assemble_batch_from_rollout_samples(
                queue_samples, self.tokenizer, self.config, None
            )
        batch.meta_info["fully_async/total_wait_time"] = time.time() - start
        return batch

    def _t_balance_batch(self, batch, metrics=None):
        """Seqlen-balance across DP ranks. Delegates to the base class helper if available."""
        # Simplified: just return as-is; subclasses can override
        return batch

    def _t_collect_metrics_from_samples(self, batch, metrics: dict):
        if hasattr(batch, "meta_info") and batch.meta_info:
            versions = batch.meta_info.get("rollout_param_versions", [])
            stale = sum(1 for v in versions if self.t_current_param_version - v >= 1)
            self.t_stale_samples_processed += stale
            traj_versions = batch.meta_info.get("trajectory_param_versions", [])
            stale_traj = sum(1 for v in traj_versions if self.t_current_param_version - v >= 1)
            self.t_stale_trajectory_processed += stale_traj
            metrics.update({
                f"fully_async_{self.train_which}/stale_samples": self.t_stale_samples_processed,
                f"fully_async_{self.train_which}/current_param_version": self.t_current_param_version,
            })
            for k, v in batch.meta_info.items():
                if k.startswith("fully_async") or k.startswith("timing_s"):
                    metrics[k] = v

    def _t_process_batch(self, batch: DataProto, metrics: dict, timing_raw: dict):
        """Run reward / old_log_prob / ref / adv / actor update for model train_which."""
        with marked_timer("reward", timing_raw):
            if self.config.reward_model.launch_reward_fn_async:
                future_reward = compute_reward_async.remote(data=batch, reward_fn=self.reward_fn)
            else:
                reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)

        with marked_timer("old_log_prob", timing_raw):
            async_training = self.config.get("async_training", None)
            if async_training and async_training.use_rollout_log_probs:
                batch.batch["old_log_probs"] = batch.batch["rollout_log_probs"]
                batch.meta_info["temperature"] = self.r_model_config.rollout.temperature
            else:
                old_log_prob_out = self.t_wg.compute_log_prob(batch)
                entropys = old_log_prob_out.batch["entropys"]
                response_masks = batch.batch["response_mask"]
                actor_cfg = self.t_model_config.actor
                entropy_agg = agg_loss(
                    loss_mat=entropys,
                    loss_mask=response_masks,
                    loss_agg_mode=actor_cfg.loss_agg_mode,
                    loss_scale_factor=actor_cfg.loss_scale_factor,
                )
                metrics[f"actor_{self.train_which}/entropy"] = entropy_agg.detach().item()
                old_log_prob_out.batch.pop("entropys")
                batch = batch.union(old_log_prob_out)

        if self.t_use_reference_policy:
            with marked_timer("ref", timing_raw):
                if not self.t_ref_in_actor:
                    ref_out = self.t_ref_wg.compute_ref_log_prob(batch)
                else:
                    ref_out = self.t_wg.compute_ref_log_prob(batch)
                batch = batch.union(ref_out)

        with marked_timer("adv", timing_raw):
            if self.config.reward_model.launch_reward_fn_async:
                reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
            batch.batch["token_level_scores"] = reward_tensor
            if reward_extra_infos_dict:
                batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

            if self.config.algorithm.use_kl_in_reward:
                batch, kl_metrics = _apply_kl_penalty(batch, kl_ctrl=self.t_kl_ctrl_in_reward,
                                                       kl_penalty=self.config.algorithm.kl_penalty)
                metrics.update(kl_metrics)
            else:
                batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

            norm_adv = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)
            batch = _compute_advantage(
                batch,
                adv_estimator=self.config.algorithm.adv_estimator,
                gamma=self.config.algorithm.gamma,
                lam=self.config.algorithm.lam,
                num_repeat=self.r_model_config.rollout.n,
                norm_adv_by_std_in_grpo=norm_adv,
                config=self.config.algorithm,
            )

        if self.config.trainer.critic_warmup <= self.t_global_steps:
            with marked_timer("update_actor", timing_raw):
                batch.meta_info["multi_turn"] = self.r_model_config.rollout.multi_turn.enable
                actor_out = self.t_wg.update_actor(batch)
            actor_metrics = reduce_metrics(actor_out.meta_info["metrics"])
            metrics.update(actor_metrics)

        return batch, reward_extra_infos_dict

    async def _t_trigger_param_sync(self, validate: bool = False):
        if self.t_local_trigger_step < self.t_trigger_parameter_sync_step and not validate:
            self.t_local_trigger_step += 1
            return

        self.t_current_param_version += 1
        self.t_local_trigger_step = 1

        if self.t_logger:
            self.t_logger.log(
                data=self.t_metrics_aggregator.get_aggregated_metrics(),
                step=self.t_current_param_version,
            )
        self.t_metrics_aggregator.reset()

        timing_param = {}
        with marked_timer("timing_s/param_sync", timing_param):
            ray.get(
                self.t_param_sync.sync_weights.remote(
                    self.t_current_param_version,
                    validate=validate,
                    global_steps=self.t_global_steps,
                    use_trainer_do_validate=False,
                )
            )
        if self.t_logger:
            self.t_logger.log(data=timing_param, step=self.t_current_param_version)

    def _t_save_checkpoint(self, timing_raw: dict):
        if self.t_current_param_version == self.t_last_ckpt_version:
            return
        esi = should_save_ckpt_esi(
            max_steps_duration=self.t_max_steps_duration,
            redundant_time=self.config.trainer.esi_redundant_time,
        )
        if self.config.trainer.save_freq > 0 and (
            self.t_current_param_version % self.config.trainer.save_freq == 0 or esi
        ):
            folder = os.path.join(
                self.config.trainer.default_local_dir,
                f"global_step_{self.t_current_param_version}",
            )
            actor_path = os.path.join(folder, f"actor_{self.train_which}")
            self.t_wg.save_checkpoint(actor_path, None, self.t_current_param_version)
            ray.get(self.t_param_sync.rollouter_save_checkpoint.remote(folder))
            self.t_last_ckpt_version = self.t_current_param_version
            with open(os.path.join(self.config.trainer.default_local_dir, "latest_checkpointed_iteration.txt"), "w") as f:
                f.write(str(self.t_current_param_version))

    async def _train_fit(self):
        print(f"[DualWorker-{self.train_which}] _train_fit starting")
        if self.t_mq_client is None:
            raise ValueError("t_mq_client not set, call set_train_mq_client() first")
        if self.t_param_sync is None:
            raise ValueError("t_param_sync not set, call set_param_synchronizer() first")

        while True:
            metrics = {}
            timing_raw = {}

            with marked_timer("step", timing_raw):
                with marked_timer("gen", timing_raw):
                    batch = self._t_get_samples_from_queue()
                    if batch is None:
                        break
                    self._t_collect_metrics_from_samples(batch, metrics)

                if "response_mask" not in batch.batch:
                    batch.batch["response_mask"] = _compute_response_mask(batch)
                batch.meta_info["global_token_num"] = torch.sum(
                    batch.batch["attention_mask"], dim=-1
                ).tolist()

                batch, reward_extra_infos_dict = self._t_process_batch(batch, metrics, timing_raw)

            steps_dur = timing_raw["step"]
            self.t_max_steps_duration = max(self.t_max_steps_duration, steps_dur)
            metrics.update({f"training_{self.train_which}/global_step": self.t_global_steps})
            self.t_metrics_aggregator.add_step_metrics(
                metrics=metrics, sample_count=self.t_required_samples, timestamp=time.time()
            )

            await self._t_trigger_param_sync()
            self._t_save_checkpoint(timing_raw)
            self.t_global_steps += 1

        # Final param sync
        await self._t_trigger_param_sync(validate=True)
        print(f"[DualWorker-{self.train_which}] _train_fit done")

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    async def fit(self):
        """Run rollout and training concurrently."""
        print(f"[DualWorker-{self.rollout_which}/{self.train_which}] fit() starting")
        await asyncio.gather(
            self._rollout_fit(),
            self._train_fit(),
        )
        print(f"[DualWorker-{self.rollout_which}/{self.train_which}] fit() done")

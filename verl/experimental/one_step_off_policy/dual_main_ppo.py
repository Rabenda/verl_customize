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
"""Entry point for dual async one-step-off-policy training.

GPU layout:
  pool_a (trainer.n_gpus_per_node × trainer.nnodes):
      Model-A rollout (SGLang) + Model-B actor (FSDP) + Model-B ref (FSDP)
  pool_b (rollout.n_gpus_per_node × rollout.nnodes, MUST == pool_a):
      Model-B rollout (SGLang) + Model-A actor (FSDP) + Model-A ref (FSDP)

Usage:
  python -m verl.experimental.one_step_off_policy.dual_main_ppo \\
      +actor_rollout_ref_a.model.path=<model_a> \\
      +actor_rollout_ref_b.model.path=<model_b> \\
      trainer.n_gpus_per_node=2  rollout.n_gpus_per_node=2 ...
"""

import asyncio
import os
import socket

import hydra
import ray

from verl.experimental.one_step_off_policy.dual_ray_trainer import (
    DualOneStepOffRayTrainer,
    _need_reference_policy,
)
from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler
from verl.trainer.ppo.reward import load_reward_manager
from verl.trainer.ppo.utils import Role
from verl.single_controller.ray import ResourcePoolManager
from verl.utils.device import auto_set_device


def create_resource_pool_manager(config) -> ResourcePoolManager:
    """
    pool_a  →  trainer_pool   (trainer.n_gpus_per_node × trainer.nnodes)
    pool_b  →  rollout_pool   (rollout.n_gpus_per_node × rollout.nnodes)
    They must be the same size.
    """
    pool_a = [config.trainer.n_gpus_per_node] * config.trainer.nnodes
    pool_b = [config.rollout.n_gpus_per_node] * config.rollout.nnodes
    assert pool_a == pool_b, (
        f"pool_a={pool_a} and pool_b={pool_b} must be equal-size. "
        "Set trainer.n_gpus_per_node == rollout.n_gpus_per_node."
    )

    resource_pool_spec = {
        "pool_a": pool_a,
        "pool_b": pool_b,
    }
    mapping = {
        Role.RolloutA: "pool_a",
        Role.ActorB:   "pool_a",
        Role.RefB:     "pool_a",
        Role.ActorA:   "pool_b",
        Role.RolloutB: "pool_b",
        Role.RefA:     "pool_b",
    }
    return ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)


def create_role_worker_mapping(config: dict) -> tuple[dict, type]:
    strategy_a = config.actor_rollout_ref_a.actor.strategy
    strategy_b = config.actor_rollout_ref_b.actor.strategy
    assert strategy_a in ("fsdp", "fsdp2"), f"Unsupported strategy: {strategy_a}"
    assert strategy_a == strategy_b, "Both models must use the same strategy"

    from verl.experimental.one_step_off_policy.fsdp_workers import (
        DetachActorWorker,
        DetachAsyncRolloutWorker,
    )
    from verl.single_controller.ray import RayWorkerGroup

    mapping = {
        Role.ActorA:   ray.remote(DetachActorWorker),
        Role.RolloutA: ray.remote(DetachAsyncRolloutWorker),
        Role.ActorB:   ray.remote(DetachActorWorker),
        Role.RolloutB: ray.remote(DetachAsyncRolloutWorker),
    }
    if _need_reference_policy(config):
        mapping[Role.RefA] = ray.remote(DetachActorWorker)
        mapping[Role.RefB] = ray.remote(DetachActorWorker)

    return mapping, RayWorkerGroup


@ray.remote(num_cpus=10, max_concurrency=100)
class DualOneStepTaskRunner:
    def run(self, config):
        from omegaconf import OmegaConf
        from verl.utils.fs import copy_to_local

        print(f"[DualOneStepTaskRunner] hostname={socket.gethostname()} PID={os.getpid()}")
        OmegaConf.resolve(config)

        role_worker_mapping, ray_worker_group_cls = create_role_worker_mapping(config)

        # Download model checkpoints
        local_path_a = copy_to_local(
            config.actor_rollout_ref_a.model.path,
            use_shm=config.actor_rollout_ref_a.model.get("use_shm", False),
        )
        local_path_b = copy_to_local(
            config.actor_rollout_ref_b.model.path,
            use_shm=config.actor_rollout_ref_b.model.get("use_shm", False),
        )

        from verl.utils import hf_tokenizer, hf_processor

        trust = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path_a, trust_remote_code=trust)
        processor = hf_processor(local_path_a, trust_remote_code=trust, use_fast=True)

        reward_fn = load_reward_manager(
            config, tokenizer, num_examine=0, **config.reward_model.get("reward_kwargs", {})
        )
        val_reward_fn = load_reward_manager(
            config, tokenizer, num_examine=1, **config.reward_model.get("reward_kwargs", {})
        )

        resource_pool_manager = create_resource_pool_manager(config)

        from verl.utils.dataset.rl_dataset import collate_fn

        train_dataset = create_rl_dataset(
            config.data.train_files, config.data, tokenizer, processor,
            max_samples=config.data.get("train_max_samples", -1),
        )
        val_dataset = create_rl_dataset(
            config.data.val_files, config.data, tokenizer, processor,
            max_samples=config.data.get("val_max_samples", -1),
        )
        train_sampler = create_rl_sampler(config.data, train_dataset)

        trainer = DualOneStepOffRayTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            collate_fn=collate_fn,
            train_sampler=train_sampler,
            device_name=config.trainer.device,
        )
        trainer.init_workers()
        asyncio.run(trainer.fit())


@hydra.main(config_path="config", config_name="dual_one_step_off_ppo_trainer", version_base=None)
def main(config):
    from time import time
    from verl.trainer.main_ppo import run_ppo

    auto_set_device(config)
    start = time()
    run_ppo(config, task_runner_class=DualOneStepTaskRunner)
    print(f"total time: {time() - start:.2f}s")


if __name__ == "__main__":
    main()

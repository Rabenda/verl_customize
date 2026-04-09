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
Entry point for dual-model fully-async PPO training.

Physical layout
---------------
  GPU-R  (rollout pool)  :  sglang-A  +  FSDP-B   ← DualFullyAsyncWorker(rollout="a", train="b")
  GPU-T  (trainer pool)  :  sglang-B  +  FSDP-A   ← DualFullyAsyncWorker(rollout="b", train="a")

Data flow
---------
  worker_r rolls out A  → MQ-A → worker_t trains A → ParamSync-A → worker_r updates sglang-A weights
  worker_t rolls out B  → MQ-B → worker_r trains B → ParamSync-B → worker_t updates sglang-B weights

Config expectations
-------------------
  config.actor_rollout_ref_a   : full actor_rollout_ref sub-config for model A
  config.actor_rollout_ref_b   : full actor_rollout_ref sub-config for model B
  config.rollout.{nnodes, n_gpus_per_node}  : GPU-R pool size
  config.trainer.{nnodes, n_gpus_per_node}  : GPU-T pool size
"""

import asyncio
import socket
import threading
from pprint import pprint

import hydra
import ray
from omegaconf import OmegaConf

from verl.experimental.fully_async_policy.fully_async_dual_worker import DualFullyAsyncWorker
from verl.experimental.fully_async_policy.message_queue import MessageQueue, MessageQueueClient
from verl.experimental.fully_async_policy.param_sync import ParameterSynchronizer
from verl.trainer.ppo.ray_trainer import ResourcePoolManager
from verl.trainer.ppo.utils import Role
from verl.utils.fs import copy_to_local


# ---------------------------------------------------------------------------
# Resource pool helpers
# ---------------------------------------------------------------------------

def _create_pool_r(config) -> ResourcePoolManager:
    """GPU-R pool: hosts sglang-A (RolloutA) and FSDP-B (ActorB), colocated."""
    pool_spec = [config.rollout.n_gpus_per_node] * config.rollout.nnodes
    return ResourcePoolManager(
        resource_pool_spec={"pool_r": pool_spec},
        mapping={
            Role.RolloutA: "pool_r",
            Role.ActorB:   "pool_r",
            Role.RefB:     "pool_r",   # ref-B colocated with actor-B
        },
    )


def _create_pool_t(config) -> ResourcePoolManager:
    """GPU-T pool: hosts sglang-B (RolloutB) and FSDP-A (ActorA), colocated."""
    pool_spec = [config.trainer.n_gpus_per_node] * config.trainer.nnodes
    return ResourcePoolManager(
        resource_pool_spec={"pool_t": pool_spec},
        mapping={
            Role.RolloutB: "pool_t",
            Role.ActorA:   "pool_t",
            Role.RefA:     "pool_t",   # ref-A colocated with actor-A
        },
    )


def _create_role_worker_mapping(config):
    """Build role → Ray-remote-worker-class mapping for both pools."""
    strategy = config.actor_rollout_ref_a.actor.strategy
    assert strategy in ("fsdp", "fsdp2"), f"Unsupported strategy: {strategy}"

    from verl.experimental.fully_async_policy.fsdp_workers import (
        DetachActorWorker,
        DetachAsyncRolloutWorker,
    )
    from verl.single_controller.ray import RayWorkerGroup

    mapping = {
        Role.RolloutA: ray.remote(DetachAsyncRolloutWorker),
        Role.ActorA:   ray.remote(DetachActorWorker),
        Role.RolloutB: ray.remote(DetachAsyncRolloutWorker),
        Role.ActorB:   ray.remote(DetachActorWorker),
        # Ref policies (same class as Actor, different role string)
        Role.RefA:     ray.remote(DetachActorWorker),
        Role.RefB:     ray.remote(DetachActorWorker),
    }
    return mapping, RayWorkerGroup


# ---------------------------------------------------------------------------
# TaskRunner
# ---------------------------------------------------------------------------

@ray.remote(num_cpus=1)
class DualFullyAsyncTaskRunner:
    """
    Orchestrates the two DualFullyAsyncWorker actors:
      worker_r (GPU-R): rollout A, train B
      worker_t (GPU-T): rollout B, train A
    """

    def __init__(self):
        self.components = {}

    def run(self, config):
        print("[DUAL MAIN] Starting dual fully-async PPO training...")
        self._init(config)
        self._run()

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def _init(self, config) -> None:
        print(f"[DUAL MAIN] hostname={socket.gethostname()}")
        OmegaConf.resolve(config)
        pprint(OmegaConf.to_container(config, resolve=True))

        # Sanity checks
        assert hasattr(config, "actor_rollout_ref_a"), "Need config.actor_rollout_ref_a"
        assert hasattr(config, "actor_rollout_ref_b"), "Need config.actor_rollout_ref_b"
        assert config.data.gen_batch_size == 1, "gen_batch_size must be 1 for fully-async"
        assert config.data.train_batch_size == 0, "train_batch_size must be 0 for fully-async"

        # Tokenizer
        local_path = copy_to_local(
            config.actor_rollout_ref_a.model.path,
            use_shm=config.actor_rollout_ref_a.model.get("use_shm", False),
        )
        from verl.utils import hf_processor, hf_tokenizer
        trust = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust)
        processor = hf_processor(local_path, trust_remote_code=trust, use_fast=True)
        self.components["tokenizer"] = tokenizer
        self.components["processor"] = processor
        self.components["config"] = config

        role_worker_mapping, ray_worker_group_cls = _create_role_worker_mapping(config)
        self.components["role_worker_mapping"] = role_worker_mapping
        self.components["ray_worker_group_cls"] = ray_worker_group_cls

        # Create the two workers
        print("[DUAL MAIN] Creating worker_r (rollout=A, train=B) on GPU-R pool ...")
        worker_r = self._create_worker("a", "b", _create_pool_r(config))
        self.components["worker_r"] = worker_r

        print("[DUAL MAIN] Creating worker_t (rollout=B, train=A) on GPU-T pool ...")
        worker_t = self._create_worker("b", "a", _create_pool_t(config))
        self.components["worker_t"] = worker_t

        # Message queues: MQ-A (worker_r produces, worker_t consumes)
        #                 MQ-B (worker_t produces, worker_r consumes)
        max_queue_size_r = ray.get(worker_r.get_max_queue_size.remote())
        max_queue_size_t = ray.get(worker_t.get_max_queue_size.remote())
        print(f"[DUAL MAIN] MQ-A max_size={max_queue_size_r}, MQ-B max_size={max_queue_size_t}")

        mq_a = MessageQueue.remote(config, max_queue_size_r)
        mq_b = MessageQueue.remote(config, max_queue_size_t)
        mq_a_client = MessageQueueClient(mq_a)
        mq_b_client = MessageQueueClient(mq_b)
        self.components.update({"mq_a": mq_a, "mq_b": mq_b,
                                 "mq_a_client": mq_a_client, "mq_b_client": mq_b_client})

        # Wire MQs: worker_r produces to MQ-A, consumes from MQ-B
        ray.get(worker_r.set_rollout_mq_client.remote(mq_a_client))
        ray.get(worker_r.set_train_mq_client.remote(mq_b_client))
        # worker_t produces to MQ-B, consumes from MQ-A
        ray.get(worker_t.set_rollout_mq_client.remote(mq_b_client))
        ray.get(worker_t.set_train_mq_client.remote(mq_a_client))

        # Parameter synchronizers (cross-group NCCL)
        #   ParamSync-A: actor_wg = worker_t's FSDP-A → rollout_wg = worker_r's sglang-A
        #                rollouter = worker_r (pause/resume the sglang-A server)
        #   ParamSync-B: actor_wg = worker_r's FSDP-B → rollout_wg = worker_t's sglang-B
        #                rollouter = worker_t (pause/resume the sglang-B server)
        print("[DUAL MAIN] Creating ParameterSynchronizer-A (FSDP-A on T → sglang-A on R) ...")
        actor_a_wg = ray.get(worker_t.get_train_wg.remote())
        rollout_a_wg = ray.get(worker_r.get_rollout_wg.remote())
        param_sync_a = ParameterSynchronizer.remote(
            config=config,
            rollouter=worker_r,
            mq=mq_a_client,
            actor_wg=actor_a_wg,
            rollout_wg=rollout_a_wg,
            sync_group_name="actor_rollout_a",
        )

        print("[DUAL MAIN] Creating ParameterSynchronizer-B (FSDP-B on R → sglang-B on T) ...")
        actor_b_wg = ray.get(worker_r.get_train_wg.remote())
        rollout_b_wg = ray.get(worker_t.get_rollout_wg.remote())
        param_sync_b = ParameterSynchronizer.remote(
            config=config,
            rollouter=worker_t,
            mq=mq_b_client,
            actor_wg=actor_b_wg,
            rollout_wg=rollout_b_wg,
            sync_group_name="actor_rollout_b",
        )
        self.components.update({"param_sync_a": param_sync_a, "param_sync_b": param_sync_b})

        # Wire param syncs: each worker's training side uses the sync that pushes ITS trained weights
        ray.get(worker_t.set_param_synchronizer.remote(param_sync_a))  # worker_t trains A → pushes via sync_a
        ray.get(worker_r.set_param_synchronizer.remote(param_sync_b))  # worker_r trains B → pushes via sync_b

        # Load checkpoints + initial weight sync
        val_before_train = config.trainer.get("val_before_train", True)
        param_version_r = ray.get(worker_r.load_train_checkpoint.remote())
        param_version_t = ray.get(worker_t.load_train_checkpoint.remote())
        ray.get(worker_r.load_rollout_checkpoint.remote())
        ray.get(worker_t.load_rollout_checkpoint.remote())

        # Initial sync: push initial weights to sglang servers
        ray.get(param_sync_a.sync_weights.remote(
            version=param_version_t,
            validate=val_before_train,
            use_trainer_do_validate=False,
        ))
        ray.get(param_sync_a.wait_last_valid.remote())

        ray.get(param_sync_b.sync_weights.remote(
            version=param_version_r,
            validate=val_before_train,
            use_trainer_do_validate=False,
        ))
        ray.get(param_sync_b.wait_last_valid.remote())

        print("[DUAL MAIN] All components initialized.")

    def _create_worker(self, rollout_which: str, train_which: str, pool_manager: ResourcePoolManager):
        config = self.components["config"]
        worker = DualFullyAsyncWorker.remote(
            config=config,
            rollout_which=rollout_which,
            train_which=train_which,
            tokenizer=self.components["tokenizer"],
            role_worker_mapping=self.components["role_worker_mapping"],
            resource_pool_manager=pool_manager,
            ray_worker_group_cls=self.components["ray_worker_group_cls"],
            processor=self.components["processor"],
            device_name=config.trainer.device,
        )
        ray.get(worker.init_workers.remote())
        ray.get(worker.set_max_required_samples.remote())
        return worker

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def _run(self):
        worker_r = self.components["worker_r"]
        worker_t = self.components["worker_t"]

        print("[DUAL MAIN] Launching worker_r.fit() and worker_t.fit() ...")
        fut_r = worker_r.fit.remote()
        fut_t = worker_t.fit.remote()
        futures = [fut_r, fut_t]
        try:
            while futures:
                done, futures = ray.wait(futures, num_returns=1, timeout=None)
                for f in done:
                    try:
                        ray.get(f)
                        print("[DUAL MAIN] One worker completed")
                    except Exception as e:
                        print(f"[DUAL MAIN] Worker failed: {e}")
                        for rem in futures:
                            ray.cancel(rem)
                        raise
        except Exception as e:
            print(f"[DUAL MAIN] Training failed: {e}")
            for f in futures:
                ray.cancel(f)
            raise
        finally:
            print("[DUAL MAIN] Training finished or interrupted")


# ---------------------------------------------------------------------------
# Hydra entry point
# ---------------------------------------------------------------------------

@hydra.main(config_path="config", config_name="fully_async_dual_ppo_trainer", version_base=None)
def main(config):
    from verl.trainer.main_ppo import run_ppo
    if not hasattr(config, "async_training"):
        raise RuntimeError("Must set async_training config")
    from time import time
    t0 = time()
    run_ppo(config, task_runner_class=DualFullyAsyncTaskRunner)
    print(f"[DUAL MAIN] Total time: {time() - t0:.2f}s")


if __name__ == "__main__":
    main()

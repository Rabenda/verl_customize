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
Note that we don't combine the main with ray_trainer as ray_trainer is used by other main.
"""

import os
import socket
from copy import deepcopy
from typing import Optional, Tuple

import hydra
import ray
from omegaconf import OmegaConf, open_dict

from verl.experimental.dataset.sampler import AbstractSampler
from verl.trainer.constants_ppo import get_ppo_ray_runtime_env
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.trainer.ppo.ray_dual_trainer import DualRayPPOTrainer
from verl.trainer.ppo.reward import load_reward_manager
from verl.trainer.ppo.utils import need_critic, need_reference_policy
from verl.utils.config import validate_config
from verl.utils.device import auto_set_device, is_cuda_available
from verl.utils.import_utils import load_extern_object

from dataclasses import fields as dc_fields

def _apply_precision_to_hfmodelconfig(cfg, desired: str):
    """
    desired: "bfloat16" / "float16" / "float32"
    This function INSPECTS HFModelConfig fields at runtime and routes precision
    into a supported knob if any, otherwise raises a clear error telling you
    which fields exist.
    """
    from verl.workers.config.model import HFModelConfig

    # What HFModelConfig actually accepts in this verl version
    allowed = {f.name for f in dc_fields(HFModelConfig)}

    # User might pass dtype/torch_dtype in hydra overrides; strip them to avoid crash
    with open_dict(cfg):
        if isinstance(cfg.model, dict):
            cfg.model.pop("dtype", None)
            cfg.model.pop("torch_dtype", None)
        else:
            # DictConfig behaves like mapping too
            try:
                cfg.model.pop("dtype", None)
                cfg.model.pop("torch_dtype", None)
            except Exception:
                pass

    # Route desired precision into a field that *actually exists*
    # Try common field names across different verl revisions
    candidates = [
        ("model_dtype", desired),
        ("param_dtype", desired),
        ("compute_dtype", desired),
        ("precision", desired),
        ("fp16", desired in ("float16", "fp16")),
        ("bf16", desired in ("bfloat16", "bf16")),
    ]

    # Try "kwargs for from_pretrained" style fields if present
    kw_candidates = [
        "from_pretrained_kwargs",
        "hf_from_pretrained_kwargs",
        "hf_load_kwargs",
        "model_init_kwargs",
        "hf_model_init_kwargs",
        "model_kwargs",
    ]

    applied = False

    with open_dict(cfg):
        # 1) direct dtype-ish knobs
        for k, v in candidates:
            if k in allowed:
                cfg.model[k] = v
                applied = True
                break

        # 2) kwargs passthrough knobs
        if not applied:
            for kk in kw_candidates:
                if kk in allowed:
                    if cfg.model.get(kk) is None:
                        cfg.model[kk] = {}
                    cfg.model[kk]["torch_dtype"] = desired
                    applied = True
                    break

    if not applied:
        # Fail loudly with actionable info
        raise RuntimeError(
            "This verl version's HFModelConfig exposes no precision/dtype field.\n"
            f"HFModelConfig fields = {sorted(list(allowed))}\n"
            "You need to either:\n"
            "  (a) set precision via whatever field your HFModelConfig supports, or\n"
            "  (b) patch verl/workers/config/model.py to add a dtype passthrough, or\n"
            "  (c) force AMP/autocast in engine (not config-level).\n"
        )

# ---------------------------
# Dual config helpers
# ---------------------------

def _require_has(obj, key_path: str):
    """Raise a clear error if OmegaConf select fails."""
    if OmegaConf.select(obj, key_path) is None:
        raise ValueError(f"Missing config field: {key_path}")


def _make_actor_cfg_from_shared(config, which: str):
    """
    Build actor_rollout_ref_a/b from shared actor_rollout_ref, using:
      actor_rollout_ref.model.path_a / path_b
      actor_rollout_ref.rollout.n_a / n_b

    If actor_rollout_ref_a/b already exist, just use them (new style).
    """
    assert which in ("a", "b")

    direct = getattr(config, f"actor_rollout_ref_{which}", None)
    if direct is not None:
        return deepcopy(direct)

    shared = getattr(config, "actor_rollout_ref", None)
    if shared is None:
        raise ValueError(
            "Dual mode requires either:\n"
            "  (1) config.actor_rollout_ref_a and config.actor_rollout_ref_b\n"
            "or\n"
            "  (2) shared config.actor_rollout_ref with model.path_a/path_b and rollout.n_a/n_b"
        )

    path_key = f"actor_rollout_ref.model.path_{which}"
    n_key = f"actor_rollout_ref.rollout.n_{which}"
    _require_has(config, path_key)
    _require_has(config, n_key)

    scoped = deepcopy(shared)
    with open_dict(scoped):
        scoped.model.path = OmegaConf.select(config, path_key)
        scoped.rollout.n = OmegaConf.select(config, n_key)

        # keep a clean config: remove the dual-only keys if present
        # (some downstream strict config / struct mode may dislike unknown keys)
        if hasattr(scoped.model, "path_a"):
            del scoped.model["path_a"]
        if hasattr(scoped.model, "path_b"):
            del scoped.model["path_b"]
        if hasattr(scoped.rollout, "n_a"):
            del scoped.rollout["n_a"]
        if hasattr(scoped.rollout, "n_b"):
            del scoped.rollout["n_b"]

    return scoped


def _ensure_dual_configs(config):
    """
    Ensure config has:
      - actor_rollout_ref_a
      - actor_rollout_ref_b
    and ALSO keep config.actor_rollout_ref as a "default" block for code paths
    that still look up actor_rollout_ref.* (legacy assumptions).
    """
    cfg_a = _make_actor_cfg_from_shared(config, "a")
    cfg_b = _make_actor_cfg_from_shared(config, "b")

    with open_dict(config):
        config.actor_rollout_ref_a = cfg_a
        config.actor_rollout_ref_b = cfg_b

        # Important compatibility trick:
        # Some components (esp. agent_loop default manager) may access config.actor_rollout_ref.*
        # We set it to cfg_a as a default so those lookups succeed.
        # (A/B specific logic should use actor_rollout_ref_a/b.)
        if getattr(config, "actor_rollout_ref", None) is None:
            config.actor_rollout_ref = deepcopy(cfg_a)

    return cfg_a, cfg_b


@hydra.main(config_path="config", config_name="ppo_trainer", version_base=None)
def main(config):
    auto_set_device(config)
    run_ppo(config)


def run_ppo(config, task_runner_class=None) -> None:
    if not ray.is_initialized():
        default_runtime_env = get_ppo_ray_runtime_env()
        ray_init_kwargs = config.ray_kwargs.get("ray_init", {})
        runtime_env_kwargs = ray_init_kwargs.get("runtime_env", {})

        if config.transfer_queue.enable:
            runtime_env_vars = runtime_env_kwargs.get("env_vars", {})
            runtime_env_vars["TRANSFER_QUEUE_ENABLE"] = "1"
            runtime_env_kwargs["env_vars"] = runtime_env_vars

        runtime_env = OmegaConf.merge(default_runtime_env, runtime_env_kwargs)
        ray_init_kwargs = OmegaConf.create({**ray_init_kwargs, "runtime_env": runtime_env})
        print(f"ray init kwargs: {ray_init_kwargs}")
        ray.init(**OmegaConf.to_container(ray_init_kwargs))

    if task_runner_class is None:
        task_runner_class = ray.remote(num_cpus=1)(TaskRunner)

    if (
        is_cuda_available
        and config.global_profiler.tool == "nsys"
        and config.global_profiler.get("steps") is not None
        and len(config.global_profiler.get("steps", [])) > 0
    ):
        from verl.utils.import_utils import is_nvtx_available

        assert is_nvtx_available(), "nvtx is not available in CUDA platform. Please 'pip3 install nvtx'"
        nsight_options = OmegaConf.to_container(
            config.global_profiler.global_tool_config.nsys.controller_nsight_options
        )
        runner = task_runner_class.options(runtime_env={"nsight": nsight_options}).remote()
    else:
        runner = task_runner_class.remote()

    ray.get(runner.run.remote(config))

    timeline_json_file = config.ray_kwargs.get("timeline_json_file", None)
    if timeline_json_file:
        ray.timeline(filename=timeline_json_file)


class TaskRunner:
    def __init__(self):
        self.role_worker_mapping = {}
        self.mapping = {}

    # ---------------------------
    # Single model workers (unchanged)
    # ---------------------------
    def add_actor_rollout_worker(self, config):
        from verl.single_controller.ray import RayWorkerGroup
        from verl.trainer.ppo.ray_trainer import Role

        use_legacy_worker_impl = config.trainer.get("use_legacy_worker_impl", "auto")

        if use_legacy_worker_impl == "disable":
            from verl.workers.engine_workers import ActorRolloutRefWorker

            actor_rollout_cls = ActorRolloutRefWorker
            ray_worker_group_cls = RayWorkerGroup

            lora_rank = config.actor_rollout_ref.model.get("lora", {}).get("rank", 0)
            if lora_rank <= 0:
                lora_rank = config.actor_rollout_ref.model.get("lora_rank", 0)
            ref_in_actor = lora_rank > 0 or config.actor_rollout_ref.model.get("lora_adapter_path") is not None
            if need_reference_policy(config) and not ref_in_actor:
                role = Role.ActorRolloutRef
            else:
                role = Role.ActorRollout
            self.role_worker_mapping[role] = ray.remote(actor_rollout_cls)
            self.mapping[role] = "global_pool"
            return actor_rollout_cls, ray_worker_group_cls

        if config.actor_rollout_ref.actor.strategy in {"fsdp", "fsdp2"}:
            from verl.workers.fsdp_workers import AsyncActorRolloutRefWorker
            actor_rollout_cls = AsyncActorRolloutRefWorker
            ray_worker_group_cls = RayWorkerGroup

        elif config.actor_rollout_ref.actor.strategy == "megatron":
            from verl.workers.megatron_workers import AsyncActorRolloutRefWorker
            actor_rollout_cls = AsyncActorRolloutRefWorker
            ray_worker_group_cls = RayWorkerGroup
        else:
            raise NotImplementedError

        self.role_worker_mapping[Role.ActorRollout] = ray.remote(actor_rollout_cls)
        self.mapping[Role.ActorRollout] = "global_pool"
        return actor_rollout_cls, ray_worker_group_cls

    # ---------------------------
    # Dual model workers (fixed)
    # ---------------------------
    def add_actor_rollout_workers(self, config):
        """
        IMPORTANT FIX:
        - Do NOT assert config.actor_rollout_ref exists / has lora fields.
        - Dual mode uses actor_rollout_ref_a/b.
        """
        from verl.trainer.ppo.ray_trainer import Role
        from verl.single_controller.ray import RayWorkerGroup
        from verl.workers.engine_workers import ActorRolloutRefWorker

        assert config.trainer.get("use_legacy_worker_impl", "auto") == "disable"

        # We do not support lora for now in dual mode (check both)
        lora_a = OmegaConf.select(config, "actor_rollout_ref_a.model.lora.rank") or 0
        lora_b = OmegaConf.select(config, "actor_rollout_ref_b.model.lora.rank") or 0
        assert int(lora_a) <= 0 and int(lora_b) <= 0, "Dual mode currently requires lora.rank == 0 for both A/B."

        self.role_worker_mapping[Role.ActorRolloutA] = ray.remote(ActorRolloutRefWorker)
        self.role_worker_mapping[Role.ActorRolloutB] = ray.remote(ActorRolloutRefWorker)

        self.mapping[Role.ActorRolloutA] = "global_pool"
        self.mapping[Role.ActorRolloutB] = "global_pool"

        return ActorRolloutRefWorker, RayWorkerGroup

    def add_critic_worker(self, config):
        use_legacy_worker_impl = config.trainer.get("use_legacy_worker_impl", "auto")
        if config.critic.strategy in {"fsdp", "fsdp2"}:
            if use_legacy_worker_impl in ["auto", "enable"]:
                from verl.workers.fsdp_workers import CriticWorker
            elif use_legacy_worker_impl == "disable":
                from verl.workers.engine_workers import TrainingWorker
                CriticWorker = TrainingWorker
                print("Using new worker implementation")
            else:
                raise ValueError(f"Invalid use_legacy_worker_impl: {use_legacy_worker_impl}")
        elif config.critic.strategy == "megatron":
            from verl.workers.megatron_workers import CriticWorker
        else:
            raise NotImplementedError

        from verl.trainer.ppo.ray_trainer import Role
        self.role_worker_mapping[Role.Critic] = ray.remote(CriticWorker)
        self.mapping[Role.Critic] = "global_pool"

    def init_resource_pool_mgr(self, config):
        global_pool_id = "global_pool"
        resource_pool_spec = {
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }
        if config.reward_model.enable_resource_pool:
            if config.reward_model.n_gpus_per_node <= 0:
                raise ValueError("config.reward_model.n_gpus_per_node must be greater than 0")
            if config.reward_model.nnodes <= 0:
                raise ValueError("config.reward_model.nnodes must be greater than 0")
            reward_pool = [config.reward_model.n_gpus_per_node] * config.reward_model.nnodes
            resource_pool_spec["reward_pool"] = reward_pool

        from verl.trainer.ppo.ray_trainer import ResourcePoolManager
        return ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=self.mapping)

    def add_reward_model_worker(self, config):
        from verl.trainer.ppo.ray_trainer import Role

        if config.reward_model.enable:
            use_legacy_worker_impl = config.trainer.get("use_legacy_worker_impl", "auto")
            if use_legacy_worker_impl in ["auto", "enable", "disable"]:
                if config.reward_model.strategy in {"fsdp", "fsdp2"}:
                    from verl.workers.fsdp_workers import RewardModelWorker
                elif config.reward_model.strategy == "megatron":
                    from verl.workers.megatron_workers import RewardModelWorker
                else:
                    raise NotImplementedError
            else:
                raise ValueError(f"Invalid use_legacy_worker_impl: {use_legacy_worker_impl}")

            self.role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
            self.mapping[Role.RewardModel] = "reward_pool" if config.reward_model.enable_resource_pool else "global_pool"

    def add_ref_policy_worker(self, config, ref_policy_cls):
        from verl.trainer.ppo.ray_trainer import Role
        use_legacy_worker_impl = config.trainer.get("use_legacy_worker_impl", "auto")
        if use_legacy_worker_impl == "disable":
            return
        if need_reference_policy(config):
            self.role_worker_mapping[Role.RefPolicy] = ray.remote(ref_policy_cls)
            self.mapping[Role.RefPolicy] = "global_pool"

    # ---------------------------
    # Main run
    # ---------------------------
    def run(self, config):
        if getattr(config.trainer, "dual_model", False):
            return self.run_dual_model(config)

        from pprint import pprint
        from verl.utils.fs import copy_to_local
        from verl.utils import hf_processor, hf_tokenizer

        print(f"TaskRunner hostname: {socket.gethostname()}, PID: {os.getpid()}")
        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)

        actor_rollout_cls, ray_worker_group_cls = self.add_actor_rollout_worker(config)
        self.add_critic_worker(config)
        self.add_reward_model_worker(config)
        self.add_ref_policy_worker(config, actor_rollout_cls)

        validate_config(
            config=config,
            use_reference_policy=need_reference_policy(config),
            use_critic=need_critic(config),
        )

        local_path = copy_to_local(
            config.actor_rollout_ref.model.path,
            use_shm=config.actor_rollout_ref.model.get("use_shm", False),
        )

        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)

        reward_fn = load_reward_manager(config, tokenizer, num_examine=0, **config.reward_model.get("reward_kwargs", {}))
        val_reward_fn = load_reward_manager(config, tokenizer, num_examine=1, **config.reward_model.get("reward_kwargs", {}))

        resource_pool_manager = self.init_resource_pool_mgr(config)

        from verl.utils.dataset.rl_dataset import collate_fn

        train_dataset = create_rl_dataset(
            config.data.train_files,
            config.data,
            tokenizer,
            processor,
            is_train=True,
            max_samples=config.data.get("train_max_samples", -1),
        )
        val_dataset = create_rl_dataset(
            config.data.val_files,
            config.data,
            tokenizer,
            processor,
            is_train=False,
            max_samples=config.data.get("val_max_samples", -1),
        )
        train_sampler = create_rl_sampler(config.data, train_dataset)

        trainer = RayPPOTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=self.role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            collate_fn=collate_fn,
            train_sampler=train_sampler,
        )
        trainer.init_workers()
        trainer.fit()

    def run_dual_model(self, config):
        """
        Dual-actor PPO training:
        - ActorRolloutA
        - ActorRolloutB
        - shared critic / reward / ref
        """
        import os
        import socket
        from copy import deepcopy
        from pprint import pprint

        from omegaconf import OmegaConf, open_dict
        from verl.trainer.ppo.utils import need_critic, need_reference_policy
        from verl.utils.config import validate_config
        from verl.trainer.ppo.reward import load_reward_manager
        from verl.utils.fs import copy_to_local

        print(f"[DualModel] Hostname: {socket.gethostname()}, PID: {os.getpid()}")
        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)

        assert hasattr(config, "actor_rollout_ref"), "config.actor_rollout_ref must exist (base schema)"

        # -----------------------------
        # Deterministic deep merge utils
        # -----------------------------
        def _to_dict(cfg):
            if cfg is None:
                return None
            return OmegaConf.to_container(cfg, resolve=False)

        def _deep_merge(base, override):
            """Recursively merge override into base (override wins), without dropping base keys."""
            if override is None:
                return base
            if base is None:
                return override
            # dict <- dict
            if isinstance(base, dict) and isinstance(override, dict):
                out = dict(base)
                for k, v in override.items():
                    if k in out:
                        out[k] = _deep_merge(out[k], v)
                    else:
                        out[k] = v
                return out
            # list: override replaces (safe default)
            if isinstance(override, list):
                return override
            # scalar / type mismatch: override replaces
            return override

        base_shared = deepcopy(config.actor_rollout_ref)
        base_shared_dict = _to_dict(base_shared)

        def _build_full_actor_cfg(which: str):
            assert which in ("a", "b")
            node = getattr(config, f"actor_rollout_ref_{which}", None)
            node_dict = _to_dict(node)

            merged_dict = _deep_merge(deepcopy(base_shared_dict), node_dict)

            # Safety: if user used path_a/path_b or n_a/n_b, map them.
            # (Your current CLI uses model.path and rollout.n directly; these are just guards.)
            model = merged_dict.get("model", {})
            if not model.get("path"):
                p = model.get(f"path_{which}")
                if p:
                    model["path"] = p
            model.pop("path_a", None)
            model.pop("path_b", None)
            merged_dict["model"] = model

            rollout = merged_dict.get("rollout", {})
            if rollout.get("n") is None:
                n = rollout.get(f"n_{which}")
                if n is not None:
                    rollout["n"] = n
            rollout.pop("n_a", None)
            rollout.pop("n_b", None)
            merged_dict["rollout"] = rollout

            # Critical: ensure actor._target_ exists (must come from base)
            actor = merged_dict.get("actor", {})
            base_actor = base_shared_dict.get("actor", {}) if isinstance(base_shared_dict, dict) else {}
            if isinstance(base_actor, dict) and "_target_" in base_actor and "_target_" not in actor:
                actor["_target_"] = base_actor["_target_"]
            # also keep actor.strategy etc from base if missing (deep_merge should already keep them,
            # but this is an extra guard for your exact failure mode)
            if isinstance(base_actor, dict):
                for k, v in base_actor.items():
                    if k not in actor:
                        actor[k] = v
            merged_dict["actor"] = actor

            return OmegaConf.create(merged_dict)

        cfg_a = _build_full_actor_cfg("a")
        cfg_b = _build_full_actor_cfg("b")

        # _apply_precision_to_hfmodelconfig(cfg_a, "bfloat16")
        # _apply_precision_to_hfmodelconfig(cfg_b, "bfloat16")

        # Write back full configs
        with open_dict(config):
            config.actor_rollout_ref_a = cfg_a
            config.actor_rollout_ref_b = cfg_b

            # IMPORTANT: validate_config() reads config.actor_rollout_ref.*
            # Make it point to a complete config (A as canonical).
            config.actor_rollout_ref = cfg_a

        # ---- workers / trainer setup ----
        actor_rollout_cls, ray_worker_group_cls = self.add_actor_rollout_workers(config)

        self.add_critic_worker(config)
        self.add_reward_model_worker(config)
        self.add_ref_policy_worker(config, actor_rollout_cls)

        validate_config(
            config=config,
            use_reference_policy=need_reference_policy(config),
            use_critic=need_critic(config),
        )

        # tokenizer / processor shared (use A)
        local_path = copy_to_local(
            config.actor_rollout_ref_a.model.path,
            use_shm=config.actor_rollout_ref_a.model.get("use_shm", False),
        )

        from verl.utils import hf_processor, hf_tokenizer
        tokenizer = hf_tokenizer(local_path, trust_remote_code=config.data.get("trust_remote_code", False))
        processor = hf_processor(local_path, trust_remote_code=False, use_fast=True)

        reward_fn = load_reward_manager(config, tokenizer, num_examine=0, **config.reward_model.get("reward_kwargs", {}))
        val_reward_fn = load_reward_manager(config, tokenizer, num_examine=1, **config.reward_model.get("reward_kwargs", {}))

        resource_pool_manager = self.init_resource_pool_mgr(config)

        from verl.utils.dataset.rl_dataset import collate_fn
        train_dataset = create_rl_dataset(config.data.train_files, config.data, tokenizer, processor, is_train=True)
        val_dataset = create_rl_dataset(config.data.val_files, config.data, tokenizer, processor, is_train=False)
        train_sampler = create_rl_sampler(config.data, train_dataset)

        trainer = DualRayPPOTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=self.role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            collate_fn=collate_fn,
            train_sampler=train_sampler,
        )

        trainer.init_workers()
        trainer.fit()

def create_rl_dataset(data_paths, data_config, tokenizer, processor, is_train=True, max_samples: int = -1):
    from verl.utils.dataset.rl_dataset import get_dataset_class
    dataset_cls = get_dataset_class(data_config)
    return dataset_cls(
        data_files=data_paths,
        tokenizer=tokenizer,
        processor=processor,
        config=data_config,
        max_samples=max_samples,
    )


def create_rl_sampler(data_config, dataset):
    import torch
    from torch.utils.data import SequentialSampler
    from torchdata.stateful_dataloader.sampler import RandomSampler

    if data_config.sampler is not None and data_config.sampler.get("class_path", None) is not None:
        curriculum_class = load_extern_object(
            data_config.sampler.class_path,
            data_config.sampler.class_name,
        )
        sampler = curriculum_class(
            data_source=dataset,
            data_config=data_config,
        )
        assert isinstance(sampler, AbstractSampler)
        assert data_config.get("dataloader_num_workers", 8) == 0, (
            "If using curriculum, num_workers must be 0 to prevent data caching. "
            "If the dataloader caches data before the batch is done the "
            "curriculum sampler won't have the opportunity to reorder it. "
        )
    elif data_config.shuffle:
        train_dataloader_generator = torch.Generator()
        seed = data_config.get("seed")
        if seed is not None:
            train_dataloader_generator.manual_seed(seed)
        sampler = RandomSampler(data_source=dataset, generator=train_dataloader_generator)
    else:
        sampler = SequentialSampler(data_source=dataset)

    return sampler


if __name__ == "__main__":
    main()
# Copyright 2023-2024 SGLang Team
# Copyright 2025 Bytedance Ltd. and/or its affiliates
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
import asyncio
import dataclasses
import json
import logging
import os
from typing import Any, Optional
from uuid import uuid4

import ray
import sglang
import sglang.srt.entrypoints.engine
from sglang.version import __version__ as _sglang_version
import torch
from packaging import version
from ray.actor import ActorHandle
from sglang.srt.entrypoints.http_server import (
    ServerArgs,
    _GlobalState,
    _launch_subprocesses,
    app,
    set_global_state,
)
from sglang.srt.managers.io_struct import (
    GenerateReqInput,
    ReleaseMemoryOccupationReqInput,
    ResumeMemoryOccupationReqInput,
)
from sglang.srt.managers.tokenizer_manager import ServerStatus

from verl.single_controller.ray import RayClassWithInitArgs
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.device import get_visible_devices_keyword
from verl.utils.net_utils import get_free_port, is_valid_ipv6_address
from verl.utils.profiler.profile import DistProfiler
from verl.workers.config import HFModelConfig, RolloutConfig
from verl.workers.rollout.replica import RolloutMode, RolloutReplica, TokenOutput
from verl.workers.rollout.sglang_rollout.sglang_rollout import ServerAdapter, _set_envs_and_config
from verl.workers.rollout.utils import get_max_position_embeddings, run_unvicorn

logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)

visible_devices_keyword = get_visible_devices_keyword()


class SGLangProfilerArgsBuilder:
    """Builder for SGLang profiling parameters, decoupling profiler parameter logic from the core service class."""

    def __init__(
        self,
        profiler_controller: DistProfiler,
        rollout_config: RolloutConfig,
        replica_rank: int,
    ):
        self.profiler_controller = profiler_controller
        self.rollout_config = rollout_config
        self.replica_rank = replica_rank
        self.auto_stop_profiling = False

    def build_profile_args(self, **kwargs) -> dict[str, Any]:
        global_step = kwargs.pop("global_step", 0)
        config = self.profiler_controller.tool_config
        contents = self.profiler_controller.tool_config.contents

        save_path = os.path.join(
            self.rollout_config.profiler.save_path,
            f"rollout_step_{global_step}",
            f"agent_loop_replica_{self.replica_rank}",
        )
        os.makedirs(save_path, exist_ok=True)

        profiler_tool = self.rollout_config.profiler.tool
        activities: Optional[list[str]] = None
        if contents and profiler_tool:
            activities_tmp = []
            check_map = {
                "cpu": ("CPU", "torch"),
                "cuda|gpu": ("GPU", "torch"),
                "MEM": ("MEM", "torch_memory"),
            }
            for key, (act, tool) in check_map.items():
                if any(k in contents for k in key.split("|")):
                    activities_tmp.append(act)
                    if profiler_tool != tool:
                        raise ValueError(f"{act} profiling requires '{tool}' (got '{profiler_tool}')")
            for unsupported in ("CUDA_PROFILER", "RPD"):
                if unsupported in contents:
                    raise NotImplementedError(f"{unsupported} profiling is not supported")
            activities = activities_tmp if len(activities_tmp) > 0 else activities

        with_stack = bool(contents) and "stack" in contents
        record_shapes = bool(contents) and "shapes" in contents
        # Profiling by stage of Prefill or Decode
        profile_by_stage = bool(contents) and "profile-by-stage" in contents
        # Merge profiles from all ranks into a single trace
        merge_profiles = bool(contents) and "merge-profiles" in contents

        # Rollout start step must be greater than 0 for sglang
        rollout_start_step = config.step_start if config.step_end is not None else 1
        rollout_end_step = config.step_end if config.step_end is not None else -1
        rollout_num_steps = rollout_end_step - rollout_start_step
        self.auto_stop_profiling = rollout_num_steps > 0

        # num_steps must be greater than 0 or None in SGLang.
        rollout_num_steps = None if rollout_num_steps <= 0 else rollout_num_steps

        if rollout_num_steps is None and profile_by_stage:
            raise Exception(
                "profile_by_stage requires rollout_num_steps to be set (possible limitation in sglang <= 0.5.5)"
            )

        # start_step must be greater than 0 for sglang
        rollout_start_step = max(rollout_start_step, 1)

        return {
            "start_step": rollout_start_step,
            "num_steps": rollout_num_steps,
            "activities": activities,
            "with_stack": with_stack,
            "record_shapes": record_shapes,
            "output_dir": save_path,
            "profile_by_stage": profile_by_stage,
            "merge_profiles": merge_profiles,
        }, self.auto_stop_profiling


class SGLangHttpServer:
    """SGLang http server in single node, this is equivalent to launch server with command line:
    ```
    python -m sglang.launch_server --node-rank 0 --nnode 1 ...
    ```

    Args:
        config (DictConfig): full config.
        rollout_mode (RolloutMode): rollout mode.
        replica_rank (int): replica rank, a replica may contain multiple nodes.
        node_rank (int): node rank.
        nnodes (int): number of nodes.
        cuda_visible_devices (str): cuda visible devices.
    """

    def __init__(
        self,
        config: RolloutConfig,
        model_config: HFModelConfig,
        rollout_mode: RolloutMode,
        workers: list[ActorHandle],
        replica_rank: int,
        node_rank: int,
        nnodes: int,
        cuda_visible_devices: str,
        base_gpu_id: int,
    ):
        print(f"SGLang http server: {rollout_mode=}, {replica_rank=}, {node_rank=}, {nnodes=}, {cuda_visible_devices=}")
        os.environ[visible_devices_keyword] = cuda_visible_devices

        self.config: RolloutConfig = omega_conf_to_dataclass(config)
        self.model_config: HFModelConfig = omega_conf_to_dataclass(model_config, dataclass_type=HFModelConfig)
        max_position_embeddings = get_max_position_embeddings(self.model_config.hf_config)
        if self.config.max_model_len is None:
            self.config.max_model_len = max_position_embeddings
        else:
            if self.config.max_model_len > max_position_embeddings:
                raise ValueError(
                    f"max_model_len ({self.config.max_model_len}) should be less than or equal to "
                    f"max_position_embeddings ({max_position_embeddings})"
                )
        self.rollout_mode = rollout_mode
        self.workers = workers

        self.replica_rank = replica_rank
        self.node_rank = node_rank
        self.nnodes = nnodes
        self.base_gpu_id = base_gpu_id

        if self.rollout_mode != RolloutMode.HYBRID and self.config.load_format == "dummy":
            logger.warning(f"rollout mode is {self.rollout_mode}, load_format is dummy, set to auto")
            self.config.load_format = "auto"

        # ===== streaming state (server-side) =====
        self._stream_queues: dict[str, asyncio.Queue] = {}
        self._stream_tasks: dict[str, asyncio.Task] = {}
        self._stream_meta: dict[str, dict] = {}

        # used for http server
        self._server_address = ray.util.get_node_ip_address().strip("[]")
        self._server_port = None

        # used for controlling sglang server profiler
        profiler_config = self.config.profiler
        tool_config = None
        if profiler_config is not None:
            if profiler_config.tool in ["torch", "npu"]:
                tool_config = omega_conf_to_dataclass((profiler_config.tool_config or {}).get(profiler_config.tool))
            else:
                logger.warning(f"agent loop only support torch and npu profiler, got {profiler_config.tool}")
                profiler_config = None
        self.profiler_controller = DistProfiler(self.replica_rank, config=profiler_config, tool_config=tool_config)

        # used for NCCL process group
        if self.node_rank == 0:
            self._master_address = self._server_address
            self._master_port, self._master_sock = get_free_port(self._server_address)
            logger.info(
                f"SGLangHttpServer, replica_rank: {self.replica_rank}, "
                f"master address: {self._master_address}, port: {self._master_port}"
            )
        else:
            self._master_address = None
            self._master_port = None

    def get_master_address(self):
        """Get master address and port for init NCCL process group."""
        return self._master_address, self._master_port

    def get_server_address(self):
        """Get http server address and port."""
        assert self._server_port is not None, "http server is not launched, port is None"
        return self._server_address, self._server_port

    async def launch_server(self, master_address: str = None, master_port: int = None):
        if self.node_rank != 0:
            assert master_address and master_port, "non-master node should provide master address and port"
            self._master_address = master_address
            self._master_port = master_port

        engine_kwargs = self.config.get("engine_kwargs", {}).get("sglang", {}) or {}
        attention_backend = engine_kwargs.pop("attention_backend", None)
        quantization = self.config.get("quantization", None)
        if quantization is not None:
            if quantization == "fp8":
                assert version.parse(_sglang_version) >= version.parse("0.5.5"), (
                    "sglang>=0.5.5 is required for FP8 quantization"
                )
                FP8_BLOCK_QUANT_KWARGS = {
                    "activation_scheme": "dynamic",
                    "fmt": "e4m3",
                    "quant_method": "fp8",
                    "weight_block_size": [128, 128],
                }
                fp8_block_quant_kwargs = dict(FP8_BLOCK_QUANT_KWARGS)
            else:
                raise ValueError(f"Currently only support fp8 quantization, got: {quantization}")
        dist_init_addr = (
            f"[{self._master_address}]:{self._master_port}"
            if is_valid_ipv6_address(self._master_address)
            else f"{self._master_address}:{self._master_port}"
        )
        infer_tp = self.config.tensor_model_parallel_size * self.config.data_parallel_size
        args = {
            "model_path": self.model_config.local_path,
            "dtype": self.config.dtype,
            "mem_fraction_static": self.config.gpu_memory_utilization,
            "disable_cuda_graph": self.config.enforce_eager,
            "enable_memory_saver": True,
            "base_gpu_id": self.base_gpu_id,
            "gpu_id_step": 1,
            "tp_size": infer_tp,
            "dp_size": self.config.data_parallel_size,
            "ep_size": self.config.expert_parallel_size,
            "node_rank": self.node_rank,
            "load_format": self.config.load_format,
            "dist_init_addr": dist_init_addr,
            "nnodes": self.nnodes,
            "trust_remote_code": self.model_config.trust_remote_code,
            "max_running_requests": self.config.get("max_num_seqs", None),
            "log_level": "error",
            "mm_attention_backend": "fa3",
            "attention_backend": attention_backend if attention_backend is not None else "fa3",
            "skip_tokenizer_init": self.config.skip_tokenizer_init,
            "skip_server_warmup": True,
            "quantization": quantization,
            "json_model_override_args": json.dumps({"quantization_config": fp8_block_quant_kwargs})
            if quantization == "fp8"
            else json.dumps({}),
            **engine_kwargs,
        }

        if self.config.prometheus.enable:
            if self.config.prometheus.served_model_name:
                # Extract model name from path if it's a full path
                served_model_name = self.config.prometheus.served_model_name
                if "/" in served_model_name:
                    # If it's a full path, extract the last part as model name
                    served_model_name = served_model_name.split("/")[-1]
                args["served_model_name"] = served_model_name

            # start sglang metrics
            args["enable_metrics"] = True

        # enable_weights_cpu_backup is supported in sglang>=0.5.3
        if "enable_weights_cpu_backup" in [f.name for f in dataclasses.fields(ServerArgs)]:
            enable_weights_cpu_backup = True if self.rollout_mode == RolloutMode.COLOCATED else False
            args["enable_weights_cpu_backup"] = enable_weights_cpu_backup

        if self.config.enable_rollout_routing_replay:
            args.update({"enable_return_routed_experts": True})

        # mtp
        if self.config.mtp.enable and self.config.mtp.enable_rollout:
            # Enable weights CPU backup for sglang >= 0.5.6
            if _sglang_version < "0.5.6":
                raise ValueError(f"sglang version {_sglang_version} is not supported for MTP rollout")

            args["speculative_algorithm"] = self.config.mtp.speculative_algorithm
            args["speculative_num_steps"] = self.config.mtp.speculative_num_steps
            args["speculative_eagle_topk"] = self.config.mtp.speculative_eagle_topk
            args["speculative_num_draft_tokens"] = self.config.mtp.speculative_num_draft_tokens

            args["enable_weights_cpu_backup"] = True
            args["enable_draft_weights_cpu_backup"] = True

        # NOTE: We can't directly call SGLang's launch_server since it's not an async function.
        # https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/entrypoints/http_server.py
        sglang.srt.entrypoints.engine._set_envs_and_config = _set_envs_and_config
        os.environ["SGLANG_BLOCK_NONZERO_RANK_CHILDREN"] = "0"
        server_args = ServerArgs(**args)
        import inspect
        _launch_sig = inspect.signature(_launch_subprocesses)
        if "init_tokenizer_manager_func" in _launch_sig.parameters:
            self.tokenizer_manager, self.template_manager, self.scheduler_info, *_ = _launch_subprocesses(
                server_args=server_args,
                init_tokenizer_manager_func=sglang.srt.entrypoints.engine.init_tokenizer_manager,
                run_scheduler_process_func=sglang.srt.entrypoints.engine.run_scheduler_process,
                run_detokenizer_process_func=sglang.srt.entrypoints.engine.run_detokenizer_process,
            )
        else:
            self.tokenizer_manager, self.template_manager, self.scheduler_info, *_ = _launch_subprocesses(
                server_args=server_args
            )

        # In multi-node cases, non-zero rank nodes should not launch http server.
        if self.node_rank > 0:
            return

        set_global_state(
            _GlobalState(
                tokenizer_manager=self.tokenizer_manager,
                template_manager=self.template_manager,
                scheduler_info=self.scheduler_info,
            )
        )
        app.is_single_tokenizer_mode = True

        # Set warmup_thread_{kw}args to avoid AttributeError in lifespan function
        app.server_args = server_args
        app.warmup_thread_kwargs = {"server_args": server_args}
        app.warmup_thread_args = (server_args, None, None)

        # Manually add Prometheus middleware before starting server
        # This ensures /metrics endpoint is available immediately
        if server_args.enable_metrics:
            from sglang.srt.utils.common import add_prometheus_middleware

            add_prometheus_middleware(app)

        self._server_port, self._server_task = await run_unvicorn(app, server_args, self._server_address)
        self.tokenizer_manager.server_status = ServerStatus.Up

    async def wake_up(self):
        if self.node_rank != 0:
            return

        if self.rollout_mode == RolloutMode.HYBRID:
            # In hybrid mode, rollout is wake up in `update_weights`
            raise ValueError(f"wake_up not support rollout_mode {self.rollout_mode}")
        elif self.rollout_mode == RolloutMode.COLOCATED:
            # Directly call engine to wake up without sync weights.
            obj = ResumeMemoryOccupationReqInput(tags=["kv_cache", "weights"])
            await self.tokenizer_manager.resume_memory_occupation(obj, None)
            await self.tokenizer_manager.flush_cache()
        elif self.rollout_mode == RolloutMode.STANDALONE:
            logger.info("skip wake_up in standalone mode")

    async def sleep(self):
        if self.node_rank != 0 or not self.config.free_cache_engine:
            return

        if self.rollout_mode == RolloutMode.HYBRID:
            obj = ReleaseMemoryOccupationReqInput(tags=["kv_cache", "weights"])
            await self.tokenizer_manager.release_memory_occupation(obj, None)
        elif self.rollout_mode == RolloutMode.COLOCATED:
            obj = ReleaseMemoryOccupationReqInput(tags=["kv_cache", "weights"])
            await self.tokenizer_manager.release_memory_occupation(obj, None)
        elif self.rollout_mode == RolloutMode.STANDALONE:
            logger.info("skip sleep in standalone mode")

    async def clear_kv_cache(self):
        obj = ReleaseMemoryOccupationReqInput(tags=["kv_cache"])
        await self.tokenizer_manager.release_memory_occupation(obj, None)

    async def generate(
        self,
        prompt_ids: torch.Tensor,
        sampling_params: dict[str, Any],
        request_id: str,
        image_data: Optional[list[Any]] = None,
        video_data: Optional[list[Any]] = None,
    ) -> TokenOutput:
        """Generate sequence with token-in-token-out."""
        # TODO(@wuxibin): switch to `/generate` http endpoint once multi-modal support ready.
        max_possible_tokens = self.config.max_model_len - len(prompt_ids)

        if max_possible_tokens < 0:
            raise ValueError(
                f"Prompt length ({len(prompt_ids)}) exceeds the model's maximum context length "
                f"({self.config.max_model_len})."
            )

        if "max_new_tokens" in sampling_params:
            max_new_tokens = sampling_params.pop("max_new_tokens")
        elif "max_tokens" in sampling_params:
            # support vllm-style 'max_tokens' param
            max_new_tokens = sampling_params.pop("max_tokens")
        else:
            max_new_tokens = self.config.response_length + self.config.prompt_length - len(prompt_ids)

        # Clamp max_new_tokens to the valid range [0, max_possible_tokens]
        max_new_tokens = max(0, min(max_new_tokens, max_possible_tokens))

        assert max_new_tokens <= max_possible_tokens, (
            f"max_new_tokens {max_new_tokens} exceeds available context space {max_possible_tokens}"
        )
        sampling_params["max_new_tokens"] = max_new_tokens
        return_logprob = sampling_params.pop("logprobs", False)

        request = {
            "rid": request_id,
            "input_ids": prompt_ids,
            "sampling_params": sampling_params,
            "return_logprob": return_logprob,
            "image_data": image_data,
            # TODO: support video input for sglang
            # video_data=video_data,
        }

        if self.config.enable_rollout_routing_replay:
            request.update({"return_routed_experts": True})

        generate_request = GenerateReqInput(**request)

        output = await self.tokenizer_manager.generate_request(generate_request, None).__anext__()
        if return_logprob:
            output_token_logprobs = output["meta_info"]["output_token_logprobs"]
            log_probs, token_ids = zip(
                *[(log_prob, token_ids) for log_prob, token_ids, _ in output_token_logprobs], strict=True
            )
        else:
            token_ids = output["output_ids"]
            log_probs = None

        routed_experts = None
        if self.config.enable_rollout_routing_replay:
            if self.config.skip_tokenizer_init:
                routed_experts = output.get("meta_info", {}).get("routed_experts", None)
            else:
                from sglang.srt.layers.moe.routed_experts_capturer import extract_routed_experts_from_meta_info

                hf_config = self.model_config.hf_config
                if not hasattr(hf_config, "num_hidden_layers") or not hasattr(hf_config, "num_experts_per_tok"):
                    raise AttributeError(
                        "enable_rollout_routing_replay is set, but hf_config is missing "
                        "'num_hidden_layers' or 'num_experts_per_tok'. This feature requires an MoE model "
                        "configuration that defines these attributes."
                    )
                routed_experts = extract_routed_experts_from_meta_info(output).reshape(
                    -1, hf_config.num_hidden_layers, hf_config.num_experts_per_tok
                )

        return TokenOutput(token_ids=token_ids, log_probs=log_probs, routed_experts=routed_experts)

    async def start_profile(self, **kwargs):
        if (
            self.profiler_controller.check_enable()
            and self.profiler_controller.check_this_rank()
            and self.profiler_controller.is_discrete_mode()
        ):
            profile_args, self._auto_stop_profiling = SGLangProfilerArgsBuilder(
                profiler_controller=self.profiler_controller, rollout_config=self.config, replica_rank=self.replica_rank
            ).build_profile_args(**kwargs)
            await self.tokenizer_manager.start_profile(**profile_args)

    async def stop_profile(self):
        if (
            self.profiler_controller.check_enable()
            and self.profiler_controller.check_this_rank()
            and self.profiler_controller.is_discrete_mode()
            and not self._auto_stop_profiling
        ):
            await self.tokenizer_manager.stop_profile()

    # =========================
    # Abort helpers
    # =========================

    async def abort_all_requests(self, reset_prefix_cache: bool = True) -> dict[str, Any]:
        try:
            self.tokenizer_manager.abort_request(abort_all=True)
            if reset_prefix_cache:
                await self.tokenizer_manager.flush_cache()
            return {"aborted_count": -1, "request_ids": []}
        except Exception as e:
            logger.error(f"Error aborting all requests: {e}")
            return {"aborted_count": 0, "request_ids": [], "error": str(e)}

    async def abort_request(self, request_id: str, reset_prefix_cache: bool = True) -> dict[str, Any]:
        try:
            self.tokenizer_manager.abort_request(rid=request_id)
            if reset_prefix_cache:
                await self.tokenizer_manager.flush_cache()
            return {"aborted": True, "request_id": request_id}
        except Exception as e:
            logger.error(f"Error aborting request {request_id}: {e}")
            return {"aborted": False, "request_id": request_id, "error": str(e)}

    async def wait_for_requests_to_drain(self):
        pass

    # =========================
    # Streaming RPC
    # =========================

    async def start_generate_stream(
        self,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        request_id: str,
        image_data: Optional[list[Any]] = None,
        video_data: Optional[list[Any]] = None,
        priority: int = 0,
    ) -> dict[str, Any]:
        if self.node_rank != 0:
            raise RuntimeError("start_generate_stream should only be called on node_rank 0")

        actual_prompt_len = len(prompt_ids)
        max_possible_tokens = self.config.max_model_len - actual_prompt_len
        if max_possible_tokens < 0:
            raise ValueError(
                f"Prompt length ({actual_prompt_len}) exceeds max_model_len ({self.config.max_model_len})."
            )

        sp = dict(sampling_params)
        if "max_tokens" in sp:
            max_tokens = sp.pop("max_tokens")
        elif "max_new_tokens" in sp:
            max_tokens = sp.pop("max_new_tokens")
        else:
            max_tokens = min(self.config.response_length, max_possible_tokens)
        max_tokens = max(0, max_tokens)

        return_logprob = sp.pop("logprobs", False)
        sp["max_new_tokens"] = max_tokens

        request = {
            "rid": request_id,
            "input_ids": prompt_ids,
            "sampling_params": sp,
            "return_logprob": return_logprob,
            "stream": True,
            "image_data": image_data,
        }

        if self.config.enable_rollout_routing_replay:
            request["return_routed_experts"] = True

        generate_request = GenerateReqInput(**request)

        handle = uuid4().hex
        q: asyncio.Queue = asyncio.Queue(maxsize=256)
        self._stream_queues[handle] = q
        self._stream_meta[handle] = {
            "request_id": request_id,
            "done": False,
            "finish_reason": None,
            "stop_reason": None,
            "num_preempted": 0,
            "total_tokens": 0,
        }

        async def _runner():
            prev_len = 0
            is_first_delta = True
            try:
                gen = self.tokenizer_manager.generate_request(generate_request, None)
                async for out in gen:
                    meta_info = out.get("meta_info", {})
                    token_ids = out.get("output_ids", [])
                    if isinstance(token_ids, torch.Tensor):
                        token_ids = token_ids.tolist()

                    cur_len = len(token_ids)
                    if cur_len > prev_len:
                        delta = token_ids[prev_len:cur_len]
                        prev_len = cur_len
                        evt = {"type": "delta", "token_ids": delta, "total_tokens": cur_len}
                        if is_first_delta:
                            is_first_delta = False
                            evt["meta_info"] = meta_info
                        await q.put(evt)

                finish_reason = "stop"
                stop_reason = "completed"

                self._stream_meta[handle].update({
                    "done": True,
                    "finish_reason": finish_reason,
                    "stop_reason": stop_reason,
                    "total_tokens": prev_len,
                })
                await q.put({
                    "type": "done",
                    "finish_reason": finish_reason,
                    "stop_reason": stop_reason,
                    "total_tokens": prev_len,
                    "meta_info": meta_info,
                })

            except Exception as e:
                self._stream_meta[handle].update({
                    "done": True,
                    "finish_reason": "error",
                    "stop_reason": "error",
                    "total_tokens": prev_len,
                })
                await q.put({"type": "error", "error": str(e)})

        self._stream_tasks[handle] = asyncio.create_task(_runner())

        return {
            "handle": handle,
            "request_id": request_id,
            "prompt_len": actual_prompt_len,
            "max_tokens": max_tokens,
        }

    async def poll_generate_stream(self, handle: str, timeout_s: float = 0.0) -> Optional[dict[str, Any]]:
        q = self._stream_queues.get(handle)
        if q is None:
            return {"type": "error", "error": f"unknown handle {handle}"}
        try:
            if timeout_s and timeout_s > 0:
                return await asyncio.wait_for(q.get(), timeout=timeout_s)
            return q.get_nowait()
        except asyncio.TimeoutError:
            return None
        except asyncio.QueueEmpty:
            return None

    async def poll_generate_stream_many(self, handles: list[str], timeout_ms: int = 5) -> list[dict[str, Any]]:
        timeout_s = max(0.0, float(timeout_ms) / 1000.0)

        async def _poll_one(h, t):
            msg = await self.poll_generate_stream(h, timeout_s=t)
            return (h, msg)

        tasks = []
        for i, h in enumerate(handles):
            tasks.append(_poll_one(h, timeout_s if i == 0 else 0.0))

        out = await asyncio.gather(*tasks, return_exceptions=False)

        results = []
        for h, msg in out:
            if msg is not None:
                results.append({"handle": h, **msg})
        return results

    async def finalize_generate_stream(self, handle: str) -> dict[str, Any]:
        t = self._stream_tasks.get(handle)
        if t is None:
            return {"ok": False, "error": f"unknown handle {handle}"}
        try:
            await t
        except Exception:
            pass
        meta = self._stream_meta.get(handle, {})
        self._stream_queues.pop(handle, None)
        self._stream_tasks.pop(handle, None)
        self._stream_meta.pop(handle, None)
        return {"ok": True, **meta}

    async def cancel_generate_stream(self, handle: str, reset_prefix_cache: bool = True) -> dict[str, Any]:
        meta = self._stream_meta.get(handle)
        if meta is None:
            return {"ok": False, "error": f"unknown handle {handle}"}

        req_id = meta.get("request_id")
        if req_id is not None:
            await self.abort_request(req_id, reset_prefix_cache=reset_prefix_cache)

        t = self._stream_tasks.get(handle)
        if t is not None and not t.done():
            t.cancel()

        self._stream_meta[handle].update({"done": True, "finish_reason": "cancel", "stop_reason": "cancel"})

        q = self._stream_queues.get(handle)
        if q is not None:
            try:
                await q.put({"type": "done", "finish_reason": "cancel", "stop_reason": "cancel"})
            except Exception:
                pass

        return {"ok": True, "handle": handle}


_rollout_worker_actor_cls = ray.remote(ServerAdapter)


class SGLangReplica(RolloutReplica):
    def __init__(
        self,
        replica_rank: int,
        config: RolloutConfig,
        model_config: HFModelConfig,
        gpus_per_node: int = 8,
        is_reward_model: bool = False,
    ):
        super().__init__(replica_rank, config, model_config, gpus_per_node, is_reward_model)
        self.server_class = ray.remote(SGLangHttpServer)

    def get_ray_class_with_init_args(self) -> RayClassWithInitArgs:
        """Get rollout worker actor class for colocated and standalone mode."""
        worker_dict_cls = RayClassWithInitArgs(
            cls=_rollout_worker_actor_cls,
            config=self.config,
            model_config=self.model_config,
            device_mesh=None,
        )
        return worker_dict_cls

    async def launch_servers(self):
        """Launch http server in each node."""
        assert len(self.workers) == self.world_size, (
            f"worker number {len(self.workers)} not equal to world size {self.world_size}"
        )

        # get (node_id, CUDA_VISIBLE_DEVICES) of all workers
        worker_infos = await asyncio.gather(
            *[
                worker.__ray_call__.remote(
                    lambda self: (ray.get_runtime_context().get_node_id(), os.environ[visible_devices_keyword])
                )
                for worker in self.workers
            ]
        )
        worker_cuda_visible_devices = [worker_info[1] for worker_info in worker_infos]
        worker_node_ids = [worker_info[0] for worker_info in worker_infos]
        base_gpu_id = 0
        infer_tp = self.config.tensor_model_parallel_size * self.config.data_parallel_size
        replica_world_size = infer_tp * self.config.pipeline_model_parallel_size
        if os.environ.get(f"RAY_EXPERIMENTAL_NOSET_{visible_devices_keyword}", None):
            logger.warning(f"RAY_EXPERIMENTAL_NOSET_{visible_devices_keyword} is set True!")
            base_gpu_id = (0 + self.replica_rank * replica_world_size) % self.gpus_per_node
        # create server actor in each node with node affinity and cuda visible devices
        for node_rank in range(self.nnodes):
            workers = self.workers[
                node_rank * self.gpus_per_replica_node : (node_rank + 1) * self.gpus_per_replica_node
            ]
            node_cuda_visible_devices_set = worker_cuda_visible_devices[
                node_rank * self.gpus_per_replica_node : (node_rank + 1) * self.gpus_per_replica_node
            ]
            node_cuda_visible_devices = ",".join(
                map(
                    str,
                    sorted(
                        set(
                            int(device)
                            for worker_devices_set in node_cuda_visible_devices_set
                            for device in worker_devices_set.split(",")
                            if device.strip()
                        )
                    ),
                )
            )

            node_id = worker_node_ids[node_rank * self.gpus_per_replica_node]
            suffix = getattr(self.config, "server_name_suffix", None) or "default"
            name = (
                f"sglang_server_{self.replica_rank}_{node_rank}_{suffix}"
                if not self.is_reward_model
                else f"sglang_server_reward_{self.replica_rank}_{node_rank}_{suffix}"
            )
            env_vars = {f"RAY_EXPERIMENTAL_NOSET_{visible_devices_keyword}": "1"}
            mps_pct = getattr(self.config, "mps_active_thread_percentage", 0)
            if mps_pct > 0:
                env_vars.update({
                    "CUDA_MPS_PIPE_DIRECTORY": "/tmp/nvidia-mps",
                    "CUDA_MPS_LOG_DIRECTORY": "/tmp/nvidia-mps-log",
                    "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE": str(mps_pct),
                })
            server = self.server_class.options(
                scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                    node_id=node_id,
                    soft=False,
                ),
                runtime_env={"env_vars": env_vars},
                name=name,
            ).remote(
                config=self.config,
                model_config=self.model_config,
                rollout_mode=self.rollout_mode,
                workers=workers,
                replica_rank=self.replica_rank,
                node_rank=node_rank,
                nnodes=self.nnodes,
                cuda_visible_devices=node_cuda_visible_devices,
                base_gpu_id=base_gpu_id,
            )
            self.servers.append(server)

        # launch http server in each node
        master_address, master_port = await self.servers[0].get_master_address.remote()
        await asyncio.gather(
            *[
                server.launch_server.remote(master_address=master_address, master_port=master_port)
                for server in self.servers
            ]
        )

        # get http server address from first server
        server_address, server_port = await self.servers[0].get_server_address.remote()
        self._server_handle = self.servers[0]
        self._server_address = (
            f"[{server_address}]:{server_port}"
            if is_valid_ipv6_address(server_address)
            else f"{server_address}:{server_port}"
        )

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
import argparse
import asyncio
import inspect
import json
import logging
import os
from pprint import pprint
from typing import Any, Callable, Optional

import numpy as np
import ray
import vllm.entrypoints.cli.serve
from packaging import version
from ray.actor import ActorHandle
from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.entrypoints.cli.serve import run_headless
from vllm.entrypoints.openai.api_server import build_app, init_app_state
from vllm.inputs import TokensPrompt
from vllm.lora.request import LoRARequest
from vllm.outputs import RequestOutput
from vllm.usage.usage_lib import UsageContext
from vllm.v1.engine.async_llm import AsyncLLM

from verl.single_controller.ray import RayClassWithInitArgs
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.device import get_resource_name, get_visible_devices_keyword
from verl.utils.net_utils import get_free_port, is_valid_ipv6_address
from verl.utils.profiler.profile import DistProfiler
from verl.utils.vllm.vllm_fp8_utils import apply_vllm_fp8_patches
from verl.workers.config import HFModelConfig, RolloutConfig
from verl.workers.rollout.replica import RolloutMode, RolloutReplica, TokenOutput
from verl.workers.rollout.utils import get_max_position_embeddings, run_unvicorn
from verl.workers.rollout.vllm_rollout import ServerAdapter
from verl.workers.rollout.vllm_rollout.utils import (
    VLLM_LORA_INT_ID,
    VLLM_LORA_NAME,
    VLLM_LORA_PATH,
    SuppressSignalInThread,
    build_cli_args_from_config,
    get_vllm_max_lora_rank,
)

from uuid import uuid4
import heapq
import time

_VLLM_VERSION = version.parse(vllm.__version__)

if _VLLM_VERSION > version.parse("0.11.0"):
    from vllm.utils.argparse_utils import FlexibleArgumentParser

    if _VLLM_VERSION == version.parse("0.12.0"):
        from vllm.entrypoints.harmony_utils import get_encoding

    elif _VLLM_VERSION >= version.parse("0.13.0"):
        from vllm.entrypoints.openai.parser.harmony_utils import get_encoding

    else:
        get_encoding = None

    if get_encoding is not None and os.getenv("VERL_USE_GPT_OSS", "0") == "1":
        get_encoding()
else:
    from vllm.utils import FlexibleArgumentParser


logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)


class vLLMHttpServer:
    """vLLM http server in single node, this is equivalent to launch server with command line:
    ```
    vllm serve --tensor-parallel-size=8 ...
    ```
    """

    def __init__(
        self,
        config: RolloutConfig,
        model_config: HFModelConfig,
        rollout_mode: RolloutMode,
        workers: list[ActorHandle],
        replica_rank: int,
        node_rank: int,
        gpus_per_node: int,
        nnodes: int,
        cuda_visible_devices: str,
    ):
        """
        Args:
            config (RolloutConfig): full config.
            model_config (HFModelConfig): model config.
            rollout_mode (RolloutMode): rollout mode.
            replica_rank (int): replica rank, a replica may contain multiple nodes.
            node_rank (int): node rank.
            gpus_per_node (int): number of gpus per node.
            nnodes (int): number of nodes.
            cuda_visible_devices (str): cuda visible devices.
        """
        os.environ[get_visible_devices_keyword()] = cuda_visible_devices

        self.config: RolloutConfig = omega_conf_to_dataclass(config)
        self.model_config: HFModelConfig = omega_conf_to_dataclass(model_config, dataclass_type=HFModelConfig)
        if not self.config.max_model_len:
            self.config.max_model_len = get_max_position_embeddings(self.model_config.hf_config)
        self.rollout_mode = rollout_mode
        self.workers = workers
        self.debug_count = 0

        # ===== streaming state (server-side) =====
        self._stream_queues: dict[str, asyncio.Queue] = {}
        self._stream_tasks: dict[str, asyncio.Task] = {}
        self._stream_meta: dict[str, dict] = {}
        
        self.replica_rank = replica_rank
        self.node_rank = node_rank
        self.gpus_per_node = gpus_per_node
        self.nnodes = nnodes

        if self.rollout_mode != RolloutMode.HYBRID and self.config.load_format == "dummy":
            logger.warning(f"rollout mode is {self.rollout_mode}, load_format is dummy, set to auto")
            self.config.load_format = "auto"

        # used for http server
        self._server_address = ray.util.get_node_ip_address().strip("[]")
        self._server_port = None

        # used for controlling vllm server profiler
        profiler_config = self.config.profiler
        tool_config = None
        if profiler_config is not None:
            if profiler_config.tool in ["torch", "npu"]:
                tool_config = omega_conf_to_dataclass((profiler_config.tool_config or {}).get(profiler_config.tool))
            else:
                logger.warning(f"agent loop only support torch and npu profiler, got {profiler_config.tool}")
                profiler_config = None
        self.profiler_controller = DistProfiler(self.replica_rank, config=profiler_config, tool_config=tool_config)
        self.server_profiler_dir = os.environ.pop("VLLM_TORCH_PROFILER_DIR", None)

        # used for data parallel: --data-parallel-address, --data-parallel-rpc-port
        if self.node_rank == 0:
            self._master_address = self._server_address
            # used for torch.distributed.init_process_group
            self._master_port, self._master_sock = get_free_port(self._server_address)
            # used for data parallel: --data-parallel-address, --data-parallel-rpc-port
            self._dp_rpc_port, self._dp_rpc_sock = get_free_port(self._server_address)
            self._dp_master_port, self._dp_master_sock = get_free_port(self._server_address)
        else:
            self._master_address = None
            self._master_port = None
            self._dp_rpc_port = None
            self._dp_master_port = None

        logger.info(
            f"vLLMHttpServer, replica_rank: {self.replica_rank}, node_rank: {self.node_rank}, "
            f"{get_visible_devices_keyword()}: {cuda_visible_devices}, "
            f"master_address: {self._master_address}, master_port: {self._master_port}, "
            f"data_parallel_rpc_port: {self._dp_rpc_port}, data_parallel_master_port: {self._dp_master_port}"
        )

    def get_master_address(self):
        """Get master address and port for data parallel.
        Returns:
            tuple: (master_address, master_port, dp_rpc_port)
        """
        return self._master_address, self._master_port, self._dp_rpc_port

    def get_server_address(self):
        """Get http server address and port."""
        assert self._server_port is not None, "http server is not launched, port is None"
        return self._server_address, self._server_port

    async def collective_rpc(
        self,
        method: str | Callable,
        timeout: float | None = None,
        args: tuple = (),
        kwargs: dict[str, Any] | None = None,
    ):
        return await self.engine.collective_rpc(
            method=method,
            timeout=timeout,
            args=args,
            kwargs=kwargs,
        )

    async def poll_generate_stream_many(self, handles, timeout_ms: int = 5):
        # timeout_ms: apply to the *whole* poll call, not per-handle
        timeout_s = max(0.0, float(timeout_ms) / 1000.0)

        async def _poll_one(h, t):
            msg = await self.poll_generate_stream(h, timeout_s=t)
            return (h, msg)

        # 给第一个 handle 一个 timeout，其它 handle 0 timeout（避免 N 倍等待）
        tasks = []
        for i, h in enumerate(handles):
            tasks.append(_poll_one(h, timeout_s if i == 0 else 0.0))

        out = await asyncio.gather(*tasks, return_exceptions=False)

        # 只返回有消息的，减少上层处理
        results = []
        for h, msg in out:
            if msg is not None:
                results.append({"handle": h, **msg})
        return results

    async def launch_server(self, master_address: str = None, master_port: int = None, dp_rpc_port: int = None):
        if self.node_rank != 0:
            assert master_address and master_port and dp_rpc_port, (
                "non-master node should provide master_address, master_port and dp_rpc_port"
            )
            self._master_address = master_address
            self._master_port = master_port
            self._dp_rpc_port = dp_rpc_port

        # 1. setup vllm serve cli args
        engine_kwargs = self.config.get("engine_kwargs", {}).get("vllm", {}) or {}
        engine_kwargs = {key: val for key, val in engine_kwargs.items() if val is not None}
        if self.config.get("limit_images", None):  # support for multi-image data
            engine_kwargs["limit_mm_per_prompt"] = {"image": self.config.get("limit_images")}
        if self.config.cudagraph_capture_sizes:
            engine_kwargs["cuda_graph_sizes"] = self.config.cudagraph_capture_sizes

        # Override default generation config from hugging face model config,
        # user can still override them by passing kwargs in each request.
        override_generation_config = dict(
            temperature=self.config.temperature,
            top_k=self.config.top_k,
            top_p=self.config.top_p,
            repetition_penalty=1.0,
            max_new_tokens=self.config.response_length,
        )
        logger.info(f"override_generation_config: {override_generation_config}")

        logger.info(f"enable_sleep_mode: {self.config.enable_sleep_mode}")
        if not self.config.enable_sleep_mode:
            from verl.utils.device import set_expandable_segments

            set_expandable_segments(True)

        quantization = self.config.quantization

        if quantization is not None:
            _SUPPORTED_QUANTIZATION = ["fp8", "torchao"]
            if quantization not in _SUPPORTED_QUANTIZATION:
                raise ValueError(f"Currently only support {_SUPPORTED_QUANTIZATION} quantization, got: {quantization}")

            if quantization == "fp8":
                FP8_BLOCK_QUANT_KWARGS = {
                    "activation_scheme": "dynamic",
                    "fmt": "e4m3",
                    "quant_method": "fp8",
                    "weight_block_size": [128, 128],
                }
                fp8_block_quant_kwargs = dict(FP8_BLOCK_QUANT_KWARGS)
                # Apply vllm fp8 patches
                # Will remove the patch after vllm support on-the-fly quant for rollout natively.
                apply_vllm_fp8_patches()
                # for subprocesses patching
                os.environ["VERL_VLLM_FP8_QUANT_ENABLED"] = "1"

        hf_overrides = {}
        if quantization is not None and self.config.quantization_config_file is not None:
            hf_overrides["quantization_config_file"] = self.config.quantization_config_file

        if quantization == "fp8":
            hf_overrides["quantization_config"] = fp8_block_quant_kwargs
        compilation_config = engine_kwargs.get("compilation_config", None)
        if compilation_config is None:
            compilation_config = json.dumps({"cudagraph_mode": "FULL_AND_PIECEWISE"})
        else:
            cudagraph_mode = compilation_config.get("cudagraph_mode", "FULL_AND_PIECEWISE")
            compilation_config = json.dumps({"cudagraph_mode": cudagraph_mode})
        args = {
            "dtype": self.config.dtype,
            "load_format": self.config.load_format,
            "skip_tokenizer_init": False,
            "distributed_executor_backend": "mp",
            "worker_extension_cls": "verl.workers.rollout.vllm_rollout.utils.vLLMColocateWorkerExtension",
            "trust_remote_code": self.model_config.trust_remote_code,
            "max_model_len": self.config.max_model_len,
            "max_num_seqs": self.config.max_num_seqs,
            "enable_chunked_prefill": self.config.enable_chunked_prefill,
            "max_num_batched_tokens": self.config.max_num_batched_tokens,
            "enable_prefix_caching": self.config.enable_prefix_caching,
            "enable_sleep_mode": self.config.enable_sleep_mode,
            "logprobs_mode": self.config.logprobs_mode,
            "enforce_eager": self.config.enforce_eager,
            "gpu_memory_utilization": self.config.gpu_memory_utilization,
            "disable_log_stats": self.config.disable_log_stats,
            "tensor_parallel_size": self.config.tensor_model_parallel_size,
            "seed": self.config.get("seed", 0),
            "override_generation_config": json.dumps(override_generation_config),
            "quantization": quantization,
            "hf_overrides": hf_overrides,
            "scheduling_policy": self.config.scheduling_policy,
            "compilation_config": compilation_config,
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

        # mtp
        if self.config.mtp.enable and self.config.mtp.enable_rollout:
            speculative_config = {
                "method": self.config.mtp.method,
                "num_speculative_tokens": self.config.mtp.num_speculative_tokens,
            }
            args["speculative_config"] = speculative_config

        if self.config.expert_parallel_size > 1:
            assert self.gpus_per_node % self.config.tensor_model_parallel_size == 0, (
                "gpus_per_node should be divisible by tensor_model_parallel_size"
            )
            data_parallel_size_local = self.gpus_per_node // self.config.tensor_model_parallel_size
            assert len(self.workers) == data_parallel_size_local * self.config.tensor_model_parallel_size, (
                f"num workers ({len(self.workers)}) should be equal to dp_size_local "
            )
            f"({data_parallel_size_local}) * tp_size ({self.config.tensor_model_parallel_size})"

            args.update(
                {
                    "enable_expert_parallel": self.config.expert_parallel_size > 1,
                    "data_parallel_size": self.config.data_parallel_size,
                    "data_parallel_size_local": data_parallel_size_local,
                    "data_parallel_start_rank": self.node_rank * data_parallel_size_local,
                    "data_parallel_address": self._master_address,
                    "data_parallel_rpc_port": self._dp_rpc_port,
                }
            )

        # used for torch.distributed.init_process_group
        if self.nnodes > 1:
            args.update(
                {
                    "master_addr": self._master_address,
                    "master_port": self._master_port,
                    "node_rank": self.node_rank,
                    "nnodes": self.nnodes,
                    "data_parallel_address": self._master_address,
                    "data_parallel_rpc_port": self._dp_rpc_port,
                }
            )

        # update lora-related args
        lora_rank = self.model_config.lora.get("rank", 0)
        megatron_lora = True
        if self.model_config.lora.get("merge", False):
            lora_rank = 0
        if lora_rank <= 0:
            megatron_lora = False
            lora_rank = self.model_config.lora_rank
        if lora_rank > 0:
            lora_args = {
                "enable_lora": True,
                "max_loras": 1,
                "max_lora_rank": get_vllm_max_lora_rank(lora_rank),
            }
            if megatron_lora:
                lora_args["fully_sharded_loras"] = True
            args.update(lora_args)

        if self.config.enable_rollout_routing_replay:
            args.update({"enable_return_routed_experts": True})

        server_args = ["serve", self.model_config.local_path] + build_cli_args_from_config(args)

        if self.replica_rank == 0:
            pprint(server_args)

        CMD_MODULES = [vllm.entrypoints.cli.serve]
        parser = FlexibleArgumentParser(description="vLLM CLI")
        subparsers = parser.add_subparsers(required=False, dest="subparser")
        cmds = {}
        for cmd_module in CMD_MODULES:
            new_cmds = cmd_module.cmd_init()
            for cmd in new_cmds:
                cmd.subparser_init(subparsers).set_defaults(dispatch_function=cmd.cmd)
                cmds[cmd.name] = cmd
        server_args = parser.parse_args(args=server_args)
        server_args.model = server_args.model_tag
        if server_args.subparser in cmds:
            cmds[server_args.subparser].validate(server_args)

        # 3. launch server
        if self.node_rank == 0:
            self._master_sock.close()
            await self.run_server(server_args)
        else:
            # TODO: avoid connect before master_sock close
            await asyncio.sleep(3)
            await self.run_headless(server_args)

    async def run_server(self, args: argparse.Namespace):
        engine_args = AsyncEngineArgs.from_cli_args(args)
        usage_context = UsageContext.OPENAI_API_SERVER
        vllm_config = engine_args.create_engine_config(usage_context=usage_context)
        vllm_config.parallel_config.data_parallel_master_port = self._dp_master_port

        fn_args = set(dict(inspect.signature(AsyncLLM.from_vllm_config).parameters).keys())
        kwargs = {}
        if "enable_log_requests" in fn_args:
            kwargs["enable_log_requests"] = engine_args.enable_log_requests
        if "disable_log_stats" in fn_args:
            kwargs["disable_log_stats"] = engine_args.disable_log_stats

        engine_client = AsyncLLM.from_vllm_config(vllm_config=vllm_config, usage_context=usage_context, **kwargs)

        # Don't keep the dummy data in memory
        await engine_client.reset_mm_cache()
        await engine_client.collective_rpc(
            method="monkey_patch_model", kwargs={"vocab_size": len(self.model_config.tokenizer)}
        )

        build_app_sig = inspect.signature(build_app)
        supported_tasks: tuple[Any, ...] = ()
        if "supported_tasks" in build_app_sig.parameters:
            supported_tasks = await engine_client.get_supported_tasks()
            app = build_app(args, supported_tasks)
        else:
            app = build_app(args)

        init_app_sig = inspect.signature(init_app_state)
        if "vllm_config" in init_app_sig.parameters:
            await init_app_state(engine_client, vllm_config, app.state, args)
        elif "supported_tasks" in init_app_sig.parameters:
            await init_app_state(engine_client, app.state, args, supported_tasks)
        else:
            await init_app_state(engine_client, app.state, args)
        if self.replica_rank == 0 and self.node_rank == 0:
            logger.info(f"Initializing a V1 LLM engine with config: {vllm_config}")

        self.engine = engine_client
        self._server_port, self._server_task = await run_unvicorn(app, args, self._server_address)

    async def run_headless(self, args: argparse.Namespace):
        """Run headless server in a separate thread."""

        def run_headless_wrapper():
            with SuppressSignalInThread():
                run_headless(args)

        def on_run_headless_done(future: asyncio.Future):
            try:
                exc = future.exception()
                if exc:
                    logger.exception(f"run_headless failed with exception: {exc}")
                else:
                    logger.warning("run_headless completed successfully, but it's not expected.")
            except Exception as e:
                logger.exception(f"get result from run_headless failed: {e}")
            finally:
                os._exit(1)

        self.task = asyncio.create_task(asyncio.to_thread(run_headless_wrapper))
        self.task.add_done_callback(on_run_headless_done)

    async def generate(
        self,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        request_id: str,
        image_data: Optional[list[Any]] = None,
        video_data: Optional[list[Any]] = None,
        priority: int = 0,
    ) -> TokenOutput:
        """针对 Workload 测试优化后的生成逻辑"""
        
        # 1. 第一步：先进行多模态 Token 去重（避免长度计算虚高）
        # 如果是纯文本模型且没有对应的处理器，该函数通常会安全返回原数据
        prompt_ids = _qwen2_5_vl_dedup_image_tokens(prompt_ids, self.model_config.processor)
        
        # 2. 计算物理剩余空间
        actual_prompt_len = len(prompt_ids)
        max_possible_tokens = self.config.max_model_len - actual_prompt_len
        
        if max_possible_tokens < 0:
            raise ValueError(
                f"Prompt length ({actual_prompt_len}) exceeds the model's maximum context length "
                f"({self.config.max_model_len})."
            )

        # 3. 确定 max_tokens (新的生成长度预算)
        if "max_tokens" in sampling_params:
            max_tokens = sampling_params.pop("max_tokens")
        elif "max_new_tokens" in sampling_params:
            max_tokens = sampling_params.pop("max_new_tokens")
        else:
            # 不再使用 (response_length + prompt_length - len) 这种会产生截断堆积的逻辑
            # 尝试满足配置的 response_length，但不能超过物理极限
            max_tokens = min(self.config.response_length, max_possible_tokens)

        # 确保 max_tokens 不为负数
        max_tokens = max(0, max_tokens)

        # 4. 封装采样参数
        sampling_params["logprobs"] = 0 if sampling_params.pop("logprobs", False) else None
        sampling_params.setdefault("repetition_penalty", self.config.get("repetition_penalty", 1.0))
        sampling_params = SamplingParams(max_tokens=max_tokens, **sampling_params)

        # 5. 准备 Prompt 数据
        multi_modal_data = {}
        if image_data is not None:
            multi_modal_data["image"] = image_data
        if video_data is not None:
            multi_modal_data["video"] = video_data

        prompt = TokensPrompt(prompt_token_ids=prompt_ids, multi_modal_data=multi_modal_data)

        # 6. 处理 LoRA 请求
        lora_request = None
        if self.model_config.lora_rank > 0:
            lora_loaded = VLLM_LORA_INT_ID in await self.engine.list_loras()
            if lora_loaded:
                lora_request = LoRARequest(
                    lora_name=VLLM_LORA_NAME, lora_int_id=VLLM_LORA_INT_ID, lora_path=VLLM_LORA_PATH
                )

        # 7. 调用 vLLM 引擎生成
        # print(f"DEBUG: sampling_params.n = {sampling_params.n}")
        generator = self.engine.generate(
            prompt=prompt,
            sampling_params=sampling_params,
            request_id=request_id,
            lora_request=lora_request,
            priority=priority,
        )

        # i = 0
        # async for output in generator:
        #     i += 1
        #     if i <= 3:
        #         print(f"[dbg] yield #{i} request_id={request_id} out_len={len(output.outputs[0].token_ids)}", flush=True)
        #     final_res = output

        # print(f"prompt length: {len(prompt)} | token id: {len(token_ids)}")
        # 8 * 512 prompt length, print output token length 
        # 打印final_res.output 的structure

        final_res: Optional[RequestOutput] = None
        async for output in generator:
            final_res = output
        assert final_res is not None

        # 8. 整理结果与日志打印
        token_ids = final_res.outputs[0].token_ids
        finish_reason = final_res.outputs[0].finish_reason
        # ========== dayin =================
        # 1. 从 engine 或 model_config 中获取 tokenizer
        # vLLM v1 通常可以通过 self.engine.get_tokenizer() 获取
        # tokenizer = await self.engine.get_tokenizer()
        
        # # 2. 将 token_ids 解码为字符串
        # decoded_text = tokenizer.decode(token_ids, skip_special_tokens=False)
        
        # # 3. 打印输出
        # print(f"DEBUG_DECODED_TEXT: {decoded_text}")
        # 如果你想看具体的 token 列表（防止合并）
        # decoded_tokens_list = [tokenizer.decode([tid]) for tid in token_ids]
        # print(f"DEBUG_TOKEN_LIST: {decoded_tokens_list}")
        
        #======================================
        # ========== 限制频率的调试打印 ==========
        self.debug_count += 0
        # 比如每 100 个 request 打印一次，或者每轮迭代的前 3 个打印
        # if self.replica_rank == 0 and self.debug_count % 100 == 0: 
        #     try:
        #         tokenizer = await self.engine.get_tokenizer()
        #         decoded_prompt = tokenizer.decode(prompt_ids, skip_special_tokens=False)
                
        #         print(f"\n@@@ [DEBUG_GROUP_SAMPLE] Count: {self.debug_count} | Request_ID: {request_id} @@@")
        #         print(f"PROMPT: {decoded_prompt.replace('\n', '\\n')[:500]}...") # 打印前500字
        #         print(f"\n[final_res.outputs] Count: {len(final_res.outputs)}")
        #         for i, output in enumerate(final_res.outputs):
        #             curr_text = tokenizer.decode(output.token_ids, skip_special_tokens=False)
        #             print(f"DECODING_{i}: {curr_text.replace('\n', '\\n')}")
                
        #         print(f"@@@ [DEBUG_GROUP_END] @@@\n", flush=True)
        #     except Exception as e:
        #         print(f"Debug print failed: {e}")
        # # =======================================
        # print(f"prompt length: {len(prompt_ids)} | generated tokens: {len(token_ids)}")
        # print(f"WORKLOAD_SAMPLE_LENGTH: {len(token_ids)} | STOP_REASON: {finish_reason}")
        # import pprint
        # print("-" * 20 + " vLLM Output Structure " + "-" * 20)
        # # 打印第一个输出（通常 n=1）的所有属性
        # for i, output in enumerate(final_res.outputs):
        #     print(f"Output index: {i}")
        #     pprint.pprint(vars(output))
        # print("-" * 60)

        log_probs = None
        if sampling_params.logprobs is not None:
            log_probs = [logprobs[token_ids[i]].logprob for i, logprobs in enumerate(final_res.outputs[0].logprobs)]

        routed_experts = None
        if self.config.enable_rollout_routing_replay:
            routed_experts = final_res.outputs[0].routed_experts

        # 确定停止状态
        if finish_reason == "abort":
            stop_reason = "aborted"
        elif finish_reason in ("stop", "length"):
            stop_reason = "completed"
        else:
            stop_reason = finish_reason


        num_preempted = 0
        if hasattr(final_res, "preempted"):
            num_preempted = final_res.preempted

        return TokenOutput(
            token_ids=token_ids,
            log_probs=log_probs,
            routed_experts=routed_experts,
            stop_reason=stop_reason,
            num_preempted=num_preempted,
        )

    async def wake_up(self):
        if self.node_rank != 0:
            return

        if self.rollout_mode == RolloutMode.HYBRID:
            # In hybrid mode, rollout is wake up in `update_weights`
            raise ValueError(f"wake_up not support rollout_mode {self.rollout_mode}")
        elif self.rollout_mode == RolloutMode.COLOCATED:
            # Directly call engine to wake up without sync weights.
            await self.engine.wake_up(tags=["kv_cache", "weights"])
            await self.engine.reset_prefix_cache()
        elif self.rollout_mode == RolloutMode.STANDALONE:
            logger.info("skip wake_up in standalone mode")

    async def sleep(self):
        if self.node_rank != 0 or not self.config.free_cache_engine:
            return

        if self.rollout_mode == RolloutMode.HYBRID:
            # Don't use engine.sleep(level=2) here
            await self.engine.collective_rpc("sleep", kwargs={"level": 2})
        elif self.rollout_mode == RolloutMode.COLOCATED:
            await self.engine.sleep(level=1)
        elif self.rollout_mode == RolloutMode.STANDALONE:
            logger.info("skip sleep in standalone mode")

    async def start_profile(self, **kwargs):
        # TODO: Persist global_step to engine server-created file/path
        kwargs.pop("global_step")
        if (
            self.profiler_controller.check_enable()
            and self.profiler_controller.check_this_rank()
            and self.profiler_controller.is_discrete_mode()
            and self.server_profiler_dir
        ):
            await self.engine.start_profile(**kwargs)

    async def stop_profile(self):
        if (
            self.profiler_controller.check_enable()
            and self.profiler_controller.check_this_rank()
            and self.profiler_controller.is_discrete_mode()
            and self.server_profiler_dir
        ):
            await self.engine.stop_profile()

    async def clear_kv_cache(self):
        if self.node_rank == 0:
            await self.engine.reset_prefix_cache()

    async def wait_for_requests_to_drain(self):
        await self.engine.wait_for_requests_to_drain()

    async def abort_all_requests(self, reset_prefix_cache: bool = True) -> dict[str, Any]:
        """Abort all ongoing generation requests.

        Returns:
            dict[str, Any]: Dictionary containing:
                - aborted_count: Number of requests aborted
                - request_ids: List of aborted request IDs
        """
        try:
            # Take an atomic snapshot to avoid race conditions with the vLLM engine thread
            request_states_snapshot = list(self.engine.output_processor.request_states.items())
            request_ids = [req_id for req_id, _ in request_states_snapshot]

            if not request_ids:
                return {"aborted_count": 0, "request_ids": []}

            # For each request, create an abort output and put it to its queue
            # This allows the generator to receive the aborted result
            from vllm.v1.engine import FinishReason

            for _, req_state in request_states_snapshot:
                request_output = req_state.make_request_output(
                    [], pooling_output=None, finish_reason=FinishReason.ABORT, stop_reason=None
                )
                req_state.queue.put(request_output)

            # Abort requests in the output processor and engine core
            self.engine.output_processor.abort_requests(request_ids)
            await self.engine.engine_core.abort_requests_async(request_ids)

            # Try to reset prefix cache to ensure clean state
            if reset_prefix_cache:
                await self.clear_kv_cache()
                logger.info("Prefix cache reset after abort")

            logger.info(f"Aborted {len(request_ids)} requests: {request_ids}")
            return {"aborted_count": len(request_ids), "request_ids": request_ids}

        except Exception as e:
            logger.error(f"Error aborting requests: {e}")
            return {"aborted_count": 0, "request_ids": [], "error": str(e)}

    async def abort_request(self, request_id: str, reset_prefix_cache: bool = True) -> dict[str, Any]:
        """Abort a specific generation request.

        Args:
            request_id: The ID of the request to abort.

        Returns:
            dict[str, Any]: Dictionary containing abort result.
        """
        try:
            request_states = self.engine.output_processor.request_states
            req_state = request_states.get(request_id)

            if req_state is None:
                return {"aborted": False, "error": f"Request {request_id} not found"}

            # Create abort output and put it to the queue
            from vllm.v1.engine import FinishReason

            request_output = req_state.make_request_output(
                [], pooling_output=None, finish_reason=FinishReason.ABORT, stop_reason=None
            )
            req_state.queue.put(request_output)

            # Abort in output processor and engine core
            self.engine.output_processor.abort_requests([request_id])
            await self.engine.engine_core.abort_requests_async([request_id])

            # Try to reset prefix cache to ensure clean state
            if reset_prefix_cache:
                await self.clear_kv_cache()
                logger.info(f"Prefix cache reset after abort request {request_id}")

            logger.info(f"Aborted request: {request_id}")
            return {"aborted": True, "request_id": request_id}

        except Exception as e:
            logger.error(f"Error aborting request {request_id}: {e}")
            return {"aborted": False, "request_id": request_id, "error": str(e)}

    # =========================
    # Streaming RPC (single-handle)
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
        prompt_ids = _qwen2_5_vl_dedup_image_tokens(prompt_ids, self.model_config.processor)

        actual_prompt_len = len(prompt_ids)
        max_possible_tokens = self.config.max_model_len - actual_prompt_len
        if max_possible_tokens < 0:
            raise ValueError(
                f"Prompt length ({actual_prompt_len}) exceeds max_model_len ({self.config.max_model_len})."
            )

        sp = dict(sampling_params)  # do not mutate caller
        if "max_tokens" in sp:
            max_tokens = sp.pop("max_tokens")
        elif "max_new_tokens" in sp:
            max_tokens = sp.pop("max_new_tokens")
        else:
            max_tokens = min(self.config.response_length, max_possible_tokens)
        max_tokens = max(0, max_tokens)

        sp["logprobs"] = 0 if sp.pop("logprobs", False) else None
        sp.setdefault("repetition_penalty", self.config.get("repetition_penalty", 1.0))
        sampling = SamplingParams(max_tokens=max_tokens, **sp)

        multi_modal_data = {}
        if image_data is not None:
            multi_modal_data["image"] = image_data
        if video_data is not None:
            multi_modal_data["video"] = video_data
        prompt = TokensPrompt(prompt_token_ids=prompt_ids, multi_modal_data=multi_modal_data)

        lora_request = None
        if self.model_config.lora_rank > 0:
            lora_loaded = VLLM_LORA_INT_ID in await self.engine.list_loras()
            if lora_loaded:
                lora_request = LoRARequest(
                    lora_name=VLLM_LORA_NAME, lora_int_id=VLLM_LORA_INT_ID, lora_path=VLLM_LORA_PATH
                )

        handle = uuid4().hex
        q: asyncio.Queue = asyncio.Queue(maxsize=256)  # backpressure
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
            final_out: Optional[RequestOutput] = None
            try:
                gen = self.engine.generate(
                    prompt=prompt,
                    sampling_params=sampling,
                    request_id=request_id,
                    lora_request=lora_request,
                    priority=priority,
                )

                async for out in gen:
                    final_out = out
                    token_ids = out.outputs[0].token_ids
                    cur_len = len(token_ids)
                    if cur_len > prev_len:
                        delta = token_ids[prev_len:cur_len]
                        prev_len = cur_len
                        await q.put({"type": "delta", "token_ids": delta, "total_tokens": cur_len})

                if final_out is None:
                    self._stream_meta[handle].update(
                        {"done": True, "finish_reason": "abort", "stop_reason": "aborted", "total_tokens": prev_len}
                    )
                    await q.put({"type": "done", "finish_reason": "abort", "stop_reason": "aborted"})
                    return

                finish_reason = final_out.outputs[0].finish_reason
                if finish_reason == "abort":
                    stop_reason = "aborted"
                elif finish_reason in ("stop", "length"):
                    stop_reason = "completed"
                else:
                    stop_reason = finish_reason

                num_preempted = 0
                if hasattr(final_out, "preempted"):
                    num_preempted = final_out.preempted

                self._stream_meta[handle].update(
                    {
                        "done": True,
                        "finish_reason": finish_reason,
                        "stop_reason": stop_reason,
                        "num_preempted": num_preempted,
                        "total_tokens": prev_len,
                    }
                )
                await q.put(
                    {
                        "type": "done",
                        "finish_reason": finish_reason,
                        "stop_reason": stop_reason,
                        "num_preempted": num_preempted,
                        "total_tokens": prev_len,
                    }
                )

            except Exception as e:
                self._stream_meta[handle].update(
                    {"done": True, "finish_reason": "error", "stop_reason": "error", "total_tokens": prev_len}
                )
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

    async def finalize_generate_stream(self, handle: str) -> dict[str, Any]:
        t = self._stream_tasks.get(handle)
        if t is None:
            return {"ok": False, "error": f"unknown handle {handle}"}
        try:
            await t
        except Exception:
            pass
        meta = self._stream_meta.get(handle, {})
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


class vLLMReplica(RolloutReplica):
    def __init__(
        self,
        replica_rank: int,
        config: RolloutConfig,
        model_config: HFModelConfig,
        gpus_per_node: int = 8,
        is_reward_model: bool = False,
    ):
        super().__init__(replica_rank, config, model_config, gpus_per_node, is_reward_model)
        self.server_class = ray.remote(vLLMHttpServer)

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

        # NOTE: We always use MP Executor backend whether it's single-node or multi-node.
        # For multi-node without DP (e.g TP=16), need vllm>=0.11.1, https://github.com/vllm-project/vllm/pull/23691
        if self.config.data_parallel_size == 1 and self.nnodes > 1:
            assert _VLLM_VERSION >= version.parse("0.11.1"), (
                "For multi-node MP Executor, either (1) set data_parallel_size > 1 or (2) upgrade vLLM to >= 0.11.1"
            )

        # get (node_id, CUDA_VISIBLE_DEVICES) of all workers
        worker_infos = await asyncio.gather(
            *[
                worker.__ray_call__.remote(
                    lambda self: (
                        ray.get_runtime_context().get_node_id(),
                        ray.get_runtime_context().get_accelerator_ids()[get_resource_name()][0],
                    )
                )
                for worker in self.workers
            ]
        )
        worker_cuda_visible_devices = [worker_info[1] for worker_info in worker_infos]
        worker_node_ids = [worker_info[0] for worker_info in worker_infos]

        # create server actor in each node with node affinity and cuda visible devices
        nnodes, gpus_per_replica_node = self.nnodes, self.gpus_per_replica_node
        for node_rank in range(nnodes):
            workers = self.workers[node_rank * gpus_per_replica_node : (node_rank + 1) * gpus_per_replica_node]
            node_cuda_visible_devices = ",".join(
                worker_cuda_visible_devices[node_rank * gpus_per_replica_node : (node_rank + 1) * gpus_per_replica_node]
            )
            node_id = worker_node_ids[node_rank * gpus_per_replica_node]
            # name = (
            #     f"vllm_server_{self.replica_rank}_{node_rank}"
            #     if not self.is_reward_model
            #     else f"vllm_server_reward_{self.replica_rank}_{node_rank}"
            # )
            # Prefix to avoid name collisions across multiple AgentLoopManagers (e.g., dual-model A/B)
            suffix = getattr(self.config, "server_name_suffix", None) or "default"

            name = (
                f"vllm_server_{self.replica_rank}_{node_rank}_{suffix}"
                if not self.is_reward_model
                else f"vllm_server_reward_{self.replica_rank}_{node_rank}_{suffix}"
            )

            server = self.server_class.options(
                scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                    node_id=node_id,
                    soft=False,
                ),
                runtime_env={"env_vars": {
                    "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1",
                    "CUDA_MPS_PIPE_DIRECTORY": "/tmp/nvidia-mps",
                    "CUDA_MPS_LOG_DIRECTORY": "/tmp/nvidia-mps-log",
                    "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE": "50"
                }},
                name=name,
            ).remote(
                config=self.config,
                model_config=self.model_config,
                rollout_mode=self.rollout_mode,
                workers=workers,
                replica_rank=self.replica_rank,
                node_rank=node_rank,
                gpus_per_node=gpus_per_replica_node,
                nnodes=nnodes,
                cuda_visible_devices=node_cuda_visible_devices,
            )
            self.servers.append(server)

        # launch http server in each node
        master_address, master_port, dp_rpc_port = await self.servers[0].get_master_address.remote()
        await asyncio.gather(
            *[
                server.launch_server.remote(
                    master_address=master_address, master_port=master_port, dp_rpc_port=dp_rpc_port
                )
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

    async def sleep(self):
        """Sleep each rollout server."""
        # Drain DP engines for safe sleep.
        await self.servers[0].wait_for_requests_to_drain.remote()
        await asyncio.gather(*[server.sleep.remote() for server in self.servers])

    async def abort_all_requests(self) -> dict[str, Any]:
        """Abort all ongoing generation requests across all servers.

        Returns:
            dict[str, Any]: Combined abort results from all servers.
        """
        results = await asyncio.gather(*[server.abort_all_requests.remote() for server in self.servers])

        total_aborted = sum(r.get("aborted_count", 0) for r in results)
        all_request_ids = []
        for r in results:
            all_request_ids.extend(r.get("request_ids", []))

        return {
            "aborted_count": total_aborted,
            "request_ids": all_request_ids,
            "server_results": results,
        }

    async def abort_request(self, request_id: str) -> dict[str, Any]:
        """Abort a specific request. Tries all servers since we don't know which one has it.

        Args:
            request_id: The ID of the request to abort.

        Returns:
            dict[str, Any]: Abort result.
        """
        # TODO(petersh6): we should only abort on the server that has the request.
        results = await asyncio.gather(*[server.abort_request.remote(request_id) for server in self.servers])

        for r in results:
            if r.get("aborted", False):
                return r

        return {"aborted": False, "request_id": request_id, "error": "Request not found on any server"}

    async def start_generate_stream(
        self,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        request_id: str,
        image_data: Optional[list[Any]] = None,
        video_data: Optional[list[Any]] = None,
        priority: int = 0,
    ) -> dict[str, Any]:
        """
        Start a streaming generation and return a handle immediately.
        Poll by poll_generate_stream(handle).
        """
        prompt_ids = _qwen2_5_vl_dedup_image_tokens(prompt_ids, self.model_config.processor)

        actual_prompt_len = len(prompt_ids)
        max_possible_tokens = self.config.max_model_len - actual_prompt_len
        if max_possible_tokens < 0:
            raise ValueError(
                f"Prompt length ({actual_prompt_len}) exceeds max_model_len ({self.config.max_model_len})."
            )

        sp = dict(sampling_params)  # do not mutate caller
        if "max_tokens" in sp:
            max_tokens = sp.pop("max_tokens")
        elif "max_new_tokens" in sp:
            max_tokens = sp.pop("max_new_tokens")
        else:
            max_tokens = min(self.config.response_length, max_possible_tokens)
        max_tokens = max(0, max_tokens)

        sp["logprobs"] = 0 if sp.pop("logprobs", False) else None
        sp.setdefault("repetition_penalty", self.config.get("repetition_penalty", 1.0))
        sampling = SamplingParams(max_tokens=max_tokens, **sp)

        multi_modal_data = {}
        if image_data is not None:
            multi_modal_data["image"] = image_data
        if video_data is not None:
            multi_modal_data["video"] = video_data
        prompt = TokensPrompt(prompt_token_ids=prompt_ids, multi_modal_data=multi_modal_data)

        lora_request = None
        if self.model_config.lora_rank > 0:
            lora_loaded = VLLM_LORA_INT_ID in await self.engine.list_loras()
            if lora_loaded:
                lora_request = LoRARequest(
                    lora_name=VLLM_LORA_NAME, lora_int_id=VLLM_LORA_INT_ID, lora_path=VLLM_LORA_PATH
                )

        handle = uuid4().hex
        q: asyncio.Queue = asyncio.Queue(maxsize=256)  # backpressure
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
            final_out: Optional[RequestOutput] = None
            try:
                gen = self.engine.generate(
                    prompt=prompt,
                    sampling_params=sampling,
                    request_id=request_id,
                    lora_request=lora_request,
                    priority=priority,
                )

                async for out in gen:
                    final_out = out
                    token_ids = out.outputs[0].token_ids
                    cur_len = len(token_ids)
                    if cur_len > prev_len:
                        delta = token_ids[prev_len:cur_len]
                        prev_len = cur_len
                        await q.put({"type": "delta", "token_ids": delta, "total_tokens": cur_len})

                if final_out is None:
                    self._stream_meta[handle].update(
                        {"done": True, "finish_reason": "abort", "stop_reason": "aborted", "total_tokens": prev_len}
                    )
                    await q.put({"type": "done", "finish_reason": "abort", "stop_reason": "aborted"})
                    return

                finish_reason = final_out.outputs[0].finish_reason
                if finish_reason == "abort":
                    stop_reason = "aborted"
                elif finish_reason in ("stop", "length"):
                    stop_reason = "completed"
                else:
                    stop_reason = finish_reason

                num_preempted = 0
                if hasattr(final_out, "preempted"):
                    num_preempted = final_out.preempted

                self._stream_meta[handle].update(
                    {
                        "done": True,
                        "finish_reason": finish_reason,
                        "stop_reason": stop_reason,
                        "num_preempted": num_preempted,
                        "total_tokens": prev_len,
                    }
                )
                await q.put(
                    {
                        "type": "done",
                        "finish_reason": finish_reason,
                        "stop_reason": stop_reason,
                        "num_preempted": num_preempted,
                        "total_tokens": prev_len,
                    }
                )

            except Exception as e:
                self._stream_meta[handle].update(
                    {"done": True, "finish_reason": "error", "stop_reason": "error", "total_tokens": prev_len}
                )
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

    async def finalize_generate_stream(self, handle: str) -> dict[str, Any]:
        t = self._stream_tasks.get(handle)
        if t is None:
            return {"ok": False, "error": f"unknown handle {handle}"}
        try:
            await t
        except Exception:
            pass
        meta = self._stream_meta.get(handle, {})
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


def _qwen2_5_vl_dedup_image_tokens(prompt_ids: list[int], processor):
    """Deduplicate consecutive image tokens in prompt_ids for Qwen2.5-VL, since vLLM will replicate the
    <|image_pad|> and <|video_pad|> token by image_data.

    For example,
    ```
    <|vision_start|><|image_pad|><|image_pad|>...<|image_pad|><|vision_end|>
    =>
    <|vision_start|><|image_pad|><|vision_end|>
    ```
    """
    if processor is not None and "Qwen2VLImageProcessor" in processor.image_processor.__class__.__name__:
        prompt_ids = np.array(prompt_ids)

        # Create a mask where True indicates elements to keep
        mask = np.ones(len(prompt_ids), dtype=bool)

        # Find where the array equals the value
        is_value = (prompt_ids == processor.image_token_id) | (prompt_ids == processor.video_token_id)

        # Find consecutive duplicates by checking if previous element is also the value
        mask[1:] &= ~(is_value[1:] & is_value[:-1])

        return prompt_ids[mask].tolist()
    else:
        return prompt_ids

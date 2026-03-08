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
import asyncio
import heapq
import logging
import os
import random
from abc import ABC, abstractmethod
from typing import Any, Optional
from uuid import uuid4

import hydra
import numpy as np
import ray
import torch
from cachetools import LRUCache
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from pydantic import BaseModel, ConfigDict
from tensordict import TensorDict
from transformers import AutoProcessor, AutoTokenizer

from verl.experimental.agent_loop.prometheus_utils import update_prometheus_config
from verl.experimental.agent_loop.utils import resolve_config_path
from verl.protocol import DataProto
from verl.single_controller.ray.base import RayResourcePool, RayWorkerGroup
from verl.utils import hf_processor, hf_tokenizer
from verl.utils.chat_template import initialize_system_prompt
from verl.utils.dataset.rl_dataset import RLHFDataset, get_dataset_class
from verl.utils.fs import copy_to_local
from verl.utils.model import compute_position_id_with_mask
from verl.utils.ray_utils import get_event_loop
from verl.utils.rollout_trace import (
    RolloutTraceConfig,
    rollout_trace_attr,
    rollout_trace_op,
)
from verl.utils.transferqueue_utils import tqbridge
from verl.workers.rollout.replica import TokenOutput, get_rollout_replica_class

import time
from dataclasses import dataclass, field

from typing import AsyncIterator, List, Dict, Any, Optional

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class AsyncLLMServerManager:
    """
    A class to manage multiple OpenAI compatible LLM servers. This class provides
    - Load balance: least requests load balancing
    - Sticky session: send multi-turn chat completions to same server for automatic prefix caching
    """

    def __init__(self, config: DictConfig, server_handles: list[ray.actor.ActorHandle], max_cache_size: int = 10000):
        """Initialize the AsyncLLMServerManager.

        Args:
            config (DictConfig): YAML config.
            server_handles (List[ray.actor.ActorHandle]): OpenAI compatible LLM server actor handles.
            max_cache_size (int, optional): max cache size for request_id to server mapping. Defaults to 10000.
        """
        self.config = config
        self.server_handles = server_handles
        random.shuffle(self.server_handles)

        # Least requests load balancing
        self.weighted_serveres = [[0, idx, server] for idx, server in enumerate(self.server_handles)]
        heapq.heapify(self.weighted_serveres)

        # LRU cache to map request_id to server
        self.request_id_to_server = LRUCache(maxsize=max_cache_size)

        # ===== scheduler =====
        # per-server inflight limiter
        self.max_inflight_per_server = int(getattr(config.actor_rollout_ref.rollout, "max_inflight_per_server", 8))
        self._inflight = {id(s): 0 for s in self.server_handles}

        # global priority queue for pending stream starts
        self._sched_lock = asyncio.Lock()
        self._sched_cv = asyncio.Condition(self._sched_lock)
        self._pending = []  # heap of _SchedItem
        self._seq = 0

        # ===== handle routing (stream handle -> server actor) =====
        self._handle_to_server: dict[str, ray.actor.ActorHandle] = {}

        # inflight decrement must be serialized per server
        self._inflight_lock = asyncio.Lock()

        self._sched_task = asyncio.create_task(self._scheduler_loop())

    def _choose_server(self, request_id: str) -> ray.actor.ActorHandle:
        # TODO: implement server pressure awareness load balancing
        if request_id in self.request_id_to_server:
            return self.request_id_to_server[request_id]

        _, _, server = self.weighted_serveres[0]
        self.weighted_serveres[0][0] += 1
        heapq.heapreplace(self.weighted_serveres, self.weighted_serveres[0])
        self.request_id_to_server[request_id] = server
        return server

    @dataclass(order=True)
    class _SchedItem:
        # heap order: smaller priority first, then older submit
        priority: int
        submit_ts: float
        seq: int
        sticky_id: str = field(compare=False)
        prompt_ids: list[int] = field(compare=False)
        sampling_params: dict[str, Any] = field(compare=False)
        image_data: Optional[list[Any]] = field(compare=False, default=None)
        video_data: Optional[list[Any]] = field(compare=False, default=None)
        fut: asyncio.Future = field(compare=False, default=None)

    async def _scheduler_loop(self):
        """
        Global scheduler: starts streams when the chosen server has capacity.
        IMPORTANT: must register handle->server, and inflight must be decremented on finish/cancel.
        """
        while True:
            async with self._sched_cv:
                while not self._pending:
                    await self._sched_cv.wait()

                item = self._pending[0]
                server = self._choose_server(item.sticky_id)
                sid = id(server)

                if self._inflight.get(sid, 0) >= self.max_inflight_per_server:
                    await self._sched_cv.wait()
                    continue

                heapq.heappop(self._pending)
                self._inflight[sid] = self._inflight.get(sid, 0) + 1

            # start stream outside lock
            try:
                vllm_req_id = uuid4().hex
                res = await server.start_generate_stream.remote(
                    prompt_ids=item.prompt_ids,
                    sampling_params=item.sampling_params,
                    request_id=vllm_req_id,
                    image_data=item.image_data,
                    video_data=item.video_data,
                    priority=item.priority,
                )

                # register handle -> server (REQUIRED for polling/finalize/cancel)
                handle = res["handle"]
                self._handle_to_server[handle] = server

                # enrich with routing info
                res["__sticky_id__"] = item.sticky_id
                res["__server_id__"] = sid
                item.fut.set_result(res)

            except Exception as e:
                item.fut.set_exception(e)
                # release capacity on start failure
                async with self._sched_cv:
                    self._inflight[sid] = max(0, self._inflight.get(sid, 0) - 1)
                    self._sched_cv.notify_all()

    async def start_generate_stream(
        self,
        *,
        sticky_id: str,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        priority: int = 0,
        image_data: Optional[list[Any]] = None,
        video_data: Optional[list[Any]] = None,
    ) -> dict[str, Any]:
        """
        Submit one stream-start request into the global scheduler and wait until the stream handle is returned.
        Returns dict that MUST include "handle".
        """
        fut = asyncio.get_running_loop().create_future()
        async with self._sched_cv:
            self._seq += 1
            item = AsyncLLMServerManager._SchedItem(
                priority=int(priority),
                submit_ts=time.time(),
                seq=self._seq,
                sticky_id=sticky_id,
                prompt_ids=prompt_ids,
                sampling_params=sampling_params,
                image_data=image_data,
                video_data=video_data,
                fut=fut,
            )
            heapq.heappush(self._pending, item)
            self._sched_cv.notify_all()
        return await fut

    async def _release_inflight(self, server: ray.actor.ActorHandle):
        sid = id(server)
        async with self._sched_cv:
            self._inflight[sid] = max(0, self._inflight.get(sid, 0) - 1)
            self._sched_cv.notify_all()

    async def poll_stream(self, handle: str, timeout_s: float = 0.02) -> Optional[dict[str, Any]]:
        """
        Poll one stream handle.
        - returns: None | {"type":"delta",...} | {"type":"done",...} | {"type":"error",...}
        - automatically releases inflight on done/error and cleans handle mapping.
        """
        server = self._handle_to_server.get(handle)
        if server is None:
            return {"type": "error", "error": f"unknown handle {handle}"}

        msg = await server.poll_generate_stream.remote(handle, timeout_s=timeout_s)
        if msg is None:
            return None

        t = msg.get("type")

        if t == "delta":
            return msg

        if t in ("done", "error"):
            await server.finalize_generate_stream.remote(handle)
            self._handle_to_server.pop(handle, None)
            await self._release_inflight(server)
            return msg

        return {"type": "error", "error": f"bad msg type: {t}, msg={msg}"}

    async def cancel_stream(self, handle: str, reset_prefix_cache: bool = True) -> dict[str, Any]:
        """
        Cancel one stream handle.
        - always releases inflight and forgets handle mapping.
        """
        server = self._handle_to_server.get(handle)
        if server is None:
            return {"ok": False, "error": f"unknown handle {handle}"}

        res = await server.cancel_generate_stream.remote(handle, reset_prefix_cache=reset_prefix_cache)
        await server.finalize_generate_stream.remote(handle)
        self._handle_to_server.pop(handle, None)
        await self._release_inflight(server)
        return res

    @rollout_trace_op
    async def generate(
        self,
        request_id,
        *,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        image_data: Optional[list[Any]] = None,
        video_data: Optional[list[Any]] = None,
    ) -> TokenOutput:
        """Generate tokens from prompt ids.

        Args:
            request_id (str): request id for sticky session.
            prompt_ids (List[int]): List of prompt token ids.
            sampling_params (Dict[str, Any]): Sampling parameters for the chat completion.

        Returns:
            TokenOutput: token output
        """

        print("[debug] calling normal generate")

        server = self._choose_server(request_id)
        output = await server.generate.remote(
            request_id=uuid4().hex,  # use new request_id for each turn
            prompt_ids=prompt_ids,
            sampling_params=sampling_params,
            image_data=image_data,
            video_data=video_data,
        )
        return output

    async def add_requests(self, prompts: DataProto) -> List[str]:
        """
        [Sync-API] 发送 Prompts 到 Server，但不开始生成。
        对应 Server 端的 add_requests 接口。
        """
        # 1. 准备 Request IDs (Client 生成，全链路追踪)
        bsz = prompts.batch["prompts"].shape[0]
        request_ids = [uuid4().hex for _ in range(bsz)]

        # 2. 准备 Payload (对齐 Codex 提到的参数格式)
        prompt_ids_batch = []
        prompts_tensor = prompts.batch["prompts"]
        attn = prompts.batch.get("attention_mask", None)

        for i in range(bsz):
            ids = prompts_tensor[i].tolist()
            # 处理 Padding (移除 leading zeros 或根据 mask 截断)
            if attn is not None:
                cur_attn = attn[i]
                if cur_attn.size(0) >= len(ids):
                    cur_attn = cur_attn[: len(ids)]
                valid_len = int(cur_attn.sum().item())
                if valid_len > 0:
                    ids = ids[-valid_len:]
                else:
                    while len(ids) > 0 and ids[0] == 0:
                        ids = ids[1:]
            else:
                while len(ids) > 0 and ids[0] == 0:
                    ids = ids[1:]
            prompt_ids_batch.append(ids)

        # 3. 准备 Sampling Params
        # 注意 Codex 提到的：必须确保 max_tokens 存在
        meta_params = prompts.meta_info.get("sampling_params", {})
        # 获取全局配置兜底 (假设你有 self.config)
        default_max_tokens = getattr(self.config, "response_length", 1024) if hasattr(self, "config") else 1024

        sampling_params = {
            "temperature": meta_params.get("temperature", 1.0),
            "top_p": meta_params.get("top_p", 1.0),
            "top_k": meta_params.get("top_k", -1),
            "max_tokens": meta_params.get("max_tokens", default_max_tokens),
            "logprobs": meta_params.get("logprobs", 1),  # 通常需要 logprobs
            "ignore_eos": meta_params.get("ignore_eos", False),
        }

        payload = {
            "request_ids": request_ids,
            "prompt_ids_batch": prompt_ids_batch,
            "sampling_params": sampling_params,
            # 如果有多模态数据，在这里加 "image_data": ...
        }

        # 4. 发送 RPC (使用 collective_rpc 调用 server 端的 add_requests)
        # 假设 self.server_handles 是一个列表，我们通常发给 Rank 0 或者所有
        # 如果是 SPMD 架构，通常发给 handle[0] 即可
        target_server = self.server_handles[0]  # 或者 self.rollout_worker

        await target_server.collective_rpc.remote("add_requests", args=(payload,))

        return request_ids

    async def step(self, max_steps: int = 1) -> None:
        """
        [Sync-API] 驱动 Server 往前推理 max_steps 步。
        """
        target_server = self.server_handles[0]
        # 这是一个 RPC 调用，但在这一行 await 保证了同步性
        await target_server.collective_rpc.remote("step", kwargs={"max_steps": max_steps})

    async def collect(self, request_ids: List[str]) -> List[Dict[str, Any]]:
        """
        [Sync-API] 收集指定 Request ID 的最新输出。
        """
        if not request_ids:
            return []

        target_server = self.server_handles[0]
        results = await target_server.collective_rpc.remote("collect", args=(request_ids,))

        # 兜底：如果 Server 返回 None
        return results if results is not None else []

    # =========================================================================
    # 2. 高级流式接口 (The Stream Iterator)
    #    这是给 AgentLoop 用的核心工具
    # =========================================================================

    async def generate_stream_iterator(self, prompts: DataProto) -> AsyncIterator[Dict[str, Any]]:
        """
        全自动步进生成器。
        用法:
            async for batch_results in manager.generate_stream_iterator(prompts):
                # check results...
        """
        # 1. 提交任务
        req_ids = await self.add_requests(prompts)
        active_req_ids = set(req_ids)

        # 2. 循环步进
        while active_req_ids:
            # --- A. 步进 (Step) ---
            # 每次只推 1 步，给你最大的控制权
            await self.step(max_steps=1)

            # --- B. 收集 (Collect) ---
            current_ids = list(active_req_ids)
            batch_results = await self.collect(current_ids)

            # --- C. 封装并 Yield ---
            yield_payload = {}
            finished_ids = []

            for res in batch_results:
                rid = res["request_id"]  # 确保 Server 端返回这个 key

                # 兼容性处理：把 Server 返回的 raw dict 转成你上层需要的格式
                # 假设 res 包含: {'new_token_ids': [123], 'finished': False, 'logprobs': ...}
                yield_payload[rid] = {
                    "new_token_ids": res.get("new_token_ids", []),
                    "finished": res.get("finished", False),
                    "logprobs": res.get("logprobs", None),
                    "text": res.get("text", ""),  # 可选
                }

                if res.get("finished", False):
                    finished_ids.append(rid)

            # 更新活跃列表
            for rid in finished_ids:
                active_req_ids.discard(rid)

            # 只有当这一步有产出时才 yield
            if yield_payload:
                yield yield_payload


class AgentLoopMetrics(BaseModel):
    """Agent loop performance metrics."""

    generate_sequences: float = 0.0
    tool_calls: float = 0.0
    num_preempted: int = -1  # -1 means not available


class AgentLoopOutput(BaseModel):
    """Agent loop output."""

    prompt_ids: list[int]
    """Prompt token ids."""
    response_ids: list[int]
    """Response token ids including LLM generated token, tool response token."""
    response_mask: list[int]
    """Response mask, 1 for LLM generated token, 0 for tool response token."""
    response_logprobs: Optional[list[float]] = None
    """Log probabilities for the response tokens."""
    routed_experts: Optional[Any] = None
    """Routed experts for the total tokens."""
    multi_modal_data: Optional[dict[str, Any]] = None
    """Multi-modal data for multi-modal tools."""
    reward_score: Optional[float] = None
    """Reward score for the trajectory."""
    num_turns: int = 0
    """Number of chat turns, including user, assistant, tool."""
    metrics: AgentLoopMetrics
    """Auxiliary performance metrics"""
    extra_fields: dict[str, Any] = {}
    """Extra fields for dynamic addition."""


class _InternalAgentLoopOutput(AgentLoopOutput):
    """Internal agent loop output with padded sequences."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    prompt_ids: torch.Tensor
    """Padded prompt token ids."""
    response_ids: torch.Tensor
    """Padded response token ids."""
    input_ids: torch.Tensor
    """Padded input ids(prompt_ids + response_ids)."""
    position_ids: torch.Tensor
    """Padded position ids."""
    response_mask: torch.Tensor
    """Padded response mask."""
    attention_mask: torch.Tensor
    """Padded attention mask."""
    response_logprobs: Optional[torch.Tensor] = None
    """Padded log probabilities for the response tokens."""
    routed_experts: Optional[torch.Tensor] = None
    """Padded routed experts for the total tokens."""
    multi_modal_inputs: Optional[dict[str, torch.Tensor]] = None
    """Multi-modal inputs for processors (e.g., pixel_values, image_grid_thw)."""
    extra_fields: dict[str, Any] = {}
    """Extra fields for dynamic addition."""


class DictConfigWrap:
    """Wrapper for DictConfig to avoid hydra.utils.instantiate recursive resolve."""

    def __init__(self, config: DictConfig):
        self.config = config


class AgentLoopBase(ABC):
    """An agent loop takes an input message, chat with OpenAI compatible LLM server and interact with various
    environments."""

    def __init__(
        self,
        trainer_config: DictConfigWrap,
        server_manager: AsyncLLMServerManager,
        tokenizer: AutoTokenizer,
        processor: AutoProcessor,
        dataset_cls: type[RLHFDataset],
        dataset_config: DictConfigWrap,
        **kwargs,
    ):
        """Initialize agent loop, each sample will have its own loop instance.

        Args:
            trainer_config (DictConfigWrap): trainer config.
            server_manager (AsyncLLMServerManager): OpenAI compatible LLM server manager.
            tokenizer (AutoTokenizer): Tokenizer for tokenize messages.
            processor (AutoProcessor): Processor for process messages.
            dataset_cls (type[Dataset]): Dataset class for creating dataset, Defaults to RLHFDataset.
            dataset_config (DictConfigWrap): Dataset config.
        """
        self.config = trainer_config.config
        self.server_manager = server_manager
        self.tokenizer = tokenizer
        self.processor = processor
        self.dataset_cls = dataset_cls
        self.dataset_config = dataset_config.config
        self.apply_chat_template_kwargs = self.dataset_config.get("apply_chat_template_kwargs", {})
        self.system_prompt = initialize_system_prompt(self.tokenizer, **self.apply_chat_template_kwargs)
        self.loop = get_event_loop()

    async def process_vision_info(self, messages: list[dict]) -> dict:
        """Extract images and videos from messages.

        Args:
            messages (list[dict]): Input messages.

        Returns:
            dict: Multi-modal data with keys "images" and "videos".
        """
        multi_modal_data = {}
        if self.processor is not None:
            images, videos = await self.dataset_cls.process_vision_info(
                messages, image_patch_size=self.processor.image_processor.patch_size, config=self.dataset_config
            )
            if images is not None:
                multi_modal_data["images"] = images
            if videos is not None:
                multi_modal_data["videos"] = videos

        return multi_modal_data

    async def apply_chat_template(
        self,
        messages: list[dict],
        tools: list[dict] = None,
        images: list[Image.Image] = None,
        videos: list[tuple[torch.Tensor, dict]] = None,
        remove_system_prompt: bool = False,
    ):
        """Apply chat template to messages with optional tools, images, and videos.

        Args:
            messages (list[dict]): Input messages.
            tools (list[dict], optional): Tools schemas. Defaults to None.
            images (list[Image.Image], optional): Input images. Defaults to None.
            videos (list[tuple[torch.Tensor, dict]], optional): Input videos. Defaults to None.
            remove_system_prompt (bool, optional): Whether to remove system prompt. Defaults to False.

        Returns:
            list[int]: Prompt token ids.
        """
        if self.processor is not None:
            raw_prompt = await self.loop.run_in_executor(
                None,
                lambda: self.processor.apply_chat_template(
                    messages,
                    tools=tools,
                    add_generation_prompt=True,
                    tokenize=False,
                    **self.apply_chat_template_kwargs,
                ),
            )

            # split the videos and according metadatas
            if videos is not None:
                videos, video_metadatas = zip(*videos, strict=False)
                videos, video_metadatas = list(videos), list(video_metadatas)
            else:
                video_metadatas = None

            model_inputs = self.processor(
                text=[raw_prompt],
                images=images,
                videos=videos,
                video_metadatas=video_metadatas,
                return_tensors="pt",
                do_sample_frames=False,
            )
            prompt_ids = model_inputs.pop("input_ids").squeeze(0).tolist()
        else:
            prompt_ids = await self.loop.run_in_executor(
                None,
                lambda: self.tokenizer.apply_chat_template(
                    messages,
                    tools=tools,
                    add_generation_prompt=True,
                    tokenize=True,
                    **self.apply_chat_template_kwargs,
                ),
            )

        if remove_system_prompt:
            prompt_ids = prompt_ids[len(self.system_prompt) :]

        return prompt_ids

    @abstractmethod
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        """Run agent loop to interact with LLM server and environment.

        Args:
            sampling_params (Dict[str, Any]): LLM sampling params.
            **kwargs: dataset fields from `verl.utils.dataset.RLHFDataset`.

        Returns:
            AgentLoopOutput: Agent loop output.
        """
        raise NotImplementedError


"""Agent loop registry: key is agent_name, value is a dict of agent loop config
used by hydra.utils.instantiate to initialize agent loop instance.

https://hydra.cc/docs/advanced/instantiate_objects/overview/
"""
_agent_loop_registry: dict[str, dict] = {}


def register(agent_name: str):
    """Register agent loop class."""

    def decorator(subclass: type[AgentLoopBase]) -> type[AgentLoopBase]:
        fqdn = f"{subclass.__module__}.{subclass.__qualname__}"
        _agent_loop_registry[agent_name] = {"_target_": fqdn}
        return subclass

    return decorator


class AgentLoopWorker:
    """Agent loop worker takes a batch of messages and run each message in an agent loop."""

    def __init__(
        self,
        config: DictConfig,
        server_handles: list[ray.actor.ActorHandle],
        reward_loop_worker_handles: list[ray.actor.ActorHandle] = None,
    ):
        """Initialize agent loop manager.
        Args:
            config (DictConfig): YAML config.
            server_handles (List[ray.actor.ActorHandle]): OpenAI compatible LLM server actor handles.
            reward_loop_worker_handles (List[ray.actor.ActorHandle]): Actor handles for streaming reward computation.
        """
        self.config = config

        # for recipe to change
        if not hasattr(self, "server_manager"):
            self.server_manager = AsyncLLMServerManager(config, server_handles)

        self.dataset_cls = get_dataset_class(config.data)
        self.reward_loop_worker_handles = reward_loop_worker_handles

        model_path = config.actor_rollout_ref.model.path
        self.model_name = "/".join(model_path.split("/")[-2:])
        local_path = copy_to_local(config.actor_rollout_ref.model.path)
        self.tokenizer = hf_tokenizer(local_path, trust_remote_code=True)
        self.processor = hf_processor(local_path, trust_remote_code=True)

        agent_loop_config_path = config.actor_rollout_ref.rollout.agent.agent_loop_config_path
        if agent_loop_config_path:
            resolved_path = resolve_config_path(agent_loop_config_path)
            agent_loop_configs = OmegaConf.load(resolved_path)
            for agent_loop_config in agent_loop_configs:
                _agent_loop_registry[agent_loop_config.name] = agent_loop_config
        if self.config.actor_rollout_ref.model.get("custom_chat_template", None) is not None:
            if self.processor is not None:
                self.processor.chat_template = self.config.actor_rollout_ref.model.custom_chat_template
            self.tokenizer.chat_template = self.config.actor_rollout_ref.model.custom_chat_template

        trace_config = self.config.actor_rollout_ref.rollout.get("trace", {})
        RolloutTraceConfig.init(
            self.config.trainer.project_name,
            self.config.trainer.experiment_name,
            trace_config.get("backend"),
            trace_config.get("token2text", False),
            trace_config.get("max_samples_per_step_per_worker", None),
        )

    @tqbridge()
    async def generate_sequences(self, batch: DataProto) -> DataProto:
        """Generate sequences from agent loop.

        Args:
            batch (DataProto): Input batch.

        Returns:
            DataProto: Output batch.
        """

        print("[debug] calling sequences in async + tool call")

        config = self.config.actor_rollout_ref.rollout
        sampling_params = dict(
            temperature=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k,
            repetition_penalty=1.0,
            logprobs=config.calculate_log_probs,
        )

        # override sampling params for validation
        if batch.meta_info.get("validate", False):
            sampling_params["top_p"] = config.val_kwargs.top_p
            sampling_params["top_k"] = config.val_kwargs.top_k
            sampling_params["temperature"] = config.val_kwargs.temperature

        # by default, we assume it's a single turn agent
        if "agent_name" not in batch.non_tensor_batch:
            default_agent_loop = config.agent.default_agent_loop
            batch.non_tensor_batch["agent_name"] = np.array([default_agent_loop] * len(batch), dtype=object)

        if "index" in batch.non_tensor_batch:
            index = batch.non_tensor_batch["index"]
        else:
            index = np.arange(len(batch))

        max_samples_per_worker = RolloutTraceConfig.get_instance().max_samples_per_step_per_worker

        # For n rollouts per sample, we trace all n rollouts for selected samples
        if max_samples_per_worker is not None:
            unique_sample_indices = np.unique(index)
            if max_samples_per_worker < len(unique_sample_indices):
                selected_samples = set(
                    np.random.choice(unique_sample_indices, max_samples_per_worker, replace=False).tolist()
                )
                traced_indices = set(i for i in range(len(batch)) if index[i] in selected_samples)
            else:
                traced_indices = set(range(len(batch)))
        else:
            traced_indices = set(range(len(batch)))

        trajectory_info = await get_trajectory_info(
            batch.meta_info.get("global_steps", -1), index.tolist(), batch.meta_info.get("validate", False)
        )

        tasks = []
        for i in range(len(batch)):
            trace_this_sample = i in traced_indices
            kwargs = {k: v[i] for k, v in batch.non_tensor_batch.items()}
            tasks.append(
                asyncio.create_task(
                    self._run_agent_loop(sampling_params, trajectory_info[i], trace=trace_this_sample, **kwargs)
                )
            )
        outputs = await asyncio.gather(*tasks)

        output = self._postprocess(outputs)

        return output

    async def _run_agent_loop(
        self,
        sampling_params: dict[str, Any],
        trajectory: dict[str, Any],
        *,
        agent_name: str,
        trace: bool = True,
        **kwargs,
    ) -> _InternalAgentLoopOutput:
        with rollout_trace_attr(
            step=trajectory["step"],
            sample_index=trajectory["sample_index"],
            rollout_n=trajectory["rollout_n"],
            validate=trajectory["validate"],
            name="agent_loop",
            trace=trace,
        ):
            assert agent_name in _agent_loop_registry, (
                f"Agent loop {agent_name} not registered, registered agent loops: {_agent_loop_registry.keys()}"
            )

            agent_loop_config = _agent_loop_registry[agent_name]
            agent_loop = hydra.utils.instantiate(
                config=agent_loop_config,
                trainer_config=DictConfigWrap(config=self.config),
                server_manager=self.server_manager,
                tokenizer=self.tokenizer,
                processor=self.processor,
                dataset_cls=self.dataset_cls,
                dataset_config=DictConfigWrap(self.config.data),
            )
            output: AgentLoopOutput = await agent_loop.run(sampling_params, **kwargs)
            return await self._agent_loop_postprocess(output, **kwargs)

    async def _agent_loop_postprocess(self, output, **kwargs) -> _InternalAgentLoopOutput:
        """Perform post-processing operations on the output of each individual agent loop."""
        output.extra_fields["raw_prompt"] = kwargs["raw_prompt"]

        self.tokenizer.padding_side = "left"
        prompt_output = self.tokenizer.pad(
            {"input_ids": output.prompt_ids},
            padding="max_length",
            max_length=self.config.actor_rollout_ref.rollout.prompt_length,
            return_tensors="pt",
            return_attention_mask=True,
        )
        if prompt_output["input_ids"].dim() == 1:
            prompt_output["input_ids"] = prompt_output["input_ids"].unsqueeze(0)
            prompt_output["attention_mask"] = prompt_output["attention_mask"].unsqueeze(0)

        self.tokenizer.padding_side = "right"
        response_output = self.tokenizer.pad(
            {"input_ids": output.response_ids},
            padding="max_length",
            max_length=self.config.actor_rollout_ref.rollout.response_length,
            return_tensors="pt",
            return_attention_mask=True,
        )
        if response_output["input_ids"].dim() == 1:
            response_output["input_ids"] = response_output["input_ids"].unsqueeze(0)
            response_output["attention_mask"] = response_output["attention_mask"].unsqueeze(0)

        response_mask_output = self.tokenizer.pad(
            {"input_ids": output.response_mask},
            padding="max_length",
            max_length=self.config.actor_rollout_ref.rollout.response_length,
            return_tensors="pt",
            return_attention_mask=False,
        )
        if response_mask_output["input_ids"].dim() == 1:
            response_mask_output["input_ids"] = response_mask_output["input_ids"].unsqueeze(0)

        response_logprobs = None
        if output.response_logprobs is not None:
            pad_size = self.config.actor_rollout_ref.rollout.response_length - len(output.response_logprobs)
            response_logprobs = torch.tensor(output.response_logprobs + [0.0] * pad_size).unsqueeze(0)

        response_mask = response_mask_output["input_ids"] * response_output["attention_mask"]
        attention_mask = torch.cat([prompt_output["attention_mask"], response_output["attention_mask"]], dim=1)
        input_ids = torch.cat([prompt_output["input_ids"], response_output["input_ids"]], dim=1)

        routed_experts = None
        if output.routed_experts is not None:
            total_length = input_ids.shape[1]
            length, layer_num, topk_num = output.routed_experts.shape
            if isinstance(output.routed_experts, np.ndarray):
                experts_tensor = torch.from_numpy(output.routed_experts)
            elif isinstance(output.routed_experts, torch.Tensor):
                experts_tensor = output.routed_experts
            else:
                raise TypeError(f"Unsupported type for routed_experts: {type(output.routed_experts)}")
            routed_experts = torch.zeros(1, total_length, layer_num, topk_num, dtype=experts_tensor.dtype)

            start_pos = prompt_output["input_ids"].shape[1] - len(output.prompt_ids)
            end_pos = min(start_pos + length, total_length)

            if start_pos < 0 or end_pos > total_length:
                raise ValueError(
                    f"Invalid position range: start_pos={start_pos}, end_pos={end_pos}, total_length={total_length}"
                )

            routed_experts[:, start_pos:end_pos] = experts_tensor.unsqueeze(0)

        multi_modal_inputs = self._compute_multi_modal_inputs(output, input_ids)
        position_ids = self._compute_position_ids(input_ids, attention_mask, multi_modal_inputs)
        await self._compute_score(
            output,
            prompts=prompt_output["input_ids"],
            responses=response_output["input_ids"],
            attention_mask=attention_mask,
            input_ids=input_ids,
            position_ids=position_ids,
            kwargs=kwargs,
        )

        return _InternalAgentLoopOutput(
            prompt_ids=prompt_output["input_ids"],
            response_ids=response_output["input_ids"],
            input_ids=input_ids,
            position_ids=position_ids,
            response_mask=response_mask,
            attention_mask=attention_mask,
            response_logprobs=response_logprobs,
            routed_experts=routed_experts,
            multi_modal_inputs=multi_modal_inputs,
            multi_modal_data=output.multi_modal_data,
            reward_score=output.reward_score,
            num_turns=output.num_turns,
            metrics=output.metrics,
            extra_fields=output.extra_fields,
        )

    def _compute_multi_modal_inputs(self, output, input_ids) -> dict[str, torch.Tensor]:
        """Compute multi-modal inputs with image and video."""
        multi_modal_inputs = {}
        if self.processor is None:
            return multi_modal_inputs

        images = output.multi_modal_data.get("images")
        videos = output.multi_modal_data.get("videos")
        if videos is not None:
            videos, video_metadatas = zip(*videos, strict=False)
            videos, video_metadatas = list(videos), list(video_metadatas)
        else:
            video_metadatas = None
        current_text = self.tokenizer.decode(input_ids.squeeze(0), skip_special_tokens=True)
        multi_modal_inputs = self.processor(
            text=[current_text],
            images=images,
            videos=videos,
            video_metadatas=video_metadatas,
            return_tensors="pt",
            do_sample_frames=False,
        )
        multi_modal_inputs.pop("input_ids", None)
        multi_modal_inputs.pop("attention_mask", None)

        multi_modal_inputs = dict(multi_modal_inputs.convert_to_tensors("pt"))
        image_grid_thw = multi_modal_inputs.get("image_grid_thw")
        if image_grid_thw is not None:
            images_seqlens = torch.repeat_interleave(image_grid_thw[:, 1] * image_grid_thw[:, 2], image_grid_thw[:, 0])
            multi_modal_inputs["images_seqlens"] = images_seqlens
        return multi_modal_inputs

    def _compute_position_ids(self, input_ids, attention_mask, multi_modal_inputs) -> torch.Tensor:
        """Compute position ids for multi-modal inputs."""
        if self.processor is None:
            return compute_position_id_with_mask(attention_mask)

        image_grid_thw = multi_modal_inputs.get("image_grid_thw")
        video_grid_thw = multi_modal_inputs.get("video_grid_thw")

        vision_position_ids, _ = self.processor.get_rope_index(
            input_ids=input_ids,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            attention_mask=attention_mask,
        )
        vision_position_ids = vision_position_ids.transpose(0, 1)

        valid_mask = attention_mask[0].bool()
        text_position_ids = torch.ones((1, len(input_ids[0])), dtype=torch.long)
        text_position_ids[0, valid_mask] = torch.arange(valid_mask.sum().item())
        text_position_ids = text_position_ids.unsqueeze(0)
        position_ids = torch.cat((text_position_ids, vision_position_ids), dim=1)
        return position_ids

    async def _compute_score(self, output, prompts, responses, attention_mask, input_ids, position_ids, kwargs):
        """Compute reward score for single sample."""
        enable_async_reward = self.reward_loop_worker_handles is not None

        if output.reward_score is None and enable_async_reward:
            batch = TensorDict(
                {
                    "prompts": prompts,
                    "responses": responses,
                    "attention_mask": attention_mask,
                    "input_ids": input_ids,
                    "position_ids": position_ids,
                },
                batch_size=1,
            )
            non_tensor_batch = {
                **{k: np.array([v]) for k, v in kwargs.items()},
                "__num_turns__": np.array([output.num_turns]),
                "tool_extra_fields": np.array([output.extra_fields], dtype=object),
            }

            data = DataProto(
                batch=batch,
                non_tensor_batch=non_tensor_batch,
            )
            selected_reward_loop_worker_handle = random.choice(self.reward_loop_worker_handles)
            result = await selected_reward_loop_worker_handle.compute_score.remote(data)
            output.reward_score = result["reward_score"]
            output.extra_fields["reward_extra_info"] = result["reward_extra_info"]

    def _postprocess(self, inputs: list[_InternalAgentLoopOutput]) -> DataProto:
        """Process the padded outputs from _run_agent_loop and combine them into a batch."""
        prompt_ids = torch.cat([input.prompt_ids for input in inputs], dim=0)
        response_ids = torch.cat([input.response_ids for input in inputs], dim=0)
        response_mask = torch.cat([input.response_mask for input in inputs], dim=0)
        attention_mask = torch.cat([input.attention_mask for input in inputs], dim=0)
        input_ids = torch.cat([input.input_ids for input in inputs], dim=0)
        position_ids = torch.cat([input.position_ids for input in inputs], dim=0)
        optional_outputs = {}
        if inputs[0].response_logprobs is not None:
            optional_outputs["rollout_log_probs"] = torch.cat([input.response_logprobs for input in inputs], dim=0)
        if inputs[0].routed_experts is not None:
            optional_outputs["routed_experts"] = torch.cat([input.routed_experts for input in inputs], dim=0)

        batch = TensorDict(
            {
                "prompts": prompt_ids,
                "responses": response_ids,
                "response_mask": response_mask,
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                **optional_outputs,
            },
            batch_size=len(inputs),
        )

        scores = [input.reward_score for input in inputs]
        if all(score is not None for score in scores):
            prompt_length = prompt_ids.size(1)
            response_length = attention_mask[:, prompt_length:].sum(dim=1) - 1
            rm_scores = torch.zeros_like(response_mask, dtype=torch.float32)
            rm_scores[torch.arange(response_mask.size(0)), response_length] = torch.tensor(scores, dtype=torch.float32)
            batch["rm_scores"] = rm_scores

        non_tensor_batch = {
            "__num_turns__": np.array([input.num_turns for input in inputs], dtype=np.int32),
        }

        reward_extra_infos = [input.extra_fields.get("reward_extra_info", {}) for input in inputs]
        reward_extra_keys = list(reward_extra_infos[0].keys())
        for key in reward_extra_keys:
            non_tensor_batch[key] = np.array([info[key] for info in reward_extra_infos])

        multi_modal_inputs_list = [input.multi_modal_inputs for input in inputs]
        if any(mmi is not None for mmi in multi_modal_inputs_list):
            non_tensor_batch["multi_modal_inputs"] = np.array(multi_modal_inputs_list, dtype=object)

        metrics = [input.metrics.model_dump() for input in inputs]
        extra_fields = {}
        all_keys = set(key for input_item in inputs for key in input_item.extra_fields)
        for key in all_keys:
            temp_arr = np.empty(len(inputs), dtype=object)
            temp_arr[:] = [input.extra_fields.get(key) for input in inputs]
            extra_fields[key] = temp_arr

        non_tensor_batch.update(extra_fields)

        if "rm_scores" in batch.keys():
            meta_info = {"metrics": metrics, "reward_extra_keys": reward_extra_keys}
        else:
            meta_info = {"metrics": metrics}

        return DataProto(
            batch=batch,
            non_tensor_batch=non_tensor_batch,
            meta_info=meta_info,
        )

    def create_transferqueue_client(
        self,
    ):
        """Create a client for data system (TransferQueue)."""
        from verl.single_controller.ray.base import get_random_string
        from verl.utils.transferqueue_utils import create_transferqueue_client

        client_name = get_random_string(length=6)

        self.tq_client = create_transferqueue_client(
            client_id=f"AgentLoopWorker_{client_name}",
            config=self.config.transfer_queue,
        )


async def get_trajectory_info(step, index, validate):
    """Get trajectory info."""
    trajectory_info = []
    rollout_n = 0
    for i in range(len(index)):
        if i > 0 and index[i - 1] == index[i]:
            rollout_n += 1
        else:
            rollout_n = 0
        trajectory_info.append({"step": step, "sample_index": index[i], "rollout_n": rollout_n, "validate": validate})
    return trajectory_info


class AgentLoopManager:
    """Agent loop manager that manages a group of agent loop workers."""

    def __init__(
        self,
        config: DictConfig,
        worker_group: RayWorkerGroup = None,
        rollout_resource_pool: RayResourcePool = None,
        reward_loop_worker_handles: list[ray.actor.ActorHandle] = None,
    ):
        """Initialize agent loop manager."""
        self.config = config
        self.worker_group = worker_group
        self.reward_loop_worker_handles = reward_loop_worker_handles

        if not hasattr(self, "rollout_replica_class"):
            self.rollout_replica_class = get_rollout_replica_class(self.config.actor_rollout_ref.rollout.name)
        if not hasattr(self, "agent_loop_workers_class"):
            self.agent_loop_workers_class = ray.remote(AgentLoopWorker)

        self._initialize_llm_servers(rollout_resource_pool)
        self._init_agent_loop_workers()

    def _initialize_llm_servers(self, rollout_resource_pool: RayResourcePool):
        rollout_world_size = (
            self.config.actor_rollout_ref.rollout.tensor_model_parallel_size
            * self.config.actor_rollout_ref.rollout.data_parallel_size
            * self.config.actor_rollout_ref.rollout.pipeline_model_parallel_size
        )
        world_size = (
            self.worker_group.world_size
            if self.worker_group
            else self.config.trainer.n_gpus_per_node * self.config.trainer.nnodes
        )
        num_replicas = world_size // rollout_world_size

        rollout_config = self.config.actor_rollout_ref.rollout
        model_config = self.config.actor_rollout_ref.model
        self.rollout_replicas = [
            self.rollout_replica_class(
                replica_rank=replica_rank,
                config=rollout_config,
                model_config=model_config,
                gpus_per_node=self.config.trainer.n_gpus_per_node,
            )
            for replica_rank in range(num_replicas)
        ]

        if self.worker_group and rollout_config.name != "trtllm":
            self._run_all([server.init_hybrid(self.worker_group) for server in self.rollout_replicas])
        elif self.worker_group and rollout_config.name == "trtllm":
            self._run_all(
                [
                    server.init_hybrid_colocated(self.worker_group, rollout_resource_pool)
                    for server in self.rollout_replicas
                ]
            )
        else:
            self._run_all([server.init_standalone() for server in self.rollout_replicas])

        self.server_handles = [server._server_handle for server in self.rollout_replicas]
        self.server_addresses = [server._server_address for server in self.rollout_replicas]

        print(f"AgentLoopManager: {self.server_addresses}")

        if rollout_config.prometheus.enable:
            if rollout_config.disable_log_stats:
                raise ValueError("PROMETHEUS needs disable_log_stats==False, but it is currently True.")
            update_prometheus_config(rollout_config.prometheus, self.server_addresses, rollout_config.name)

    def _init_agent_loop_workers(self):
        self.agent_loop_workers = []
        num_workers = self.config.actor_rollout_ref.rollout.agent.num_workers

        node_ids = [node["NodeID"] for node in ray.nodes() if node["Alive"] and node["Resources"].get("CPU", 0) > 0]
        for i in range(num_workers):
            node_id = node_ids[i % len(node_ids)]
            self.agent_loop_workers.append(
                self.agent_loop_workers_class.options(
                    name=f"agent_loop_worker_{i}" + f"_{uuid4().hex[:8]}",
                    scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                        node_id=node_id, soft=True
                    ),
                ).remote(self.config, self.server_handles, self.reward_loop_worker_handles)
            )

    def generate_sequences(self, prompts: DataProto) -> DataProto:
        """Split input batch and dispatch to agent loop workers."""
        print("[debug] Calling generate sequences batched")

        chunkes = prompts.chunk(len(self.agent_loop_workers))
        outputs = ray.get(
            [
                worker.generate_sequences.remote(chunk)
                for worker, chunk in zip(self.agent_loop_workers, chunkes, strict=True)
            ]
        )
        output = DataProto.concat(outputs)

        metrics = [output.meta_info.pop("metrics") for output in outputs]
        timing = self._performance_metrics(metrics, output)

        output.meta_info = {"timing": timing, **outputs[0].meta_info}
        return output

    # =========================================================================
    # [Added for Overlap Decode] Manual Control Interface
    # 这些是新加的方法，用于支持 fit_overlap_decode
    # =========================================================================

    async def add_requests(self, prompts: DataProto) -> List[str]:
        """
        [Sync-API] 发送 Prompts 到 Server，但不开始生成。
        """
        bsz = prompts.batch["prompts"].shape[0]
        request_ids = [uuid4().hex for _ in range(bsz)]

        prompt_ids_batch = []
        prompts_tensor = prompts.batch["prompts"]
        attn = prompts.batch.get("attention_mask", None)

        for i in range(bsz):
            ids = prompts_tensor[i].tolist()
            if attn is not None:
                cur_attn = attn[i]
                if cur_attn.size(0) >= len(ids):
                    cur_attn = cur_attn[: len(ids)]
                valid_len = int(cur_attn.sum().item())
                if valid_len > 0:
                    ids = ids[-valid_len:]
                else:
                    while len(ids) > 0 and ids[0] == 0:
                        ids = ids[1:]
            else:
                while len(ids) > 0 and ids[0] == 0:
                    ids = ids[1:]
            prompt_ids_batch.append(ids)

        meta_params = prompts.meta_info.get("sampling_params", {})
        default_max_tokens = getattr(self.config, "response_length", 1024) if hasattr(self, "config") else 1024

        sampling_params = {
            "temperature": meta_params.get("temperature", 1.0),
            "top_p": meta_params.get("top_p", 1.0),
            "top_k": meta_params.get("top_k", -1),
            "max_tokens": meta_params.get("max_tokens", default_max_tokens),
            "logprobs": meta_params.get("logprobs", 1),
            "ignore_eos": meta_params.get("ignore_eos", False),
        }

        payload = {
            "request_ids": request_ids,
            "prompt_ids_batch": prompt_ids_batch,
            "sampling_params": sampling_params,
        }

        target_server = self.server_handles[0]
        await target_server.collective_rpc.remote("add_requests", args=(payload,))

        return request_ids

    async def step(self, max_steps: int = 1) -> None:
        """
        [Sync-API] 驱动 Server 往前推理 max_steps 步。
        """
        target_server = self.server_handles[0]
        await target_server.collective_rpc.remote("step", kwargs={"max_steps": max_steps})

    async def collect(self, request_ids: List[str]) -> List[Dict[str, Any]]:
        """
        [Sync-API] 收集指定 Request ID 的最新输出。
        """
        if not request_ids:
            return []

        target_server = self.server_handles[0]
        results = await target_server.collective_rpc.remote("collect", args=(request_ids,))
        return results if results is not None else []

    # =========================================================================

    def _performance_metrics(self, metrics: list[list[dict[str, str]]], output: DataProto) -> dict[str, float]:
        timing = {}
        t_generate_sequences = np.array([metric["generate_sequences"] for chunk in metrics for metric in chunk])
        t_tool_calls = np.array([metric["tool_calls"] for chunk in metrics for metric in chunk])
        num_preempted = np.array([metric["num_preempted"] for chunk in metrics for metric in chunk])
        timing["agent_loop/num_preempted/min"] = num_preempted.min()
        timing["agent_loop/num_preempted/max"] = num_preempted.max()
        timing["agent_loop/num_preempted/mean"] = num_preempted.mean()
        timing["agent_loop/generate_sequences/min"] = t_generate_sequences.min()
        timing["agent_loop/generate_sequences/max"] = t_generate_sequences.max()
        timing["agent_loop/generate_sequences/mean"] = t_generate_sequences.mean()
        timing["agent_loop/tool_calls/min"] = t_tool_calls.min()
        timing["agent_loop/tool_calls/max"] = t_tool_calls.max()
        timing["agent_loop/tool_calls/mean"] = t_tool_calls.mean()

        slowest = np.argmax(t_generate_sequences + t_tool_calls)
        attention_mask = output.batch["attention_mask"][slowest]
        prompt_length = output.batch["prompts"].shape[1]
        timing["agent_loop/slowest/generate_sequences"] = t_generate_sequences[slowest]
        timing["agent_loop/slowest/tool_calls"] = t_tool_calls[slowest]
        timing["agent_loop/slowest/prompt_length"] = attention_mask[:prompt_length].sum().item()
        timing["agent_loop/slowest/response_length"] = attention_mask[prompt_length:].sum().item()
        timing["agent_loop/slowest/num_preempted"] = num_preempted[slowest]

        return timing

    def clear_kv_cache(self):
        """Clear all rollout kv cache, but don`t sleep."""
        self._run_all([replica.clear_kv_cache() for replica in self.rollout_replicas])

    def start_profile(self, **kwargs):
        """Start profiling on all rollout replicas."""
        self._run_all([replica.start_profile(**kwargs) for replica in self.rollout_replicas])

    def stop_profile(self):
        """Stop profiling on all rollout replicas."""
        self._run_all([replica.stop_profile() for replica in self.rollout_replicas])

    def _run_all(self, tasks: list[asyncio.Task]):
        async def run_all():
            await asyncio.gather(*tasks)

        asyncio.run(run_all())
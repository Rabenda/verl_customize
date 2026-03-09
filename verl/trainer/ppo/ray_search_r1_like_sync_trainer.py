import asyncio
import ast
import json
import os
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import torch
import yaml
from tensordict import TensorDict

from verl import DataProto
from verl.experimental.agent_loop.tool_parser import ToolParser, _extract_first_json_object, _parse_tool_call_obj
from verl.experimental.agent_loop.utils import build_gpt_oss_tool_response_text
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.utils.chat_template import initialize_system_prompt
from verl.utils.model import compute_position_id_with_mask


def _parse_query_list_fallback(s: str) -> Optional[list[str]]:
    s = s.strip()
    if not s:
        return None
    if len(s) >= 4 and s.startswith("['") and s.endswith("']"):
        inner = s[2:-2].strip()
        if not inner:
            return None
        parts = [p.strip().strip("'").strip() for p in inner.split("', '")]
        return [p for p in parts if p]
    if len(s) >= 4 and s.startswith('["') and s.endswith('"]'):
        inner = s[2:-2].strip()
        if not inner:
            return None
        parts = [p.strip().strip('"').strip() for p in inner.split('", "')]
        return [p for p in parts if p]
    return [s]


def _normalize_query_list(raw_query_list: Any) -> list[str]:
    if isinstance(raw_query_list, str):
        s = raw_query_list.strip()
        if not s or s in ("[]", "''", '""'):
            return []
        try:
            parsed = json.loads(s)
        except json.JSONDecodeError:
            try:
                parsed = ast.literal_eval(s)
            except (ValueError, SyntaxError):
                parsed = _parse_query_list_fallback(s)
        raw_query_list = parsed

    if not isinstance(raw_query_list, list):
        return []

    query_strings = []
    for item in raw_query_list:
        if isinstance(item, str) and item.strip():
            query_strings.append(item.strip())
        elif isinstance(item, dict) and item.get("query"):
            query_strings.append(str(item["query"]).strip())
        elif item is not None:
            query_strings.append(str(item).strip())
    return [query for query in query_strings if query]


def _format_single_retrieval_result(retrieval_result: list[dict[str, Any]]) -> str:
    formatted = []
    for idx, doc in enumerate(retrieval_result):
        if not isinstance(doc, dict):
            content = str(doc)
        else:
            content = (
                doc.get("contents")
                or doc.get("text")
                or doc.get("content")
                or doc.get("body")
                or doc.get("passage")
                or ""
            )
        title = content.split("\n")[0] if content else ""
        text = "\n".join(content.split("\n")[1:]) if content else ""
        formatted.append(f"Doc {idx + 1} (Title: {title})\n{text}".strip())
    return "\n\n".join(item for item in formatted if item).strip()


@dataclass
class _SyncSearchState:
    prompt_ids: list[int]
    response_ids: list[int] = field(default_factory=list)
    response_mask: list[int] = field(default_factory=list)
    assistant_turns: int = 0
    tool_turns: int = 0
    finished: bool = False
    finish_time: float = 0.0

    @property
    def num_turns(self) -> int:
        return 1 + self.assistant_turns + self.tool_turns

    def append_llm_tokens(self, token_ids: list[int]) -> None:
        self.prompt_ids.extend(token_ids)
        self.response_ids.extend(token_ids)
        self.response_mask.extend([1] * len(token_ids))
        self.assistant_turns += 1

    def append_tool_tokens(self, token_ids: list[int]) -> None:
        self.prompt_ids.extend(token_ids)
        self.response_ids.extend(token_ids)
        self.response_mask.extend([0] * len(token_ids))
        self.tool_turns += 1


class SearchR1LikeSyncRayPPOTrainer(RayPPOTrainer):
    """Driver-side sync multi-turn rollout for Search-R1-like retrieval."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._tool_config_cache = None
        self._tool_schemas_cache = None
        self._search_tool_config_cache = None
        self._tool_parser_cache = None
        self._local_retriever = None
        self._local_retriever_topk = None

    def _print_longest_response(self, label: str, data: "DataProto", step: int, **kwargs) -> None:
        """No-op: skip printing long responses for Search-R1 sync rollout."""
        pass

    def _use_sync_search_rollout(self) -> bool:
        multi_turn_cfg = self.config.actor_rollout_ref.rollout.multi_turn
        return bool(multi_turn_cfg.enable and multi_turn_cfg.tool_config_path)

    def _load_tool_config(self) -> dict[str, Any]:
        if self._tool_config_cache is not None:
            return self._tool_config_cache

        tool_config_path = self.config.actor_rollout_ref.rollout.multi_turn.tool_config_path
        if tool_config_path is None:
            raise ValueError("multi_turn.tool_config_path must be set for sync Search-R1-like rollout")
        if not os.path.isabs(tool_config_path):
            tool_config_path = os.path.join(os.getcwd(), tool_config_path)

        with open(tool_config_path, "r", encoding="utf-8") as f:
            payload = yaml.safe_load(f) or {}
        if not isinstance(payload, dict):
            raise ValueError(f"Invalid tool config at {tool_config_path}: expected a mapping")
        self._tool_config_cache = payload
        return payload

    def _get_tool_schemas(self) -> list[dict[str, Any]]:
        if self._tool_schemas_cache is None:
            tools = self._load_tool_config().get("tools", [])
            self._tool_schemas_cache = [tool["tool_schema"] for tool in tools if tool.get("tool_schema")]
        return self._tool_schemas_cache

    def _get_search_tool_config(self) -> dict[str, Any]:
        if self._search_tool_config_cache is not None:
            return self._search_tool_config_cache

        tools = self._load_tool_config().get("tools", [])
        for tool in tools:
            class_name = tool.get("class_name", "")
            tool_schema = tool.get("tool_schema", {})
            function_name = ((tool_schema.get("function") or {}).get("name")) if isinstance(tool_schema, dict) else None
            if function_name == "search" or class_name.endswith("SearchTool"):
                self._search_tool_config_cache = tool.get("config", {}) or {}
                return self._search_tool_config_cache

        raise ValueError("Could not find a search tool entry in multi_turn.tool_config_path")

    def _get_tool_parser(self):
        if self._tool_parser_cache is not None:
            return self._tool_parser_cache
        parser_name = self.config.actor_rollout_ref.rollout.multi_turn.format
        self._tool_parser_cache = ToolParser.get_tool_parser(parser_name, self.tokenizer)
        return self._tool_parser_cache

    def _messages_to_prompt_ids(self, messages: list[dict[str, Any]]) -> list[int]:
        apply_kwargs = dict(self.config.data.get("apply_chat_template_kwargs", {}))
        tool_schemas = self._get_tool_schemas()
        if self.processor is not None:
            raw_prompt = self.processor.apply_chat_template(
                messages, tools=tool_schemas, add_generation_prompt=True, tokenize=False, **apply_kwargs
            )
            model_inputs = self.processor(text=[raw_prompt], return_tensors="pt")
            return model_inputs["input_ids"].squeeze(0).tolist()
        return self.tokenizer.apply_chat_template(
            messages, tools=tool_schemas, add_generation_prompt=True, tokenize=True, **apply_kwargs
        )

    def _tool_messages_to_prompt_ids(self, messages: list[dict[str, Any]]) -> list[int]:
        apply_kwargs = dict(self.config.data.get("apply_chat_template_kwargs", {}))
        system_prompt = initialize_system_prompt(self.tokenizer, **apply_kwargs)
        if self.processor is not None:
            raw_prompt = self.processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False, **apply_kwargs
            )
            model_inputs = self.processor(text=[raw_prompt], return_tensors="pt")
            prompt_ids = model_inputs["input_ids"].squeeze(0).tolist()
        else:
            prompt_ids = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True, **apply_kwargs
            )
        return prompt_ids[len(system_prompt) :]

    def _extract_query_list(self, response_ids: list[int]) -> list[str]:
        if not response_ids:
            return []

        def _extract_queries_from_tool_args(tool_args: Any) -> list[str]:
            """兼容不同 tool_args 格式（dict/list/str）。"""
            if tool_args is None:
                return []
            # 常见：{"query_list": [...]}
            if isinstance(tool_args, dict):
                return _normalize_query_list(
                    tool_args.get("query_list", tool_args.get("queries", tool_args.get("query")))
                )
            # 有些 parser 会直接给 list：["q1", "q2"] 或 [{"query_list":[...]}]
            if isinstance(tool_args, list):
                merged: list[str] = []
                for item in tool_args:
                    if isinstance(item, dict):
                        merged.extend(
                            _normalize_query_list(
                                item.get("query_list", item.get("queries", item.get("query")))
                            )
                        )
                    else:
                        merged.extend(_normalize_query_list(item))
                return merged
            # 兜底：单个字符串/其它类型
            return _normalize_query_list(tool_args)

        tool_calls = []
        try:
            _, tool_calls = asyncio.run(self._get_tool_parser().extract_tool_calls(response_ids))
        except Exception:
            tool_calls = []

        merged_queries = []
        for tool_call in tool_calls:
            if (tool_call.name or "").strip() != "search":
                continue
            try:
                tool_args = json.loads(tool_call.arguments)
            except json.JSONDecodeError:
                continue
            merged_queries.extend(_extract_queries_from_tool_args(tool_args))
        if merged_queries:
            return merged_queries

        decoded = self.tokenizer.decode(response_ids, skip_special_tokens=False)
        first_json = _extract_first_json_object(decoded)
        if first_json:
            try:
                obj = json.loads(first_json)
            except json.JSONDecodeError:
                try:
                    obj = ast.literal_eval(first_json)
                except (ValueError, SyntaxError):
                    obj = None
            fc = _parse_tool_call_obj(obj) if isinstance(obj, dict) else None
            if fc is not None and (fc.name or "").strip() == "search":
                try:
                    tool_args = json.loads(fc.arguments)
                except json.JSONDecodeError:
                    return []
                return _extract_queries_from_tool_args(tool_args)

        return []

    def _get_local_retriever(self):
        if self._local_retriever is not None:
            return self._local_retriever

        search_cfg = self._get_search_tool_config()
        local_cfg = search_cfg.get("local_retriever", search_cfg)

        index_path = local_cfg.get("index_path")
        corpus_path = local_cfg.get("corpus_path")
        if not index_path or not corpus_path:
            raise ValueError(
                "Sync Search-R1-like trainer requires local retriever config. "
                "Please add `local_retriever.index_path` and `local_retriever.corpus_path` "
                f"to `{self.config.actor_rollout_ref.rollout.multi_turn.tool_config_path}`."
            )

        retrieval_method = local_cfg.get("retrieval_method", local_cfg.get("retriever_name", "bm25"))
        retrieval_topk = int(local_cfg.get("retrieval_topk", local_cfg.get("topk", 3)))
        retrieval_model_path = local_cfg.get(
            "retrieval_model_path",
            local_cfg.get("retriever_model", local_cfg.get("model_path")),
        )
        if retrieval_method != "bm25" and not retrieval_model_path:
            raise ValueError("Dense local retriever requires `retrieval_model_path` (or `retriever_model`) to be set.")

        from examples.sglang_multiturn.search_r1_like.local_dense_retriever.retrieval_server import Config, get_retriever

        retriever_config = Config(
            retrieval_method=retrieval_method,
            retrieval_topk=retrieval_topk,
            index_path=index_path,
            corpus_path=corpus_path,
            dataset_path=local_cfg.get("dataset_path", "./data"),
            data_split=local_cfg.get("corpus_split", local_cfg.get("data_split", "train")),
            faiss_gpu=bool(local_cfg.get("faiss_gpu", True)),
            retrieval_model_path=retrieval_model_path or "",
            retrieval_pooling_method=local_cfg.get("retrieval_pooling_method", "mean"),
            retrieval_query_max_length=int(local_cfg.get("retrieval_query_max_length", 256)),
            retrieval_use_fp16=bool(local_cfg.get("retrieval_use_fp16", True)),
            retrieval_batch_size=int(local_cfg.get("retrieval_batch_size", 128)),
        )
        self._local_retriever = get_retriever(retriever_config)
        self._local_retriever_topk = retrieval_topk
        return self._local_retriever

    def _run_local_search_batch(self, query_lists: list[list[str]]) -> list[str]:
        retriever = self._get_local_retriever()
        flat_queries = [query for query_list in query_lists for query in query_list]
        if not flat_queries:
            return [json.dumps({"result": "No search queries provided."}, ensure_ascii=False) for _ in query_lists]

        batch_results = retriever.batch_search(flat_queries, num=self._local_retriever_topk, return_score=False)
        output_texts = []
        offset = 0
        for query_list in query_lists:
            formatted = []
            for _ in query_list:
                docs = batch_results[offset]
                formatted.append(_format_single_retrieval_result(docs))
                offset += 1
            tool_text = "\n---\n".join(item for item in formatted if item).strip()
            if not tool_text:
                tool_text = "No search results found."
            output_texts.append(json.dumps({"result": tool_text}, ensure_ascii=False))
        return output_texts

    def _build_output_from_states(
        self,
        states: list[_SyncSearchState],
        elapsed: float,
        tool_wall: float,
        source_non_tensor_batch: Optional[dict[str, np.ndarray]] = None,
    ) -> DataProto:
        prompt_length = int(self.config.actor_rollout_ref.rollout.prompt_length)
        response_length = int(self.config.actor_rollout_ref.rollout.response_length)
        pad_id = self.tokenizer.pad_token_id or 0
        batch_size = len(states)

        prompts = torch.full((batch_size, prompt_length), int(pad_id), dtype=torch.long)
        prompt_attn = torch.zeros((batch_size, prompt_length), dtype=torch.long)
        responses = torch.full((batch_size, response_length), int(pad_id), dtype=torch.long)
        response_attn = torch.zeros((batch_size, response_length), dtype=torch.long)
        response_mask = torch.zeros((batch_size, response_length), dtype=torch.long)
        finish_times = torch.zeros((batch_size,), dtype=torch.float32)
        generated_lens = torch.zeros((batch_size,), dtype=torch.int32)

        for i, state in enumerate(states):
            prompt_prefix = state.prompt_ids[: len(state.prompt_ids) - len(state.response_ids)] if state.response_ids else state.prompt_ids
            prompt_prefix = prompt_prefix[-prompt_length:]
            if prompt_prefix:
                lp = len(prompt_prefix)
                prompts[i, prompt_length - lp : prompt_length] = torch.tensor(prompt_prefix, dtype=torch.long)
                prompt_attn[i, prompt_length - lp : prompt_length] = 1

            trimmed_response_ids = state.response_ids[:response_length]
            trimmed_response_mask = state.response_mask[:response_length]
            if trimmed_response_ids:
                lr = len(trimmed_response_ids)
                responses[i, :lr] = torch.tensor(trimmed_response_ids, dtype=torch.long)
                response_attn[i, :lr] = 1
                response_mask[i, :lr] = torch.tensor(trimmed_response_mask, dtype=torch.long)

            finish_times[i] = float(state.finish_time)
            generated_lens[i] = int(sum(trimmed_response_mask))

        attention_mask = torch.cat([prompt_attn, response_attn], dim=1)
        input_ids = torch.cat([prompts, responses], dim=1)
        position_ids = compute_position_id_with_mask(attention_mask)

        batch = TensorDict(
            {
                "prompts": prompts,
                "responses": responses,
                "response_mask": response_mask,
                "finish_times": finish_times,
                "generated_lens": generated_lens,
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            },
            batch_size=batch_size,
        )
        if source_non_tensor_batch is not None and "multi_modal_inputs" in source_non_tensor_batch:
            multi_modal_inputs = source_non_tensor_batch["multi_modal_inputs"]
        else:
            multi_modal_inputs = np.array([{} for _ in states], dtype=object)

        non_tensor_batch = {
            "__num_turns__": np.array([state.num_turns for state in states], dtype=np.int32),
            "tool_extra_fields": np.array([{} for _ in states], dtype=object),
            "multi_modal_inputs": multi_modal_inputs,
        }
        meta_info = {
            "timing": {
                "agent_loop/generate_sequences/min": float(elapsed),
                "agent_loop/generate_sequences/max": float(elapsed),
                "agent_loop/generate_sequences/mean": float(elapsed),
                "agent_loop/tool_calls/min": float(tool_wall),
                "agent_loop/tool_calls/max": float(tool_wall),
                "agent_loop/tool_calls/mean": float(tool_wall),
            }
        }
        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch, meta_info=meta_info)

    def _generate_rollout_batch(self, gen_batch: DataProto, *, curr_step_profile: bool = False) -> DataProto:
        if not self._use_sync_search_rollout():
            return super()._generate_rollout_batch(gen_batch, curr_step_profile=curr_step_profile)

        if curr_step_profile:
            self.async_rollout_manager.start_profile(global_step=self.global_steps)

        rollout_cfg = self.config.actor_rollout_ref.rollout
        do_sample = bool(gen_batch.meta_info.get("do_sample", rollout_cfg.do_sample))
        base_sampling_params = {
            "temperature": float(gen_batch.meta_info.get("temperature", rollout_cfg.temperature if do_sample else 0.0)),
            "top_p": float(gen_batch.meta_info.get("top_p", rollout_cfg.top_p)),
            "top_k": int(gen_batch.meta_info.get("top_k", rollout_cfg.top_k)),
            "logprobs": bool(rollout_cfg.calculate_log_probs),
        }

        poll_timeout_ms = int(self.config.trainer.get("sync_search_r1_like_poll_timeout_ms", 20))
        response_length = int(rollout_cfg.response_length)
        max_assistant_turns = rollout_cfg.multi_turn.max_assistant_turns
        max_user_turns = rollout_cfg.multi_turn.max_user_turns

        states = []
        raw_prompts = gen_batch.non_tensor_batch.get("raw_prompt")
        if raw_prompts is None:
            raise ValueError(
                "Sync Search-R1-like rollout expects `raw_prompt` in non_tensor_batch. "
                "Please keep `data.return_raw_chat=True`."
            )

        for raw_prompt in raw_prompts:
            states.append(_SyncSearchState(prompt_ids=self._messages_to_prompt_ids(list(raw_prompt))))

        rollout_start = time.perf_counter()
        tool_wall = 0.0
        turn_gen_times: list[float] = []
        turn_search_times: list[float] = []
        active_indices = list(range(len(states)))

        try:
            turn_num = 0
            while active_indices:
                turn_num += 1
                turn_gen_t0 = time.perf_counter()
                print(f"[Search-R1 Sync] Turn {turn_num}: generation", flush=True)
                handles = []
                handle_to_idx = {}
                token_buffers = {}
                finish_times = {}

                for idx in active_indices:
                    state = states[idx]
                    remaining = response_length - len(state.response_mask)
                    if remaining <= 0:
                        state.finished = True
                        continue
                    if max_user_turns is not None and state.tool_turns >= max_user_turns:
                        state.finished = True
                        continue

                    sampling_params = dict(base_sampling_params)
                    sampling_params["max_tokens"] = remaining
                    request_id = uuid.uuid4().hex
                    ret = self.async_rollout_manager.start_generate_stream(
                        prompt_ids=state.prompt_ids,
                        sampling_params=sampling_params,
                        request_id=request_id,
                        training_global_step=self.global_steps,
                    )
                    handle = ret["handle"]
                    handles.append(handle)
                    handle_to_idx[handle] = idx
                    token_buffers[handle] = []

                if not handles:
                    turn_gen_times.append(time.perf_counter() - turn_gen_t0)
                    turn_search_times.append(0.0)
                    break

                active_handles = set(handles)
                while active_handles:
                    events = self.async_rollout_manager.poll_generate_stream_many(
                        list(active_handles), timeout_ms=poll_timeout_ms
                    )
                    for event in events:
                        handle = event["handle"]
                        typ = event.get("type")
                        if typ == "delta":
                            token_buffers[handle].extend(event.get("token_ids", []))
                        elif typ in ("done", "error"):
                            active_handles.discard(handle)
                            finish_times[handle] = time.perf_counter() - rollout_start

                for handle in handles:
                    self.async_rollout_manager.finalize_generate_stream(handle)

                next_query_lists = []
                next_query_indices = []
                for handle in handles:
                    idx = handle_to_idx[handle]
                    state = states[idx]
                    llm_tokens = token_buffers[handle]
                    state.append_llm_tokens(llm_tokens)
                    state.finish_time = finish_times.get(handle, time.perf_counter() - rollout_start)

                    if len(state.response_mask) >= response_length:
                        state.finished = True
                        continue
                    if max_assistant_turns is not None and state.assistant_turns >= max_assistant_turns:
                        state.finished = True
                        continue

                    query_list = self._extract_query_list(llm_tokens)
                    if not query_list:
                        state.finished = True
                        continue

                    next_query_indices.append(idx)
                    next_query_lists.append(query_list)

                if not next_query_indices:
                    turn_gen_times.append(time.perf_counter() - turn_gen_t0)
                    turn_search_times.append(0.0)
                    break

                print(f"[Search-R1 Sync] Turn {turn_num}: retrieval", flush=True)
                tool_t0 = time.perf_counter()
                tool_results = self._run_local_search_batch(next_query_lists)
                search_elapsed = time.perf_counter() - tool_t0
                tool_wall += search_elapsed
                turn_search_times.append(search_elapsed)

                parser_name = self.config.actor_rollout_ref.rollout.multi_turn.format
                for idx, tool_text in zip(next_query_indices, tool_results, strict=True):
                    state = states[idx]
                    add_messages = [{"role": "tool", "content": tool_text}]
                    if parser_name == "gpt-oss":
                        tool_response_text = build_gpt_oss_tool_response_text(add_messages, ["search"])
                        tool_response_ids = self.tokenizer.encode(tool_response_text, add_special_tokens=False)
                    else:
                        tool_response_ids = self._tool_messages_to_prompt_ids(add_messages)

                    if len(state.response_mask) + len(tool_response_ids) >= response_length:
                        state.finished = True
                        continue
                    state.append_tool_tokens(tool_response_ids)

                turn_gen_times.append(time.perf_counter() - turn_gen_t0)
                active_indices = [idx for idx, state in enumerate(states) if not state.finished]
        finally:
            self.checkpoint_manager.sleep_replicas()
            if curr_step_profile:
                self.async_rollout_manager.stop_profile()

        elapsed = time.perf_counter() - rollout_start
        n_turns = len(turn_gen_times)
        print(f"[Search-R1 Sync] Done: {n_turns} turns, gen={sum(turn_gen_times):.3f}s, retrieval={sum(turn_search_times):.3f}s, wall={elapsed:.3f}s", flush=True)
        return self._build_output_from_states(
            states,
            elapsed=elapsed,
            tool_wall=tool_wall,
            source_non_tensor_batch=gen_batch.non_tensor_batch,
        )

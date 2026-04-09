import asyncio
import ast
import json
import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pprint import pprint
from typing import Any, Optional

import numpy as np
import ray
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


def _doc_token_lengths_from_tool_results(tool_results: list[str], tokenizer) -> list[list[int]]:
    """Parse tool_results (JSON with 'result'), split by doc separator, tokenize each doc; return per-sample list of doc token lengths."""
    out: list[list[int]] = []
    for raw in tool_results:
        try:
            obj = json.loads(raw)
            text = obj.get("result", "") or ""
        except (json.JSONDecodeError, TypeError):
            text = ""
        if not text.strip():
            out.append([0])
            continue
        segments = [s.strip() for s in text.split("\n---\n") if s.strip()]
        if not segments:
            out.append([0])
            continue
        lengths = []
        for seg in segments:
            ids = tokenizer.encode(seg, add_special_tokens=False)
            lengths.append(len(ids))
        out.append(lengths)
    return out


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

    def fit(self):
        """Dispatch to the correct fit implementation based on ``trainer.fit_method``."""
        fit_method = self.config.trainer.get("fit_method", "fit")
        print(f"[SearchR1] fit dispatching → fit_method={fit_method!r}", flush=True)
        if fit_method in ("fit", "naive", "sequential"):
            return super().fit()
        elif fit_method == "fit_overlap_decode":
            return self.fit_overlap_decode()
        else:
            raise ValueError(
                f"Unknown fit_method='{fit_method}' for SearchR1LikeSyncRayPPOTrainer. "
                "Valid choices: fit | naive | sequential | fit_overlap_decode"
            )

    # ── stage-level progress prints (cover both naive and overlap paths) ──

    def _compute_old_log_prob(self, batch):
        print("[STAGE] old_log_prob: start", flush=True)
        result = super()._compute_old_log_prob(batch)
        print("[STAGE] old_log_prob: done", flush=True)
        return result

    def _compute_ref_log_prob(self, batch):
        print("[STAGE] ref_log_prob: start", flush=True)
        result = super()._compute_ref_log_prob(batch)
        print("[STAGE] ref_log_prob: done", flush=True)
        return result

    def _compute_values(self, batch):
        print("[STAGE] compute_values: start", flush=True)
        result = super()._compute_values(batch)
        print("[STAGE] compute_values: done", flush=True)
        return result

    def _update_critic(self, batch):
        print("[STAGE] update_critic: start", flush=True)
        result = super()._update_critic(batch)
        print("[STAGE] update_critic: done", flush=True)
        return result

    def _update_actor(self, batch):
        print("[STAGE] update_actor: start", flush=True)
        result = super()._update_actor(batch)
        print("[STAGE] update_actor: done", flush=True)
        return result

    def _compute_or_extract_reward(self, batch, **kwargs):
        print("[STAGE] compute_reward: start", flush=True)
        result = super()._compute_or_extract_reward(batch, **kwargs)
        print("[STAGE] compute_reward: done", flush=True)
        return result

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

        from examples.sglang_multiturn.search_r1_like.local_dense_retriever.retrieval_server import Config, RetrieverActor

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
            faiss_nprobe=local_cfg.get("faiss_nprobe"),
        )
        # Dense retrieval needs 1 GPU; BM25 needs none.
        num_gpus = 0 if retrieval_method == "bm25" else 1
        self._local_retriever = RetrieverActor.options(num_gpus=num_gpus).remote(retriever_config)
        self._local_retriever_topk = retrieval_topk
        return self._local_retriever

    def _run_local_search_batch(
        self,
        query_lists: list[list[str]],
        return_timing: bool = False,
    ):
        """Run retrieval for all query_lists. If return_timing=True, return (output_texts, timing_dict)."""
        retriever = self._get_local_retriever()
        flat_queries = [query for query_list in query_lists for query in query_list]
        if not flat_queries:
            out = [json.dumps({"result": "No search queries provided."}, ensure_ascii=False) for _ in query_lists]
            return (out, {"encode_s": 0.0, "faiss_s": 0.0, "load_docs_s": 0.0, "format_s": 0.0}) if return_timing else out

        batch_results, timing_out = ray.get(
            retriever.batch_search.remote(flat_queries, num=self._local_retriever_topk)
        )

        format_t0 = time.perf_counter()
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
        format_s = time.perf_counter() - format_t0

        timing_out["format_s"] = format_s
        if return_timing:
            return output_texts, timing_out
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
        turn_faiss_times: list[float] = []  # per turn: faiss_s only (for tool_times CSV)
        turn_doc_token_lengths: list[list[int]] = []  # per turn: flattened list of doc token lengths
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
                    turn_faiss_times.append(0.0)
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
                    turn_faiss_times.append(0.0)
                    break

                print(f"[Search-R1 Sync] Turn {turn_num}: retrieval", flush=True)
                tool_t0 = time.perf_counter()
                tool_results, tool_timing = self._run_local_search_batch(next_query_lists, return_timing=True)
                search_elapsed = time.perf_counter() - tool_t0
                tool_wall += search_elapsed
                turn_search_times.append(search_elapsed)
                turn_faiss_times.append(float(tool_timing.get("faiss_s", 0.0)))

                # 打印本 turn 的 tool call 分项耗时
                encode_s = tool_timing.get("encode_s", 0.0)
                faiss_s = tool_timing.get("faiss_s", 0.0)
                load_docs_s = tool_timing.get("load_docs_s", 0.0)
                format_s = tool_timing.get("format_s", 0.0)
                retrieval_total_s = tool_timing.get("retrieval_total_s", 0.0)
                print(
                    f"[Search-R1 Sync] Turn {turn_num} tool_call breakdown: "
                    f"encode_s={encode_s:.3f} faiss_s={faiss_s:.3f} load_docs_s={load_docs_s:.3f} "
                    f"format_s={format_s:.3f} retrieval_total_s={retrieval_total_s:.3f} tool_call_total_s={search_elapsed:.3f}",
                    flush=True,
                )

                # 1) 统计 tokenize 后每个 document 的 length（按 turn 汇总）
                per_sample_doc_lens = _doc_token_lengths_from_tool_results(tool_results, self.tokenizer)
                flat_lens = [l for lengths in per_sample_doc_lens for l in lengths]
                turn_doc_token_lengths.append(flat_lens)
                if flat_lens:
                    n_docs = len(flat_lens)
                    print(
                        f"[Search-R1 Sync] Turn {turn_num} doc token lengths: n_docs={n_docs} mean={sum(flat_lens)/n_docs:.1f} max={max(flat_lens)} sum={sum(flat_lens)}",
                        flush=True,
                    )

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
        # 2) 两个 turn 的 retrieval 时间：按 turn 列出
        retrieval_turn_str = ",".join(f"t{i+1}={t:.3f}s" for i, t in enumerate(turn_search_times))
        tool_call_total_s = sum(turn_search_times)
        print(
            f"[Search-R1 Sync] Done: {n_turns} turns, gen={sum(turn_gen_times):.3f}s, retrieval={tool_call_total_s:.3f}s ({retrieval_turn_str}), wall={elapsed:.3f}s",
            flush=True,
        )
        print(
            f"[Search-R1 Sync] tool_call_total_s (this step): {tool_call_total_s:.3f}s",
            flush=True,
        )
        if turn_doc_token_lengths:
            all_lens = [l for flat in turn_doc_token_lengths for l in flat]
            if all_lens:
                print(
                    f"[Search-R1 Sync] Doc token length total: n_docs={len(all_lens)} mean={sum(all_lens)/len(all_lens):.1f} max={max(all_lens)}",
                    flush=True,
                )
        # 将 document length & tool time 统计写入 CSV：使用单独的 doc_length_csv_dir，避免和 decoding length 混在一起
        _doc_csv_dir = self.config.trainer.get("doc_length_csv_dir", None)
        if _doc_csv_dir and turn_doc_token_lengths:
            os.makedirs(_doc_csv_dir, exist_ok=True)
            exp_name = getattr(self.config.trainer, "experiment_name", "run")
            safe_name = "".join(c if c.isalnum() or c in "._-" else "_" for c in str(exp_name))
            step = self.global_steps

            # 1) doc length 分布：length,turn,step
            doc_csv_path = os.path.join(_doc_csv_dir, f"doc_lengths_{safe_name}.csv")
            write_header = not os.path.exists(doc_csv_path) or os.path.getsize(doc_csv_path) == 0
            with open(doc_csv_path, "a", newline="") as f:
                if write_header:
                    f.write("length,turn,step\n")
                for turn_0, flat_lens in enumerate(turn_doc_token_lengths):
                    turn_1based = turn_0 + 1
                    for length in flat_lens:
                        f.write(f"{length},{turn_1based},{step}\n")

            # 2) 每个 turn 的 faiss 时间（仅 faiss_s，用于 MFU 图中间块）：time_s,turn,step
            tool_csv_path = os.path.join(_doc_csv_dir, f"tool_times_{safe_name}.csv")
            write_header_tool = not os.path.exists(tool_csv_path) or os.path.getsize(tool_csv_path) == 0
            with open(tool_csv_path, "a", newline="") as f:
                if write_header_tool:
                    f.write("time_s,turn,step\n")
                for turn_0, t in enumerate(turn_faiss_times):
                    turn_1based = turn_0 + 1
                    f.write(f"{t:.6f},{turn_1based},{step}\n")
        return self._build_output_from_states(
            states,
            elapsed=elapsed,
            tool_wall=tool_wall,
            source_non_tensor_batch=gen_batch.non_tensor_batch,
        )

    # =========================================================================
    # Overlap-decode fit: pipeline step-N rollout with step-(N-1) training
    # =========================================================================

    def fit_overlap_decode(self):
        """
        Overlap the multi-turn search rollout of step N with PPO training of step N-1.

        Both phases run on completely different hardware:
          - Rollout  : sglang inference engine  +  retrieval Ray actor  (inference GPUs)
          - Training : FSDP actor / critic workers                      (training GPUs)

        Off-policy note
        ---------------
        ``update_weights()`` must be called *after* the rollout thread finishes to avoid
        modifying inference-engine weights mid-generation.  Consequently, the rollout of
        step N uses the inference-engine weights that were loaded at the *end of step N-2*
        (1-step off-policy).  This is the same trade-off as ``one_step_off_policy`` and is
        acceptable for most GRPO / PPO-style search RL training.

        Timeline per step t (t >= 2)
        -----------------------------
          [Rollout_t  thread  ────────────────────────────────────────────]
          [                   Train_{t-1}: reward + logprob + adv + actor ]
          wait for rollout thread
          update_weights()   ← safe: inference engine is idle now
          store gen_out_t as prev batch for next iteration
        """
        from omegaconf import OmegaConf

        from verl import DataProto
        from verl.trainer.ppo.core_algos import agg_loss
        from verl.trainer.ppo.metric_utils import (
            compute_data_metrics,
            compute_timing_metrics,
            compute_throughout_metrics,
        )
        from verl.trainer.ppo.ray_trainer import (
            apply_kl_penalty,
            compute_advantage,
            compute_response_mask,
        )
        from verl.utils.checkpoint.checkpoint_manager import should_save_ckpt_esi
        from verl.utils.debug import marked_timer
        from verl.utils.metric import reduce_metrics
        from verl.utils.tracking import Tracking
        from tqdm import tqdm

        _ts = lambda: time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        _fmt = lambda v: f"{v:.3f}s" if v is not None else "NA"

        print("\n========== ENTER fit_overlap_decode (Search-R1) ==========", flush=True)

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0
        self._load_checkpoint()
        self.checkpoint_manager.update_weights()

        current_epoch = self.global_steps // len(self.train_dataloader) if len(self.train_dataloader) > 0 else 0

        # if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
        #     val_metrics = self._validate()
        #     assert val_metrics
        #     pprint(f"Initial validation metrics: {val_metrics}")
        #     logger.log(data=val_metrics, step=self.global_steps)
        #     if self.config.trainer.get("val_only", False):
        #         print("========== EXIT fit_overlap_decode (val_only) ==========", flush=True)
        #         return

        progress_bar = tqdm(
            total=self.total_training_steps, initial=self.global_steps, desc="Search-R1 Overlap Decode"
        )
        self.global_steps += 1
        last_val_metrics = None
        self.max_steps_duration = 0

        # Batch carried forward from the previous iteration, ready to train.
        # Contains the full union of (repeated-dataloader-batch  ∪  gen_out) with
        # response_mask already set.  Training metrics are written into it in-place.
        prev_full_batch: Optional[DataProto] = None

        # ── helpers ──────────────────────────────────────────────────────────

        def _make_gen_batch(batch_dict: dict) -> tuple[DataProto, DataProto]:
            """Build raw DataProto and the initial gen_batch_output from a dataloader dict."""
            b = DataProto.from_single_dict(batch_dict)
            b.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature
            b.non_tensor_batch["uid"] = np.array(
                [str(uuid.uuid4()) for _ in range(len(b.batch))], dtype=object
            )
            gb = self._get_gen_batch(b)
            gb.meta_info["global_steps"] = self.global_steps
            gb_out = gb.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
            return b, gb_out

        def _assemble_full_batch(raw_batch: DataProto, gen_out: DataProto) -> DataProto:
            """Union repeated raw batch with rollout output and compute auxiliary fields."""
            fb = raw_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
            fb = fb.union(gen_out)
            if "response_mask" not in fb.batch:
                fb.batch["response_mask"] = compute_response_mask(fb)
            if self.config.trainer.balance_batch:
                self._balance_batch(fb, metrics={})
            fb.meta_info["global_token_num"] = torch.sum(fb.batch["attention_mask"], dim=-1).tolist()
            fb.meta_info["images_seqlens"] = []
            for mm in fb.non_tensor_batch.get("multi_modal_inputs", []):
                if "image_grid_thw" in mm:
                    fb.meta_info["images_seqlens"].extend(mm["images_seqlens"].tolist())
            return fb

        def _run_train_step(
            full_batch: DataProto,
            step_metrics: dict,
            step_timing: dict,
        ) -> DataProto:
            """
            Execute the full PPO training step (reward → logprob → adv → actor update).

            Does NOT call ``update_weights()`` — the caller does that after the
            rollout thread has finished so the inference engine is not disturbed.
            """
            _t = lambda: time.strftime("%H:%M:%S")
            print(f"[{_t()}] [OVERLAP/TRAIN] _run_train_step: start", flush=True)

            # ── reward ──────────────────────────────────────────────────────
            print(f"[{_t()}] [OVERLAP/TRAIN] reward: start", flush=True)
            with marked_timer("reward", step_timing, color="yellow"):
                if self.use_rm and "rm_scores" not in full_batch.batch:
                    if not self.use_reward_loop:
                        rm_out = self.rm_wg.compute_rm_score(full_batch)
                    else:
                        assert self.reward_loop_manager is not None
                        rm_out = self.reward_loop_manager.compute_rm_score(full_batch)
                    full_batch = full_batch.union(rm_out)
                reward_tensor, reward_extra_infos_dict = self._compute_or_extract_reward(
                    full_batch, reward_fn=self.reward_fn, reward_for_val=False
                )

            # ── old log-prob ─────────────────────────────────────────────────
            print(f"[{_t()}] [OVERLAP/TRAIN] old_log_prob: start", flush=True)
            rollout_corr_config = self.config.algorithm.get("rollout_correction", None)
            bypass = rollout_corr_config and rollout_corr_config.get("bypass_mode", False)
            if bypass:
                from verl.trainer.ppo.rollout_corr_helper import apply_bypass_mode

                apply_bypass_mode(
                    batch=full_batch,
                    rollout_corr_config=rollout_corr_config,
                    policy_loss_config=self.config.actor_rollout_ref.actor.policy_loss,
                )
            else:
                with marked_timer("old_log_prob", step_timing, color="blue"):
                    old_log_prob, old_log_prob_mfu = self._compute_old_log_prob(full_batch)
                    actor_cfg = self.config.actor_rollout_ref.actor
                    entropy_agg = agg_loss(
                        loss_mat=old_log_prob.batch["entropys"],
                        loss_mask=full_batch.batch["response_mask"],
                        loss_agg_mode=actor_cfg.loss_agg_mode,
                        loss_scale_factor=actor_cfg.loss_scale_factor,
                    )
                    step_metrics["actor/entropy"] = entropy_agg.detach().item()
                    step_metrics["perf/mfu/actor_infer"] = old_log_prob_mfu
                    old_log_prob.batch.pop("entropys")
                    full_batch = full_batch.union(old_log_prob)
            print(f"[{_t()}] [OVERLAP/TRAIN] old_log_prob: done", flush=True)

            assert "old_log_probs" in full_batch.batch

            # ── reference log-prob ───────────────────────────────────────────
            if self.use_reference_policy:
                print(f"[{_t()}] [OVERLAP/TRAIN] ref_log_prob: start", flush=True)
                with marked_timer("ref_log_prob", step_timing, color="olive"):
                    ref_log_prob = self._compute_ref_log_prob(full_batch)
                    full_batch = full_batch.union(ref_log_prob)
                print(f"[{_t()}] [OVERLAP/TRAIN] ref_log_prob: done", flush=True)

            # ── values (critic) ──────────────────────────────────────────────
            if self.use_critic:
                print(f"[{_t()}] [OVERLAP/TRAIN] compute_values: start", flush=True)
                with marked_timer("values", step_timing, color="cyan"):
                    values = self._compute_values(full_batch)
                    full_batch = full_batch.union(values)
                print(f"[{_t()}] [OVERLAP/TRAIN] compute_values: done", flush=True)

            # ── advantage ────────────────────────────────────────────────────
            print(f"[{_t()}] [OVERLAP/TRAIN] advantage: start", flush=True)
            with marked_timer("adv", step_timing, color="brown"):
                full_batch.batch["token_level_scores"] = reward_tensor
                if reward_extra_infos_dict:
                    full_batch.non_tensor_batch.update(
                        {k: np.array(v) for k, v in reward_extra_infos_dict.items()}
                    )
                if self.config.algorithm.use_kl_in_reward:
                    full_batch, kl_m = apply_kl_penalty(
                        full_batch,
                        kl_ctrl=self.kl_ctrl_in_reward,
                        kl_penalty=self.config.algorithm.kl_penalty,
                    )
                    step_metrics.update(kl_m)
                else:
                    full_batch.batch["token_level_rewards"] = full_batch.batch["token_level_scores"]

                full_batch = compute_advantage(
                    full_batch,
                    adv_estimator=self.config.algorithm.adv_estimator,
                    gamma=self.config.algorithm.gamma,
                    lam=self.config.algorithm.lam,
                    num_repeat=self.config.actor_rollout_ref.rollout.n,
                    norm_adv_by_std_in_grpo=self.config.algorithm.get("norm_adv_by_std_in_grpo", True),
                    config=self.config.algorithm,
                )
            print(f"[{_t()}] [OVERLAP/TRAIN] advantage: done", flush=True)

            # ── critic update ────────────────────────────────────────────────
            if self.use_critic:
                print(f"[{_t()}] [OVERLAP/TRAIN] update_critic: start", flush=True)
                with marked_timer("update_critic", step_timing, color="pink"):
                    critic_out = self._update_critic(full_batch)
                step_metrics.update(reduce_metrics(critic_out.meta_info["metrics"]))
                print(f"[{_t()}] [OVERLAP/TRAIN] update_critic: done", flush=True)

            # ── actor update ─────────────────────────────────────────────────
            if self.config.trainer.critic_warmup <= self.global_steps:
                print(f"[{_t()}] [OVERLAP/TRAIN] update_actor: start", flush=True)
                with marked_timer("update_actor", step_timing, color="red"):
                    actor_out = self._update_actor(full_batch)
                step_metrics.update(reduce_metrics(actor_out.meta_info["metrics"]))
                print(f"[{_t()}] [OVERLAP/TRAIN] update_actor: done", flush=True)

            print(f"[{_t()}] [OVERLAP/TRAIN] _run_train_step: done", flush=True)
            return full_batch

        # ── main training loop ────────────────────────────────────────────────

        for epoch in range(current_epoch, self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                print(f"[{_ts()}] [OVERLAP_DECODE] epoch={epoch} step={self.global_steps}", flush=True)
                metrics: dict = {}
                timing_raw: dict = {}
                is_last_step = self.global_steps >= self.total_training_steps

                with marked_timer("step", timing_raw):
                    raw_batch, gen_batch_output = _make_gen_batch(batch_dict)

                    # ── Phase 1: launch current-step rollout in background ──
                    t_rollout_start = time.perf_counter()
                    with ThreadPoolExecutor(max_workers=1) as ex:
                        rollout_fut = ex.submit(self._generate_rollout_batch, gen_batch_output)

                        # ── Phase 2: train previous step while rollout runs ──
                        train_wall = 0.0
                        if prev_full_batch is not None:
                            print(
                                f"[{_ts()}] [OVERLAP_DECODE] training prev step (step {self.global_steps - 1})"
                                " while rolling out current step",
                                flush=True,
                            )
                            t_train_start = time.perf_counter()
                            prev_full_batch = _run_train_step(prev_full_batch, metrics, timing_raw)
                            train_wall = time.perf_counter() - t_train_start
                            print(
                                f"[{_ts()}] [OVERLAP_DECODE] prev training done: {_fmt(train_wall)}",
                                flush=True,
                            )

                        # ── Phase 3: wait for rollout ──
                        t_wait_start = time.perf_counter()
                        gen_batch_output = rollout_fut.result()
                        wait_wall = time.perf_counter() - t_wait_start

                    rollout_wall = time.perf_counter() - t_rollout_start
                    print(
                        f"[{_ts()}] [OVERLAP_DECODE] rollout done: {_fmt(rollout_wall)}"
                        f" (waited {_fmt(wait_wall)} for thread)",
                        flush=True,
                    )

                    # ── Phase 4: update_weights after rollout thread exits ──
                    # Must come AFTER rollout to avoid modifying inference-engine
                    # weights while generation is in flight.
                    if prev_full_batch is not None:
                        with marked_timer("update_weights", timing_raw, color="red"):
                            self.checkpoint_manager.update_weights()

                    # Checkpoint saving (only when a training step was executed)
                    if prev_full_batch is not None and self.config.trainer.critic_warmup <= self.global_steps:
                        esi_close = should_save_ckpt_esi(
                            max_steps_duration=self.max_steps_duration,
                            redundant_time=self.config.trainer.esi_redundant_time,
                        )
                        if self.config.trainer.save_freq > 0 and (
                            (self.global_steps - 1) % self.config.trainer.save_freq == 0 or esi_close
                        ):
                            with marked_timer("save_checkpoint", timing_raw, color="green"):
                                self._save_checkpoint()

                    # Rollout timing from gen_batch_output meta_info
                    timing_raw.update(gen_batch_output.meta_info.get("timing", {}))
                    gen_batch_output.meta_info.pop("timing", None)

                    # Overlap stats
                    overlap_time = min(rollout_wall, train_wall) if train_wall > 0 else 0.0
                    timing_raw["puzzrl_rollout"] = rollout_wall
                    timing_raw["train_overlap_wall"] = train_wall
                    timing_raw["overlap"] = overlap_time
                    timing_raw["rollout_wait_wall"] = wait_wall
                    print(
                        f"[{_ts()}] [OVERLAP_DECODE] "
                        f"rollout={_fmt(rollout_wall)} train={_fmt(train_wall)} "
                        f"overlap={_fmt(overlap_time)} wait={_fmt(wait_wall)}",
                        flush=True,
                    )

                    # Assemble full batch for NEXT iteration's training.
                    # This is intentionally deferred so the next rollout can start
                    # immediately without waiting for assembly.
                    cur_full_batch = _assemble_full_batch(raw_batch, gen_batch_output)
                    prev_full_batch = cur_full_batch

                # ── validation ───────────────────────────────────────────────
                if (
                    self.val_reward_fn is not None
                    and self.config.trainer.test_freq > 0
                    and (self.global_steps % self.config.trainer.test_freq == 0)
                ):
                    with marked_timer("testing", timing_raw, color="green"):
                        val_metrics = self._validate()
                        if is_last_step:
                            last_val_metrics = val_metrics
                    metrics.update(val_metrics)

                # ── metrics & logging ────────────────────────────────────────
                steps_duration = timing_raw.get("step", 0.0)
                self.max_steps_duration = max(self.max_steps_duration, steps_duration)

                metrics.update({"training/global_step": self.global_steps, "training/epoch": epoch})
                # Use cur_full_batch for data metrics (current rollout's structural stats).
                # Training quality metrics (reward, actor loss, etc.) are already in `metrics`
                # from _run_train_step and correspond to the previous step's data — this is
                # expected behaviour in a 1-step-off-policy pipeline.
                metrics.update(compute_data_metrics(batch=cur_full_batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=cur_full_batch, timing_raw=timing_raw))
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(
                    compute_throughout_metrics(batch=cur_full_batch, timing_raw=timing_raw, n_gpus=n_gpus)
                )

                if getattr(self, "profiler", None) is not None:
                    self.profiler.update(timing_raw=timing_raw, step=self.global_steps)

                logger.log(data=metrics, step=self.global_steps)
                progress_bar.update(1)
                self.global_steps += 1

                if is_last_step:
                    break

        # ── final step: train the last batch with no rollout to overlap ──────
        if prev_full_batch is not None:
            print(
                f"[{_ts()}] [OVERLAP_DECODE] training final batch (step {self.global_steps}, no overlap)",
                flush=True,
            )
            final_metrics: dict = {}
            final_timing: dict = {}
            prev_full_batch = _run_train_step(prev_full_batch, final_metrics, final_timing)
            self.checkpoint_manager.update_weights()
            if self.config.trainer.save_freq > 0:
                self._save_checkpoint()
            logger.log(data=final_metrics, step=self.global_steps)

        pprint(f"Final validation metrics: {last_val_metrics}")
        progress_bar.close()
        print("========== EXIT fit_overlap_decode (Search-R1) ==========", flush=True)

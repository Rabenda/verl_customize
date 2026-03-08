# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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
Dual-model One-Step Off-Policy Trainer.

Inherits from DualRayPPOTrainer, adds two fit modes:
1. fit_serial_step_rollout_train_overlap: Each step runs A rollout + A train in parallel with B rollout + B train.
   Within each step: (A rollout || B rollout) -> (A train || B train), with next-step rollout overlapped.
2. fit_full_parallel: A rollout, B rollout, A train, B train all overlap when possible.
"""

from __future__ import annotations

import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pprint import pprint
import numpy as np
from omegaconf import OmegaConf
from tqdm import tqdm

from verl import DataProto
from verl.trainer.ppo.ray_dual_trainer import DualRayPPOTrainer, StepProfiler, rtprint
from verl.utils.checkpoint.checkpoint_manager import should_save_ckpt_esi
from verl.utils.debug import marked_timer
from verl.utils.metric import reduce_metrics
from verl.utils.py_functional import rename_dict
from verl.utils.rollout_skip import RolloutSkip

from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    compute_variance_proxy_metrics,
)


def _ts():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def _fmt(sec: float) -> str:
    return f"{sec*1000:.1f}ms" if sec < 1 else f"{sec:.2f}s"


def _print_profiling(step: int, timing_raw: dict, prefix: str = "[PROFILE]"):
    """Print timing to screen."""
    keys = [
        "step",
        "gen_a", "gen_b", "gen_parallel_wall",
        "reward_a", "reward_b",
        "old_log_prob_a", "old_log_prob_b",
        "ref_log_prob_a", "ref_log_prob_b",
        "values_a", "values_b",
        "adv_a", "adv_b",
        "update_critic_a", "update_critic_b",
        "update_actor_a", "update_actor_b",
        "update_weights_a", "update_weights_b",
        "a_train_wall", "b_train_wall",
        "overlap", "save_checkpoint", "testing",
    ]
    parts = [f"{prefix} step={step}"]
    for k in keys:
        if k in timing_raw and timing_raw[k] is not None:
            v = timing_raw[k]
            parts.append(f"{k}={_fmt(v)}")
    rtprint(" | ".join(parts))


class DualRayOneStepOffPPOTrainer(DualRayPPOTrainer):
    """
    Dual-model trainer with one-step off-policy style overlap.

    Two fit modes:
    - fit_serial_step_rollout_train_overlap: Per-step serial, but A rollout+train || B rollout+train overlap.
      Next-step rollout overlaps with current-step train.
    - fit_full_parallel: A rollout, B rollout, A train, B train all overlap when possible.
    """

    def fit_serial_step_rollout_train_overlap(self):
        """
        Each step: (A rollout || B rollout) -> (A train || B train).
        Next-step rollout overlaps with current-step train (one-step off-policy style).
        """
        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0
        self._load_checkpoint()
        self.checkpoint_manager_a.update_weights()
        self.checkpoint_manager_b.update_weights()

        if self.cfg_a.rollout.get("skip_rollout", False):
            RolloutSkip(self._build_actor_scoped_config("a"), self.actor_rollout_wg_a).wrap_generate_sequences()
        if self.cfg_b.rollout.get("skip_rollout", False):
            RolloutSkip(self._build_actor_scoped_config("b"), self.actor_rollout_wg_b).wrap_generate_sequences()

        current_epoch = self.global_steps // len(self.train_dataloader) if len(self.train_dataloader) > 0 else 0

        prof_cfg = self.config.trainer.get("profile", {})
        self.profiler = StepProfiler(
            enabled=prof_cfg.get("enabled", True),
            verbose=prof_cfg.get("verbose", True),
            print_every=prof_cfg.get("print_every", 1),
            window=prof_cfg.get("window", 1),
            print_rollout_each=prof_cfg.get("print_rollout_each", False),
        )

        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Dual OneStepOff Progress")
        self.global_steps += 1
        last_val_metrics = None
        self.max_steps_duration = 0

        # Continuous iterator for pipeline
        def _continuous_iterator():
            for ep in range(current_epoch, self.config.trainer.total_epochs):
                for batch_dict in self.train_dataloader:
                    yield ep, batch_dict

        data_iter = iter(_continuous_iterator())
        gen_future = None  # Future for current batch rollout (launched in prev step)
        current = None

        def _run_rollout_both(epoch, batch_dict):
            """Run A and B rollout in parallel, return (batch_a, batch_b, gen_out_a, gen_out_b, timing)."""
            batch_a, batch_b = DataProto.from_single_dict_two_copies(batch_dict)
            actor_cfg_a = self._actor_cfg("a")
            actor_cfg_b = self._actor_cfg("b")
            batch_a.meta_info["temperature"] = actor_cfg_a.rollout.temperature
            batch_b.meta_info["temperature"] = actor_cfg_b.rollout.temperature
            batch_a.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(batch_a.batch))], dtype=object)
            batch_b.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(batch_b.batch))], dtype=object)

            gen_a = self._get_gen_batch(batch_a)
            gen_a.meta_info["global_steps"] = self.global_steps
            gen_a = gen_a.repeat(repeat_times=actor_cfg_a.rollout.n, interleave=True)
            gen_b = self._get_gen_batch(batch_b)
            gen_b.meta_info["global_steps"] = self.global_steps
            gen_b = gen_b.repeat(repeat_times=actor_cfg_b.rollout.n, interleave=True)

            timing = {}
            t0 = time.perf_counter()
            with ThreadPoolExecutor(max_workers=2) as ex:
                fut_a = ex.submit(self._async_mgr("a").generate_sequences, gen_a)
                fut_b = ex.submit(self._async_mgr("b").generate_sequences, gen_b)
                gen_out_a = fut_a.result()
                gen_out_b = fut_b.result()
            timing["gen_parallel_wall"] = time.perf_counter() - t0
            self.checkpoint_manager_a.sleep_replicas()
            self.checkpoint_manager_b.sleep_replicas()
            timing.update(gen_out_a.meta_info.get("timing", {}))
            timing.update(gen_out_b.meta_info.get("timing", {}))
            gen_out_a.meta_info.pop("timing", None)
            gen_out_b.meta_info.pop("timing", None)
            return epoch, batch_a, batch_b, gen_out_a, gen_out_b, timing

        def _run_train_both(batch_a, batch_b, gen_out_a, gen_out_b, timing_raw, metrics):
            """Run _step_one_actor for A and B (with override), then critic/actor update."""
            batch_a, _ = self._step_one_actor("a", batch_a, timing_raw, metrics, gen_batch_output_override=gen_out_a)
            batch_b, _ = self._step_one_actor("b", batch_b, timing_raw, metrics, gen_batch_output_override=gen_out_b)

            if self.use_critic:
                with marked_timer("update_critic_a", timing_raw, color="pink"):
                    critic_out_a = self._update_critic(batch_a)
                metrics.update(reduce_metrics(critic_out_a.meta_info["metrics"]))
                with marked_timer("update_critic_b", timing_raw, color="pink"):
                    critic_out_b = self._update_critic(batch_b)
                metrics.update(reduce_metrics(critic_out_b.meta_info["metrics"]))

            if self.config.trainer.critic_warmup <= self.global_steps:
                with marked_timer("update_actor_a", timing_raw, color="red"):
                    actor_out_a = self._update_actor("a", batch_a)
                metrics.update(reduce_metrics(actor_out_a.meta_info["metrics"]))
                with marked_timer("update_actor_b", timing_raw, color="red"):
                    actor_out_b = self._update_actor("b", batch_b)
                metrics.update(reduce_metrics(actor_out_b.meta_info["metrics"]))

                esi_close = should_save_ckpt_esi(
                    max_steps_duration=self.max_steps_duration,
                    redundant_time=self.config.trainer.esi_redundant_time,
                )
                if self.config.trainer.save_freq > 0 and (
                    self.global_steps % self.config.trainer.save_freq == 0 or esi_close
                ):
                    with marked_timer("save_checkpoint", timing_raw, color="green"):
                        self._save_checkpoint()
                with marked_timer("update_weights_a", timing_raw, color="red"):
                    self.checkpoint_manager_a.update_weights()
                with marked_timer("update_weights_b", timing_raw, color="red"):
                    self.checkpoint_manager_b.update_weights()

            return batch_a, batch_b

        ex = ThreadPoolExecutor(max_workers=2)

        try:
            try:
                current = next(data_iter)
            except StopIteration:
                progress_bar.close()
                return

            while True:
                epoch, batch_dict = current
                rtprint(f"[{_ts()}] [FIT_SERIAL_OVERLAP] epoch={epoch} step={self.global_steps} begin")
                metrics = {}
                timing_raw = {}
                is_last_step = self.global_steps >= self.total_training_steps

                with marked_timer("step", timing_raw):
                    # 1) Wait for previous rollout (or run first rollout for current batch)
                    if gen_future is not None:
                        prev_epoch, batch_a, batch_b, gen_out_a, gen_out_b, prev_timing = gen_future.result()
                        timing_raw.update(prev_timing)
                        rtprint(f"[{_ts()}] [ROLLOUT] prev done gen_parallel_wall={_fmt(prev_timing.get('gen_parallel_wall', 0))}")
                    else:
                        prev_epoch, batch_a, batch_b, gen_out_a, gen_out_b, prev_timing = _run_rollout_both(epoch, batch_dict)
                        timing_raw.update(prev_timing)
                        rtprint(f"[{_ts()}] [ROLLOUT] first done gen_parallel_wall={_fmt(prev_timing.get('gen_parallel_wall', 0))}")

                    # 2) Launch next rollout in background (overlap with train)
                    gen_future = None
                    next_item = None
                    try:
                        next_item = next(data_iter)
                        gen_future = ex.submit(_run_rollout_both, next_item[0], next_item[1])
                        rtprint(f"[{_ts()}] [OVERLAP] launched next rollout in background")
                    except StopIteration:
                        pass

                    # 3) Run train (overlaps with next rollout if launched)
                    t_train0 = time.perf_counter()
                    batch_a, batch_b = _run_train_both(batch_a, batch_b, gen_out_a, gen_out_b, timing_raw, metrics)
                    t_train1 = time.perf_counter()
                    timing_raw["a_train_wall"] = (t_train1 - t_train0) / 2  # approx
                    timing_raw["b_train_wall"] = (t_train1 - t_train0) / 2
                    rtprint(f"[{_ts()}] [TRAIN] done wall={_fmt(t_train1 - t_train0)}")

                if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and (
                    is_last_step or self.global_steps % self.config.trainer.test_freq == 0
                ):
                    with marked_timer("testing", timing_raw, color="green"):
                        val_a = self._validate_one("a")
                        val_b = self._validate_one("b")
                        metrics.update(val_a)
                        metrics.update(val_b)
                        if is_last_step:
                            last_val_metrics = {"a": val_a, "b": val_b}

                steps_duration = timing_raw.get("step", 0.0)
                self.max_steps_duration = max(self.max_steps_duration, steps_duration)
                metrics.update({"training/global_step": self.global_steps, "training/epoch": epoch})
                metrics.update(rename_dict(compute_data_metrics(batch=batch_a, use_critic=self.use_critic), "data_a/"))
                metrics.update(rename_dict(compute_data_metrics(batch=batch_b, use_critic=self.use_critic), "data_b/"))
                metrics.update(compute_timing_metrics(batch=batch_a, timing_raw=timing_raw))
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch_a, timing_raw=timing_raw, n_gpus=n_gpus))
                grad_norm = metrics.get("actor_a/grad_norm") or metrics.get("actor_b/grad_norm")
                metrics.update(compute_variance_proxy_metrics(batch=batch_a, gradient_norm=grad_norm))

                _print_profiling(self.global_steps, timing_raw, "[FIT_SERIAL_OVERLAP]")
                if getattr(self, "profiler", None) is not None:
                    self.profiler.update(timing_raw=timing_raw, step=self.global_steps)

                progress_bar.update(1)
                self.global_steps += 1

                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

                if next_item is None:
                    break
                current = next_item
        finally:
            ex.shutdown(wait=True)

    def fit_full_parallel(self):
        """
        A rollout, B rollout, A train, B train all overlap when possible.
        Pipeline: (A rollout || B rollout) -> (A train || B train), with next batch rollout overlapped.
        """
        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0
        self._load_checkpoint()
        self.checkpoint_manager_a.update_weights()
        self.checkpoint_manager_b.update_weights()

        if self.cfg_a.rollout.get("skip_rollout", False):
            RolloutSkip(self._build_actor_scoped_config("a"), self.actor_rollout_wg_a).wrap_generate_sequences()
        if self.cfg_b.rollout.get("skip_rollout", False):
            RolloutSkip(self._build_actor_scoped_config("b"), self.actor_rollout_wg_b).wrap_generate_sequences()

        current_epoch = self.global_steps // len(self.train_dataloader) if len(self.train_dataloader) > 0 else 0

        prof_cfg = self.config.trainer.get("profile", {})
        self.profiler = StepProfiler(
            enabled=prof_cfg.get("enabled", True),
            verbose=prof_cfg.get("verbose", True),
            print_every=prof_cfg.get("print_every", 1),
            window=prof_cfg.get("window", 1),
            print_rollout_each=prof_cfg.get("print_rollout_each", False),
        )

        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Dual Full Parallel")
        self.global_steps += 1
        last_val_metrics = None
        self.max_steps_duration = 0

        def _continuous_iterator():
            for ep in range(current_epoch, self.config.trainer.total_epochs):
                for batch_dict in self.train_dataloader:
                    yield ep, batch_dict

        data_iter = iter(_continuous_iterator())
        gen_future = None

        def _run_rollout_both(epoch, batch_dict):
            batch_a, batch_b = DataProto.from_single_dict_two_copies(batch_dict)
            actor_cfg_a = self._actor_cfg("a")
            actor_cfg_b = self._actor_cfg("b")
            batch_a.meta_info["temperature"] = actor_cfg_a.rollout.temperature
            batch_b.meta_info["temperature"] = actor_cfg_b.rollout.temperature
            batch_a.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(batch_a.batch))], dtype=object)
            batch_b.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(batch_b.batch))], dtype=object)

            gen_a = self._get_gen_batch(batch_a)
            gen_a.meta_info["global_steps"] = self.global_steps
            gen_a = gen_a.repeat(repeat_times=actor_cfg_a.rollout.n, interleave=True)
            gen_b = self._get_gen_batch(batch_b)
            gen_b.meta_info["global_steps"] = self.global_steps
            gen_b = gen_b.repeat(repeat_times=actor_cfg_b.rollout.n, interleave=True)

            t0 = time.perf_counter()
            with ThreadPoolExecutor(max_workers=2) as ex:
                fut_a = ex.submit(self._async_mgr("a").generate_sequences, gen_a)
                fut_b = ex.submit(self._async_mgr("b").generate_sequences, gen_b)
                gen_out_a = fut_a.result()
                gen_out_b = fut_b.result()
            wall = time.perf_counter() - t0
            self.checkpoint_manager_a.sleep_replicas()
            self.checkpoint_manager_b.sleep_replicas()
            return batch_a, batch_b, gen_out_a, gen_out_b, wall

        def _run_train_both(batch_a, batch_b, gen_out_a, gen_out_b, timing_raw, metrics):
            batch_a, _ = self._step_one_actor("a", batch_a, timing_raw, metrics, gen_batch_output_override=gen_out_a)
            batch_b, _ = self._step_one_actor("b", batch_b, timing_raw, metrics, gen_batch_output_override=gen_out_b)
            if self.use_critic:
                with marked_timer("update_critic_a", timing_raw, color="pink"):
                    critic_out_a = self._update_critic(batch_a)
                metrics.update(reduce_metrics(critic_out_a.meta_info["metrics"]))
                with marked_timer("update_critic_b", timing_raw, color="pink"):
                    critic_out_b = self._update_critic(batch_b)
                metrics.update(reduce_metrics(critic_out_b.meta_info["metrics"]))
            if self.config.trainer.critic_warmup <= self.global_steps:
                with marked_timer("update_actor_a", timing_raw, color="red"):
                    actor_out_a = self._update_actor("a", batch_a)
                metrics.update(reduce_metrics(actor_out_a.meta_info["metrics"]))
                with marked_timer("update_actor_b", timing_raw, color="red"):
                    actor_out_b = self._update_actor("b", batch_b)
                metrics.update(reduce_metrics(actor_out_b.meta_info["metrics"]))
                esi_close = should_save_ckpt_esi(
                    max_steps_duration=self.max_steps_duration,
                    redundant_time=self.config.trainer.esi_redundant_time,
                )
                if self.config.trainer.save_freq > 0 and (
                    self.global_steps % self.config.trainer.save_freq == 0 or esi_close
                ):
                    with marked_timer("save_checkpoint", timing_raw, color="green"):
                        self._save_checkpoint()
                with marked_timer("update_weights_a", timing_raw, color="red"):
                    self.checkpoint_manager_a.update_weights()
                with marked_timer("update_weights_b", timing_raw, color="red"):
                    self.checkpoint_manager_b.update_weights()
            return batch_a, batch_b

        ex = ThreadPoolExecutor(max_workers=2)

        try:
            try:
                current = next(data_iter)
            except StopIteration:
                progress_bar.close()
                return

            while True:
                epoch, batch_dict = current
                rtprint(f"[{_ts()}] [FIT_FULL_PARALLEL] epoch={epoch} step={self.global_steps} begin")
                metrics = {}
                timing_raw = {}
                is_last_step = self.global_steps >= self.total_training_steps

                with marked_timer("step", timing_raw):
                    # 1) Wait for prev rollout (or run first)
                    if gen_future is not None:
                        batch_a, batch_b, gen_out_a, gen_out_b, gen_wall = gen_future.result()
                        timing_raw["gen_parallel_wall"] = gen_wall
                        rtprint(f"[{_ts()}] [ROLLOUT] prev done gen_wall={_fmt(gen_wall)}")
                    else:
                        batch_a, batch_b, gen_out_a, gen_out_b, gen_wall = _run_rollout_both(epoch, batch_dict)
                        timing_raw["gen_parallel_wall"] = gen_wall
                        rtprint(f"[{_ts()}] [ROLLOUT] first done gen_wall={_fmt(gen_wall)}")

                    # 2) Launch next rollout in background (overlap with train)
                    gen_future = None
                    next_item = None
                    try:
                        next_item = next(data_iter)
                        gen_future = ex.submit(_run_rollout_both, next_item[0], next_item[1])
                        rtprint(f"[{_ts()}] [OVERLAP] launched next rollout in background")
                    except StopIteration:
                        pass

                    t_train0 = time.perf_counter()
                    batch_a, batch_b = _run_train_both(batch_a, batch_b, gen_out_a, gen_out_b, timing_raw, metrics)
                    t_train1 = time.perf_counter()
                    timing_raw["a_train_wall"] = (t_train1 - t_train0) / 2
                    timing_raw["b_train_wall"] = (t_train1 - t_train0) / 2

                    if gen_future is not None:
                        try:
                            next_result = gen_future.result()
                            next_gen_wall = next_result[4]
                            overlap = min(t_train1 - t_train0, next_gen_wall)
                            timing_raw["overlap"] = overlap
                            rtprint(f"[{_ts()}] [OVERLAP] train={_fmt(t_train1-t_train0)} next_gen={_fmt(next_gen_wall)} overlap={_fmt(overlap)}")
                        except Exception:
                            timing_raw["overlap"] = 0.0

                if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and (
                    is_last_step or self.global_steps % self.config.trainer.test_freq == 0
                ):
                    with marked_timer("testing", timing_raw, color="green"):
                        val_a = self._validate_one("a")
                        val_b = self._validate_one("b")
                        metrics.update(val_a)
                        metrics.update(val_b)
                        if is_last_step:
                            last_val_metrics = {"a": val_a, "b": val_b}

                steps_duration = timing_raw.get("step", 0.0)
                self.max_steps_duration = max(self.max_steps_duration, steps_duration)
                metrics.update({"training/global_step": self.global_steps, "training/epoch": epoch})
                metrics.update(rename_dict(compute_data_metrics(batch=batch_a, use_critic=self.use_critic), "data_a/"))
                metrics.update(rename_dict(compute_data_metrics(batch=batch_b, use_critic=self.use_critic), "data_b/"))
                metrics.update(compute_timing_metrics(batch=batch_a, timing_raw=timing_raw))
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch_a, timing_raw=timing_raw, n_gpus=n_gpus))
                grad_norm = metrics.get("actor_a/grad_norm") or metrics.get("actor_b/grad_norm")
                metrics.update(compute_variance_proxy_metrics(batch=batch_a, gradient_norm=grad_norm))

                _print_profiling(self.global_steps, timing_raw, "[FIT_FULL_PARALLEL]")
                if getattr(self, "profiler", None) is not None:
                    self.profiler.update(timing_raw=timing_raw, step=self.global_steps)

                progress_bar.update(1)
                self.global_steps += 1

                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

                if next_item is None:
                    break
                current = next_item
        finally:
            ex.shutdown(wait=True)

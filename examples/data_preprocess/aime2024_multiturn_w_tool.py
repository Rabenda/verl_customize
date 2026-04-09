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
Preprocess the AIME-2024 dataset to multiturn format.

Prompt is set to answer-only evaluation: natural language reasoning only, no code.
Reward uses calc_math_reward (same as math_tool_config): parse final answer from \\boxed{}.
"""

import argparse
import os

import datasets

from verl.utils.hdfs_io import copy, makedirs

# Answer-only, no code: align with math_tool_config (calc_math_reward).
INSTRUCTION_FOLLOWING = (
    "Let's think step by step using natural language only (do not write or run code). "
    "Output your final answer as an integer between 000 and 999 inside \\boxed{}."
)
SYSTEM_CONTENT = (
    "You are an expert in contest math (AIME). Solve the problem step by step using reasoning only; "
    "do not write or execute code. Use the `calc_math_reward` tool after you have an answer to check it; "
    "put your final answer inside \\boxed{} as an integer from 000 to 999."
)


def _get_question_from_prompt(prompt):
    """Extract the first user message content from prompt (list of messages)."""
    if not prompt or not isinstance(prompt, list):
        return ""
    for msg in prompt:
        if isinstance(msg, dict) and msg.get("role") == "user":
            content = msg.get("content", "")
            return content if isinstance(content, str) else ""
    return ""


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default=None, help="The save directory for the preprocessed dataset.")
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument("--local_dataset_path", default=None, help="The local path to the raw dataset, if it exists.")
    parser.add_argument(
        "--local_save_dir", default="~/data/retool_aime2024", help="The save directory for the preprocessed dataset."
    )

    args = parser.parse_args()
    local_dataset_path = args.local_dataset_path

    data_path = "BytedTsinghua-SIA/AIME-2024"

    if local_dataset_path is not None:
        dataset = datasets.load_dataset(local_dataset_path, "default")
    else:
        dataset = datasets.load_dataset(data_path, "default")

    train_dataset = dataset["train"]
    data_source = "BytedTsinghua-SIA/AIME-2024"

    def make_map_fn(split):
        def process_fn(example, idx):
            ground_truth = example["reward_model"]["ground_truth"]
            if isinstance(ground_truth, list):
                ground_truth = ground_truth[0] if ground_truth else ""
            ground_truth = str(ground_truth).strip()

            question_raw = _get_question_from_prompt(example.get("prompt"))
            if not question_raw:
                question_raw = example.get("problem") or example.get("question") or ""
            question = (question_raw + " " + INSTRUCTION_FOLLOWING).strip()

            orig_extra_info = example.pop("extra_info", {}) or {}
            extra_info = dict(orig_extra_info)
            extra_info["need_tools_kwargs"] = True
            # Align with math_tool_config (calc_math_reward); answer-only, no code.
            extra_info["tools_kwargs"] = {
                "calc_math_reward": {
                    "create_kwargs": {"ground_truth": ground_truth},
                },
            }
            extra_info.setdefault("split", split)
            extra_info.setdefault("index", idx)

            example["data_source"] = data_source
            example["prompt"] = [
                {"role": "system", "content": SYSTEM_CONTENT},
                {"role": "user", "content": question},
            ]
            example["extra_info"] = extra_info
            return example

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)

    hdfs_dir = args.hdfs_dir
    local_save_dir = args.local_dir
    if local_save_dir is not None:
        print("Warning: Argument 'local_dir' is deprecated. Please use 'local_save_dir' instead.")
    else:
        local_save_dir = args.local_save_dir

    train_dataset.to_parquet(os.path.join(local_save_dir, "train.parquet"))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_save_dir, dst=hdfs_dir)

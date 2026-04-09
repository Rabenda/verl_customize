# Copyright 2026 Bytedance Ltd. and/or its affiliates
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
Preprocess the MATH-lighteval dataset to multiturn + tool format (similar to gsm8k_multiturn_w_tool.py).
"""

import argparse
import os

import datasets

from verl.utils.hdfs_io import copy, makedirs
from verl.utils.reward_score.math_reward import last_boxed_only_string, remove_boxed


def extract_solution(solution_str: str) -> str:
    return remove_boxed(last_boxed_only_string(solution_str))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default=None)
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument("--local_dataset_path", default=None, help="The local path to the raw dataset, if it exists.")
    parser.add_argument("--local_save_dir", default="~/data/math", help="The save directory for the preprocessed dataset.")
    args = parser.parse_args()

    # 'lighteval/MATH' is no longer available on huggingface.
    # Use mirror repo: DigitalLearningGmbH/MATH-lighteval
    data_source = "DigitalLearningGmbH/MATH-lighteval"
    print(f"Loading the {data_source} dataset from huggingface...", flush=True)
    if args.local_dataset_path is not None:
        dataset = datasets.load_dataset(args.local_dataset_path)
    else:
        dataset = datasets.load_dataset(data_source)

    train_dataset = dataset["train"]
    test_dataset = dataset["test"]

    instruction_following = "Let's think step by step and output the final answer within \\\\boxed{}."

    def make_map_fn(split):
        def process_fn(example, idx):
            problem = example.pop("problem")
            question = problem + " " + instruction_following

            answer_raw = example.pop("solution")
            solution = extract_solution(answer_raw)

            data = {
                "data_source": data_source,
                "prompt": [
                    {
                        "role": "system",
                        "content": (
                            "You are a math expert. You are given a question and you need to solve it step by step. "
                            "Reasoning step by step before any tool call. "
                            "You should use the `calc_math_reward` tool after step by step solving the question, "
                            "before generate final answer at least once and refine your answer if necessary. "
                            "Put your final answer within \\\\boxed{}."
                        ),
                    },
                    {"role": "user", "content": question},
                ],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": solution},
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "need_tools_kwargs": True,
                    "tools_kwargs": {
                        "calc_math_reward": {
                            "create_kwargs": {"ground_truth": solution},
                        }
                    },
                    "interaction_kwargs": {
                        "query": question,
                        "ground_truth": solution,
                    },
                },
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True)

    local_save_dir = args.local_dir
    if local_save_dir is not None:
        print("Warning: Argument 'local_dir' is deprecated. Please use 'local_save_dir' instead.")
    else:
        local_save_dir = args.local_save_dir

    local_dir = os.path.expanduser(local_save_dir)
    os.makedirs(local_dir, exist_ok=True)

    train_dataset.to_parquet(os.path.join(local_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(local_dir, "test.parquet"))

    if args.hdfs_dir is not None:
        makedirs(args.hdfs_dir)
        copy(src=local_dir, dst=args.hdfs_dir)


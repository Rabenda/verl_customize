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

import warnings
from enum import Enum

from omegaconf import DictConfig

from verl.single_controller.base import Worker
from verl.trainer.ppo.core_algos import AdvantageEstimator

WorkerType = type[Worker]


class Role(Enum):
    """
    To create more roles dynamically, you can subclass Role and add new members
    """

    Actor = 0
    Rollout = 1
    ActorRollout = 2
    Critic = 3
    RefPolicy = 4
    RewardModel = 5
    ActorRolloutRef = 6
    Env = 7
    ActorRolloutA = 8
    ActorRolloutB = 9
    # Dual one-step-off roles: separate actor/rollout per model on dedicated pools
    ActorA = 10    # Model-A actor (trains on pool_b)
    RolloutA = 11  # Model-A rollout (infers on pool_a)
    ActorB = 12    # Model-B actor (trains on pool_a)
    RolloutB = 13  # Model-B rollout (infers on pool_b)
    RefA = 14      # Model-A ref policy (pool_b, colocated with ActorA)
    RefB = 15      # Model-B ref policy (pool_a, colocated with ActorB)
    # Multi-model (N-way colocated) roles: index 0..7
    ActorRolloutM0 = 16
    ActorRolloutM1 = 17
    ActorRolloutM2 = 18
    ActorRolloutM3 = 19
    ActorRolloutM4 = 20
    ActorRolloutM5 = 21
    ActorRolloutM6 = 22
    ActorRolloutM7 = 23

    def __str__(self):
        return self._get_role_string()

    def _get_role_string(self):
        role_mapping = {
            Role.Actor: "actor",
            Role.Rollout: "rollout",
            Role.ActorRollout: "actor_rollout",
            Role.Critic: "critic",
            Role.RefPolicy: "ref",
            Role.RewardModel: "rm",
            Role.ActorRolloutRef: "actor_rollout_ref",
            Role.ActorRolloutA: "actor_rollout_a",
            Role.ActorRolloutB: "actor_rollout_b",
            Role.ActorA: "actor_a",
            Role.RolloutA: "rollout_a",
            Role.ActorB: "actor_b",
            Role.RolloutB: "rollout_b",
            Role.RefA: "ref_a",
            Role.RefB: "ref_b",
            Role.ActorRolloutM0: "actor_rollout_m0",
            Role.ActorRolloutM1: "actor_rollout_m1",
            Role.ActorRolloutM2: "actor_rollout_m2",
            Role.ActorRolloutM3: "actor_rollout_m3",
            Role.ActorRolloutM4: "actor_rollout_m4",
            Role.ActorRolloutM5: "actor_rollout_m5",
            Role.ActorRolloutM6: "actor_rollout_m6",
            Role.ActorRolloutM7: "actor_rollout_m7",
        }
        return role_mapping.get(self, self.name.lower())

    @classmethod
    def from_string(cls, name: str):
        string_mapping = {
            "actor": cls.Actor,
            "rollout": cls.Rollout,
            "actor_rollout": cls.ActorRollout,
            "actor_rollout_a": cls.ActorRolloutA,
            "actor_rollout_b": cls.ActorRolloutB,
            "critic": cls.Critic,
            "ref": cls.RefPolicy,
            "rm": cls.RewardModel,
            "actor_rollout_ref": cls.ActorRolloutRef,
            "actor_a": cls.ActorA,
            "rollout_a": cls.RolloutA,
            "actor_b": cls.ActorB,
            "rollout_b": cls.RolloutB,
            "ref_a": cls.RefA,
            "ref_b": cls.RefB,
            "actor_rollout_m0": cls.ActorRolloutM0,
            "actor_rollout_m1": cls.ActorRolloutM1,
            "actor_rollout_m2": cls.ActorRolloutM2,
            "actor_rollout_m3": cls.ActorRolloutM3,
            "actor_rollout_m4": cls.ActorRolloutM4,
            "actor_rollout_m5": cls.ActorRolloutM5,
            "actor_rollout_m6": cls.ActorRolloutM6,
            "actor_rollout_m7": cls.ActorRolloutM7,
        }
        role = string_mapping.get(name.lower())
        if role is None:
            raise ValueError(f"No Role found for string: {name}")
        return role


def need_reference_policy(
    config: DictConfig,
) -> bool:
    """Given the config, do we need ref policy."""
    return config.algorithm.use_kl_in_reward or config.actor_rollout_ref.actor.use_kl_loss


def need_reward_model(
    role_worker_mapping: dict[Role, WorkerType],
) -> bool:
    """Given a role worker mapping, do we need reward model."""
    return Role.RewardModel in role_worker_mapping


def need_critic(config: DictConfig) -> bool:
    """Given a config, do we need critic."""
    if config.critic.enable is not None:
        return bool(config.critic.enable)
    elif config.algorithm.adv_estimator == AdvantageEstimator.GAE:
        return True
    else:
        warnings.warn(
            "Disabled critic as algorithm.adv_estimator != gae. If it is not intended, please set critic.enable=True",
            stacklevel=2,
        )
        return False

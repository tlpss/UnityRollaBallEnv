import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np

sys.path.append(str(Path(__file__).parents[1]))

from goalgym_unity import BaseAction, BaseGoal, BaseObservation, BaseUnityToGoalGymWrapper, UnityObservation


@dataclass
class Goal(BaseGoal):
    """
    Goal of this env is the 2D position that the agent has to reach,
    visually indicated with a cube in Unity
    """

    goal_tolerance = 1.0

    target_position_x: float
    target_position_z: float

    def to_numpy(self) -> np.ndarray:
        return np.array([self.target_position_x, self.target_position_z])


@dataclass
class Observation(BaseObservation):
    """
    Observation of this env is the full state
    """

    target_position: np.ndarray  # (3,)
    agent_position: np.ndarray  # (3,)

    agent_velocity_x: float
    agent_velocity_z: float

    def to_numpy(self) -> np.ndarray:
        return np.array(
            [self.target_position, self.agent_position, self.agent_velocity_x, self.agent_velocity_z]
        ).flatten()


@dataclass
class Action(BaseAction):
    """
    Action of this env is the 2D force to apply on the agent
    """

    force_x: float
    force_z: float

    def to_list(self):
        return [self.force_x, self.force_z]


class UnityToGoalGymWrapper(BaseUnityToGoalGymWrapper):
    """
    Implements Base Interface using the previously defined Agent Goal, Action and Observation
    Unity observation has shape [np.ndarray(84,84,3) , np.ndarray(8,)], where the order of
    the vector observation corresponds with the Observation dataclass
    """

    goal_tolerance = 1.0 # L2 norm tolerance for target position

    def compute_reward(self, achieved_goal: BaseGoal, desired_goal: BaseGoal, info: Dict = None):
        # sparse reward: 1 on success, 0 else
        distance = np.linalg.norm(achieved_goal.to_numpy() - desired_goal.to_numpy())
        if distance < self.goal_tolerance:
            return 1.0
        return 0.0

    def _generate_observation(self, observation: UnityObservation) -> Observation:
        # Unity observation has shape [ np.ndarray(res,res,3) , np.ndarray(8,)]
        # can be seen in the Agent Script
        vector_observation = observation[-1]
        return Observation(
            vector_observation[:3], vector_observation[3:6], vector_observation[6], vector_observation[7]
        )

    def _sample_goal(self) -> Goal:
        return Goal(0.0, 0.0)

    def _set_achieved_goal(self, observation: UnityObservation):
        # Unity observation has shape [ np.ndarray(res,res,3) , np.ndarray(8,)]
        # can be seen in the Agent Script
        vector_obs = observation[-1]
        # these positions are hardcoded in Unity, cf Agent Script to get their values
        agent_x = vector_obs[3]
        agent_z = vector_obs[5]
        self._achieved_goal = Goal(agent_x, agent_z)

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).parents[1]))

from goalgym_unity import BaseAction, BaseGoal, BaseObservation, BaseUnityToGoalGymWrapper, UnityObservation

@dataclass
class Goal(BaseGoal):
    """
    Goal of this env is the 2D position that the agent has to reach,
    visually indicated with a cube in Unity
    """

    target_position_x: float
    target_position_z: float

    def to_numpy(self) -> np.ndarray:
        return np.array([self.target_position_x, self.target_position_z])


@dataclass
class Observation(BaseObservation):
    """
    Observation of this env is the camera input of the overhead camera
    this is a 84x84x3 numpy array with values between 0 and 255
    """

    camera_input: np.ndarray  # 84x84x3

    def to_numpy(self) -> np.ndarray:
        return self.camera_input


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
    the vector is defined in Unity as [target_pose, agent_pose, agent_velocity_x, agent_velocity_z]
    """

    def _generate_observation(self, observation: UnityObservation) -> Observation:
        # Unity observation has shape [ np.ndarray(res,res,3) , np.ndarray(8,)]
        # can be seen in the Agent Script
        camera_observation = observation[0]
        return Observation(camera_observation)

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

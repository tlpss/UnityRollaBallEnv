from dataclasses import dataclass
from typing import List

import numpy as np
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.environment_parameters_channel import EnvironmentParametersChannel

from goalgym_unity import BaseAction, BaseGoal, BaseObservation, BaseUnityToGoalGymWrapper, UnityObservation


@dataclass
class Goal(BaseGoal):
    target_position_x: float
    target_position_z: float

    def to_numpy(self) -> np.ndarray:
        return np.array([self.target_position_x, self.target_position_z])


@dataclass
class Observation(BaseObservation):
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
    force_x: float
    force_z: float

    def to_list(self) -> List:
        return [self.force_x, self.force_z]


class RollABallGym(BaseUnityToGoalGymWrapper):
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


if __name__ == "__main__":
    """
    demo interaction with the gym interface using live Unity environment
    """
    channel = EnvironmentParametersChannel()  # create a params sidechannel
    # None -> live interaction = blocking call to unity
    unity_env = UnityEnvironment(file_name=None, seed=1, side_channels=[channel])

    env = RollABallGym(unity_env, channel, uint8_visual=True, allow_multiple_obs=True)

    while True:
        obs = env.reset(Goal(3.0, 3.0))

        for i in range(1000):
            obs_dict = env.step(Action(0.01, 0.01))
            gym_observation = obs_dict["observation"]
            observation, reward, done, info = gym_observation
            print(observation)

            if done:
                break

from dataclasses import dataclass
from typing import Any, List, Union

import numpy as np
from gym import spaces
from gym_unity.envs import BaseEnv, UnityToGymWrapper
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.environment_parameters_channel import EnvironmentParametersChannel

UnityObservation = Union[List[np.ndarray], np.ndarray]


@dataclass
class GoalState:
    target_position_x: float
    target_position_z: float

    def get_size(self):
        """
        get dimensions of state in numpy format
        """
        return self.to_numpy().shape

    def to_numpy(self):
        """
        return state as numpy vector
        """
        return np.array([self.target_position_x, self.target_position_z])


class UnityToGoalGymWrapper(UnityToGymWrapper):
    """extend Unity Gym interface to work with Goal-based environments.
    This imposes certain structure on the observation:  the observation
    space is required to contain at least three elements, namely `observation`, `desired_goal`, and
    `achieved_goal`. Here, `desired_goal` specifies the goal that the agent should attempt to achieve.
    `achieved_goal` is the goal that it currently achieved instead. `observation` contains the
    actual observations of the environment as per usual.
    cf https://github.com/openai/gym/blob/c8a659369d98706b3c98b84b80a34a832bbdc6c0/gym/core.py#L163

    Furthermore the Unity environment is as such that the there is a vectorobservation,
    containing the relevant state of the environment and a number of visual observations


    """

    def __init__(
        self,
        unity_env: BaseEnv,
        channel: EnvironmentParametersChannel,
        uint8_visual: bool = False,
        flatten_branched: bool = False,
        allow_multiple_obs: bool = False,
    ) -> None:

        super().__init__(unity_env, uint8_visual, flatten_branched, allow_multiple_obs)

        self._channel = channel
        self._desired_goal: GoalState = self._sample_goal()
        self._achieved_goal: GoalState = None
        self._goal_space = spaces.Box(-np.inf, np.inf, shape=self._desired_goal.get_size(), dtype="float32")
        self._observation_space = spaces.Dict(
            {
                "desired_goal": self._goal_space,
                "achieved_goal": self._goal_space,
                "observation": self._observation_space,
            }
        )

    def reset(self, goalstate: GoalState) -> UnityObservation:
        self._desired_goal = goalstate
        # set env params before reset, in particular goal state
        self._channel.set_float_parameter("target_position_x", goalstate.target_position_x)
        self._channel.set_float_parameter("target_position_z", goalstate.target_position_z)
        # reset
        obs = super().reset()
        return obs

    def step(self, action: List[Any]) -> dict:
        obs = super().step(action)

        # extract achieved goal
        self._extract_achieved_goal(obs[0])  # obs[O] is the actual observation

        # return
        return {
            "observation": obs,
            "achieved_goal": self._achieved_goal.to_numpy().copy(),
            "desired_goal": self._desired_goal.to_numpy().copy(),
        }

    def _sample_goal(self) -> GoalState:
        return GoalState(0.0, 0.0)

    def _extract_achieved_goal(self, observation: UnityObservation):
        vector_obs = observation[-1]
        print(f"vector_obs = {vector_obs}")
        # these positions are hardcoded in Unity, cf Agent Script to get their values
        agentX = vector_obs[3]
        agentZ = vector_obs[5]
        self._achieved_goal = GoalState(agentX, agentZ)


channel = EnvironmentParametersChannel()  # create a params sidechannel
# None -> live interaction

unity_env = UnityEnvironment(file_name=None, seed=1, side_channels=[channel])

env = UnityToGoalGymWrapper(unity_env, channel, uint8_visual=True, allow_multiple_obs=True)

print(env.observation_space)
while True:
    obs = env.reset(GoalState(3.0, 3.0))

    for i in range(1000):
        obs_dict = env.step([0.01, 0.01])
        obs, reward, done, info = obs_dict["observation"]
        if i == 0:
            print(obs_dict)
        if done:
            break

import abc
from typing import Dict, List, Tuple, Union

import numpy as np
from gym import spaces
from gym_unity.envs import UnityToGymWrapper
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.environment_parameters_channel import EnvironmentParametersChannel

UnityObservation = Union[List[np.ndarray], np.ndarray]

GymObservation = Tuple[UnityObservation, float, bool, Dict]


class BaseAction(abc.ABC):
    """
    Container Base class for Gym Action
    """

    @abc.abstractmethod
    def to_list(self) -> List:
        pass


class BaseGoal(abc.ABC):
    """
    Container Base Class for Goal-Based Gym Goal

    attribute names should match the exact names
    that are used in unity for reading them from the Academy side channel

    e.g. Academy.instance.EnvironmentParameters.GetWithDefault(<key>,0) implies
    the attribute name in python is <key>

    No other attributes should be present in the class
    """

    @abc.abstractmethod
    def to_numpy(self) -> np.ndarray:
        pass

    def get_size(self):
        return self.to_numpy().shape

    def dict(self):
        return self.__dict__


class BaseObservation(abc.ABC):
    """
    Container Base Class for Gym Observation

    This Observation will be a subset of the full Unity Observation that contains
    the full state as vectorObservation and optional visual observations
    """

    @abc.abstractmethod
    def to_numpy(self) -> np.ndarray:
        pass

    def get_size(self):
        return self.to_numpy().shape


class BaseParams(abc.ABC):
    """
    Container class for external environment parameters
    attribute names should match the exact names
    that are used in unity for reading them from the Academy side channel

    e.g. Academy.instance.EnvironmentParameters.GetWithDefault(<key>,0) implies
    the attribute name in python is <key>

    No other attributes should be present in the class
    """

    def dict(self) -> Dict:
        return self.__dict__


class BaseUnityToGoalGymWrapper(UnityToGymWrapper, abc.ABC):
    """extend Unity Gym interface to work with Goal-based environments.
    This imposes certain structure on the observation:  the observation
    space is required to contain at least three elements, namely `observation`, `desired_goal`, and
    `achieved_goal`. Here, `desired_goal` specifies the goal that the agent should attempt to achieve.
    `achieved_goal` is the goal that it currently achieved instead. `observation` contains the
    actual observations of the environment as per usual.
    cf https://github.com/openai/gym/blob/c8a659369d98706b3c98b84b80a34a832bbdc6c0/gym/core.py#L163

    Furthermore the Unity environment is as such that the there is a vectorobservation,
    containing the relevant state of the environment and an optional number of visual observations

    """

    def __init__(
        self,
        unity_env: UnityEnvironment,
        channel: EnvironmentParametersChannel,
        uint8_visual: bool = False,
        flatten_branched: bool = False,
        allow_multiple_obs: bool = True,
    ) -> None:

        super().__init__(unity_env, uint8_visual, flatten_branched, allow_multiple_obs)

        self._channel = channel
        self._desired_goal: BaseGoal = self._sample_goal()
        self._achieved_goal: BaseGoal = self._sample_goal()
        self._goal_space = spaces.Box(-np.inf, np.inf, shape=self._desired_goal.get_size(), dtype="float32")
        self._observation_space: spaces.Dict = spaces.Dict(
            {
                "desired_goal": self._goal_space,
                "achieved_goal": self._goal_space,
                "observation": self._observation_space,  # unity determines unity_observation_space, which is used here,
                # however this is not the actual observation space
                # TODO -> change to the actual observation space
            }
        )

    def reset(self, goal: BaseGoal, parameters: BaseParams = None) -> BaseObservation:
        self._desired_goal = goal

        # set goal parameters before reset which triggers onEpisodeStart in Unity
        for key, value in goal.dict().items():
            self._channel.set_float_parameter(key, value)
        # set env parameters before reset
        if parameters:
            for key, value in parameters.dict().items():
                self._channel.set_float_parameter(key, value)

        unity_observation = super().reset()
        return self._generate_observation(unity_observation)

    def step(self, action: BaseAction) -> dict:
        unity_gym_observation: GymObservation = super().step(action.to_list())
        unity_observation = unity_gym_observation[0]  # first element is the actual observation
        # extract achieved goal
        self._set_achieved_goal(unity_observation)

        # replace the unity observation by the agent's observation
        observation = self._generate_observation(unity_observation)
        gym_observation = (observation,) + unity_gym_observation[1:]

        return {
            "observation": gym_observation,
            "achieved_goal": self._achieved_goal.to_numpy().copy(),
            "desired_goal": self._desired_goal.to_numpy().copy(),
        }

    @abc.abstractmethod
    def _sample_goal(self) -> BaseGoal:
        """
        creates an instance of the Goal Class for the environment
        """

    @abc.abstractmethod
    def _set_achieved_goal(self, observation: UnityObservation) -> None:
        """
        takes the complete Unity observation consisting of optional visual input
        and the vectorobservation. extracts the achieved goal, corresponding to the
        Goal Class for the environment. assigns this newly created instance to the _achieved_goal attr
        """

    @abc.abstractmethod
    def _generate_observation(self, observation: UnityObservation) -> BaseObservation:
        """
        takes the complete unity observation consisting of optional visual input
        and the vector Observation that represents the full state. creates an
        Observation as defined for the environment
        """

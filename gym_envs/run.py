#
from env_1 import Action, Goal, UnityToGoalGymWrapper
from env_2 import Action, Goal, UnityToGoalGymWrapper
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.environment_parameters_channel import EnvironmentParametersChannel

if __name__ == "__main__":
    """
    demo interaction with the gym interface using live Unity environment
    """
    channel = EnvironmentParametersChannel()  # create a params sidechannel
    # None -> live interaction = blocking call to unity
    unity_env = UnityEnvironment(file_name=None, seed=1, side_channels=[channel])

    env = UnityToGoalGymWrapper(unity_env, channel, uint8_visual=True, allow_multiple_obs=True)

    while True:
        obs = env.reset(Goal(3.0, 3.0))

        for i in range(1000):
            obs_dict = env.step(Action(0.01, 0.01))
            gym_observation = obs_dict["observation"]
            observation, reward, done, info = gym_observation
            print(observation)
            print(f"reward = {reward}")

            if done:
                break

from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.environment_parameters_channel import EnvironmentParametersChannel

from gym_unity.envs import UnityToGymWrapper

channel = EnvironmentParametersChannel() ## create a params sidechannel
from gym.core import GoalEnv
# None -> live interaction

unity_env = UnityEnvironment(file_name = None,seed = 1,side_channels=[channel])

env = UnityToGymWrapper(unity_env,uint8_visual=True,allow_multiple_obs = True)

print(env.observation_space)
while(True):
    channel.set_float_parameter("target_x", 3.0)
    obs = env.reset()

    for i in range(1000):
        obs,reward,done,info = env.step([0.01,0.01])
        if i == 0:
            print(obs)
            print(reward)
            print(info)
        if done:
            break








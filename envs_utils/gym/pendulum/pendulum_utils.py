import numpy as np
from gym.spaces import Box
from envs_utils.gym.pendulum.pendulum_configs import env_config

def inverted_pendulum_customize(env):
    # Settings

    # env.env.max_torque = max_torque  # you could also used env.unwrapped.max_torque
    env.unwrapped.max_torque = env_config.max_torque
    env.unwrapped.max_speed = env_config.max_speed  # you could also used env.unwrapped.max_speed
    env.unwrapped.dt = env_config.timestep

    env.action_space = Box(
        low=-env_config.max_torque,
        high=env_config.max_torque, shape=(1,),
        dtype=np.float32
    )
    high = np.array([1., 1., env_config.max_speed], dtype=np.float32)
    env.observation_space = Box(
        low=-high,
        high=high,
        dtype=np.float32
    )

    return env



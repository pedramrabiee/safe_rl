import numpy as np
from utils.misc import scaler

def inverted_pendulum_reward_gen(bounds):

    def inverted_pendulum_reward_func(obs, ac):

        obs = scaler(obs,
                     lim_from=(bounds.new.obs.low, bounds.new.obs.high),
                     lim_to=(bounds.old.obs.low, bounds.old.obs.high))
        ac = scaler(ac,
                     lim_from=(bounds.new.ac.low, bounds.new.ac.high),
                     lim_to=(bounds.old.ac.low, bounds.old.ac.high))
        cos_theta = obs[:, 0]
        sin_theta = obs[:, 1]
        theta_dot = obs[:, 2].reshape(-1, 1)

        theta = np.arctan2(sin_theta, cos_theta).reshape(-1, 1)

        return -(theta ** 2 + 0.1 * theta_dot ** 2 + 0.001 * ac ** 2)

    return inverted_pendulum_reward_func

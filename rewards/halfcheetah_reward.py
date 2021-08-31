import numpy as np
from utils.misc import scaler

def halfcheetah_reward_gen(bounds):
    forward_reward_weight = 1.0
    ctrl_cost_weight = 0.1
    def halfcheetah_reward_func(obs, ac):
        control_cost = ctrl_cost_weight * np.sum(np.square(ac))

        obs = scaler(obs,
                     lim_from=(bounds.new.obs.low, bounds.new.obs.high),
                     lim_to=(bounds.old.obs.low, bounds.old.obs.high))
        ac = scaler(ac,
                     lim_from=(bounds.new.ac.low, bounds.new.ac.high),
                     lim_to=(bounds.old.ac.low, bounds.old.ac.high))
        # cos_theta = obs[:, 0]
        # sin_theta = obs[:, 1]
        # theta_dot = obs[:, 2].reshape(-1, 1)
        #
        # theta = np.arctan2(sin_theta, cos_theta).reshape(-1, 1)
        #
        # return -(theta ** 2 + 0.1 * theta_dot ** 2 + 0.001 * ac ** 2)

        x_position_before = self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        x_position_after = self.sim.data.qpos[0]
        x_velocity = ((x_position_after - x_position_before)
                      / self.dt)

        ctrl_cost = self.control_cost(action)

        forward_reward = self._forward_reward_weight * x_velocity

    return halfcheetah_reward_func

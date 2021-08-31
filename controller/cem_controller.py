from controller.base_controller import BaseController
from utils.misc import discount_cumsum
import numpy as np
from attrdict import AttrDict
from scipy.stats import truncnorm


class CEMController(BaseController):
    def initialize(self, params, init_dict):
        self.horizon = init_dict.horizon
        self.num_particles = init_dict.num_particles
        self.gamma = init_dict.gamma
        self.lr = init_dict.lr
        self.num_elites = int(init_dict.elites_fraction * self.num_particles)
        self.cem_itr = init_dict.cem_itr
        self.var_threshold = init_dict.var_threshold
        # reset the controller
        self.reset()

    def reset(self):
        self._ac_rolling_mean = np.zeros([self.horizon, self.ac_dim]) +\
                                (self.bounds.new.ac.low + self.bounds.new.ac.high) / 2


    def act(self, obs, stats):
        mean = self._ac_rolling_mean
        var = np.zeros([self.horizon, self.ac_dim]) + np.square(self.bounds.new.ac.low - self.bounds.new.ac.high) / 16
        self.ac_base_dist = truncnorm(2 * self.bounds.new.ac.low, 2 * self.bounds.new.ac.high,
                                      loc=np.zeros_like(mean), scale=np.ones_like(mean))

        itr = 0
        while itr < self.cem_itr and np.max(var) > self.var_threshold:
            experiences = self.dream(obs=obs, horizon=self.horizon,
                                     num_particles=self.num_particles, stats=stats,
                                     control_dict=AttrDict(mean=mean, var=var),
                                     get_entire_ac_seq=True)
            returns = discount_cumsum(experiences.reward, discount=self.gamma, return_first=True)
            if self.ensemble_size is None:
                elites = experiences.action[np.argsort(returns, axis=0).squeeze()][-self.num_elites:]
            else:
                elites = experiences.action[np.argsort(returns.mean(axis=0), axis=0).squeeze()][-self.num_elites:]

            mean = self.lr * mean + (1 - self.lr) * np.mean(elites, axis=0)
            var = self.lr * var + (1 - self.lr) * np.var(elites, axis=0)

            itr += 1
        self._ac_rolling_mean[:-1] = mean[1:]

        return np.clip(mean[0, :], self.bounds.new.ac.low, self.bounds.new.ac.high)

    def _control(self, obs, control_dict=None):
        mean = control_dict.mean
        lb_dist, ub_dist = mean - self.bounds.new.ac.low, self.bounds.new.ac.high - mean
        var = control_dict.var
        constrained_var = np.minimum(np.minimum(np.square(lb_dist / 2),
                                                np.square(ub_dist / 2)), var)
        ac = self.ac_base_dist.rvs(size=[self.num_particles, self.horizon, self.ac_dim]) * np.sqrt(constrained_var) + mean
        return ac

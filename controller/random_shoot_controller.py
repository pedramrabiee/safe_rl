from controller.base_controller import BaseController
import numpy as np
from utils.misc import discount_cumsum
from utils.seed import rng


class RandomShootController(BaseController):

    def initialize(self, params, init_dict):
        self.horizon = init_dict.horizon
        self.num_particles = init_dict.num_particles
        self.gamma = init_dict.gamma

    def act(self, obs, stats):
        experiences = self.dream(obs=obs, horizon=self.horizon, num_particles=self.num_particles, stats=stats)
        returns = discount_cumsum(experiences.reward, discount=self.gamma, return_first=True)
        if self.ensemble_size is None:
            best_sequence = np.argmax(returns)              # find the sequence that resulted in highest return in the horizon
        else:
            best_sequence = np.argmax(returns.mean(axis=0))
        return experiences.action[best_sequence, 0, :]  # return first action from the best sequence of actions

    def _control(self, obs, control_dict=None):
        if self.ensemble_size is None:
            return rng.uniform(low=self.bounds.new.ac.low, high=self.bounds.new.ac.high,
                               size=(self.num_particles, self.ac_dim))
        else:
            return rng.uniform(low=self.bounds.new.ac.low, high=self.bounds.new.ac.high,
                               size=(self.num_particles, self.ac_dim))

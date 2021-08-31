from controller.base_controller import BaseController
import numpy as np

class RandomController(BaseController):
    def act(self, obs, stats):
        return np.random.uniform(low=self.bounds.new.ac.low, high=self.bounds.new.ac.high, size=(self.ac_dim))
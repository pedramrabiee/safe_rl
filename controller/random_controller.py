from controller.base_controller import BaseController
from utils.seed import rng

class RandomController(BaseController):
    def act(self, obs, stats):
        return rng.uniform(low=self.bounds.new.ac.low, high=self.bounds.new.ac.high, size=(self.ac_dim))
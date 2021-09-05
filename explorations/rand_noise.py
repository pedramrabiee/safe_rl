import numpy as np
from utils.seed import rng

class RandNoise:
    def __init__(self, ac_dim, dist, dist_kwargs):
        """
        :param ac_dim: action dimension
        :param dist: distribution of choice for random noise among normal and uniform  
        :param dist_kwargs: distribution args: for normal dist : loc, stale, for uniform dist: low, high
        """
        self.ac_dim = ac_dim
        self.dist = dist
        self.dist_kwargs = dist_kwargs

    def noise(self):
        if self.dist == 'normal':
            return rng.normal(**self.dist_kwargs, size=self.ac_dim)
        elif self.dist == 'uniform':
            return rng.uniform(**self.dist_kwargs, size=self.ac_dim)
        elif self.dist == 'ignore':
            return np.zeros(self.ac_dim, dtype=np.float32)


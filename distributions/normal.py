import torch
from distributions.base_dist_ext import DistributionExt
from torch.distributions.normal import Normal

class NormalExt(Normal, DistributionExt):
    def kl_divergence(self, other):
        pass

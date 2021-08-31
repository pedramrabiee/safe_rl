from torch.distributions import Distribution

class DistributionExt(Distribution):
    def kl_divergence(self, other):
        raise NotImplementedError

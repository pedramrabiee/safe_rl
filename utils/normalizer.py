import numpy as np
import torch
from attrdict import AttrDict


_KEYS = ['obs', 'ac', 'next_obs', 'delta_obs']


def init_stats(obs_dim, ac_dim):
    return AttrDict(obs=AttrDict(mean=np.zeros(obs_dim, dtype=np.float32),
                             std=np.ones(obs_dim, dtype=np.float32)),
                  ac=AttrDict(mean=np.zeros(ac_dim, dtype=np.float32),
                            std=np.ones(ac_dim, dtype=np.float32)),
                  next_obs=AttrDict(mean=np.zeros(obs_dim, dtype=np.float32),
                                  std=np.ones(obs_dim, dtype=np.float32)),
                  delta_obs=AttrDict(mean=np.zeros(obs_dim, dtype=np.float32),
                                   std=np.ones(obs_dim, dtype=np.float32)))


def update_stats(normalizer_stats, samples):
    for key in normalizer_stats:
        normalizer_stats[key]['mean'] = torch.mean(samples[key], dim=0).detach()
        normalizer_stats[key]['std'] = torch.std(samples[key], dim=0).detach()
    return normalizer_stats

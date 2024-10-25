import torch.nn as nn
import torch
import torch.nn.functional as F
from networks.mlp import MLPNetwork
from torch.distributions.normal import Normal
import numpy as np


class GaussianMLP(MLPNetwork):
    def __init__(self, in_dim, out_dim,
                 hidden_dim=64, num_layers=2,
                 unit_activation=nn.ReLU, out_activation=nn.Identity,
                 batch_norm=False, layer_norm=False, batch_norm_first_layer=False,
                 out_layer_initialize_small=False, last_layer_normed=False):

        super(GaussianMLP, self).__init__(in_dim=in_dim, out_dim=out_dim * 2,
                                          hidden_dim=hidden_dim, num_layers=num_layers,
                                          unit_activation=unit_activation, out_activation=out_activation,
                                          batch_norm=batch_norm, layer_norm=layer_norm,
                                          batch_norm_first_layer=batch_norm_first_layer,
                                          out_layer_initialize_small=out_layer_initialize_small,
                                          last_layer_normed=last_layer_normed)
        self.out_dim = out_dim

    def forward(self, x, ret_logvar=False):
        mu, log_var = torch.split(self.pipeline(x), self.out_dim, dim=-1)
        if ret_logvar:
            return mu, log_var
        else:
            var = torch.exp(log_var)
            var = F.softplus(var) + 1e-6
            return mu, var

LOG_STD_MIN = -20
LOG_STD_MAX = 2
class SquashedGaussianMLP(MLPNetwork):
    def __init__(self, in_dim, out_dim,
                 hidden_dim=64, num_layers=2,
                 unit_activation=nn.ReLU, out_activation=nn.Identity,
                 batch_norm=False, layer_norm=False, batch_norm_first_layer=False,
                 out_layer_initialize_small=False, last_layer_normed=False):

        super(SquashedGaussianMLP, self).__init__(in_dim=in_dim, out_dim=out_dim * 2,
                                                  hidden_dim=hidden_dim, num_layers=num_layers,
                                                  unit_activation=unit_activation, out_activation=out_activation,
                                                  batch_norm=batch_norm, layer_norm=layer_norm,
                                                  batch_norm_first_layer=batch_norm_first_layer,
                                                  out_layer_initialize_small=out_layer_initialize_small,
                                                  last_layer_normed=last_layer_normed)
        self.out_dim = out_dim

    def forward(self, x, deterministic=False, with_log_prob=True):
        mu, log_std = torch.split(self.pipeline(x), self.out_dim, dim=-1)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)

        res_dist = Normal(mu, std)
        if deterministic:
            res = mu
        else:
            res = res_dist.rsample()

        log_res = None
        if with_log_prob:
            log_res = res_dist.log_prob(res).sum(axis=-1)
            log_res -= (2*(np.log(2) - res - F.softplus(-2*res))).sum(axis=-1)
            log_res.unsqueeze_(-1)

        res = torch.tanh(res)

        return res if not with_log_prob else (res, log_res)

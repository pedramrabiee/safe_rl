import torch.nn as nn
import torch
import torch.nn.functional as F
from networks.mlp import MLPNetwork


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

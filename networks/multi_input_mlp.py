import torch
import torch.nn as nn
from networks.mlp import MLPNetwork
#
#
class MultiInputMLP(nn.Module):
    def __init__(self, in1_dim, in2_dim, out_dim, in2_cat_layer=1,
                 hidden_dim=64, num_layers=2,
                 unit_activation=nn.ReLU, out_activation=nn.Identity,
                 batch_norm=False, layer_norm=False, batch_norm_first_layer=False,
                 out_layer_initialize_small=False):

        super(MultiInputMLP, self).__init__()
        self.in1_dim = in1_dim
        self.in2_dim= in2_dim

        # before 2nd input concatenation
        net1_num_layers = in2_cat_layer - 1
        self.net1 = MLPNetwork(in_dim=in1_dim, out_dim=hidden_dim,
                               hidden_dim=hidden_dim, num_layers=net1_num_layers,
                               unit_activation=unit_activation, out_activation=unit_activation,
                               batch_norm=batch_norm, layer_norm=layer_norm,
                               batch_norm_first_layer=batch_norm_first_layer,
                               out_layer_initialize_small=False,
                               last_layer_normed=layer_norm or batch_norm)

        # after 2nd input concatenation
        net2_num_layers = num_layers - in2_cat_layer
        self.net2 = MLPNetwork(in_dim=hidden_dim+in2_dim, out_dim=out_dim,
                               hidden_dim=hidden_dim, num_layers=net2_num_layers,
                               unit_activation=unit_activation, out_activation=out_activation,
                               batch_norm=batch_norm, layer_norm=layer_norm,
                               batch_norm_first_layer=False,
                               out_layer_initialize_small=out_layer_initialize_small)

    def forward(self, x):
        x1, x2 = torch.split(x, [self.in1_dim, self.in2_dim], dim=-1)
        x = self.net1(x1)
        x = torch.cat((x, x2), dim=-1)
        x = self.net2(x)
        return x

import pdb

import torch.nn as nn


class MLPNetwork(nn.Module):
    def __init__(self, in_dim, out_dim,
                 hidden_dim=64, num_layers=2,
                 unit_activation=nn.ReLU, out_activation=nn.Identity,
                 batch_norm=False, layer_norm=False, batch_norm_first_layer=False,
                 out_layer_initialize_small=False, last_layer_normed=False):

        super(MLPNetwork, self).__init__()

        assert not (batch_norm and layer_norm)      # raise error if batch_norm and layer_norm are True at the same time

        if num_layers == 0:
            layers = []
            if batch_norm_first_layer:
                layers.append(nn.BatchNorm1d(in_dim))
            layers.append(nn.Linear(in_dim, out_dim))
            if layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(out_activation())
        else:
            fc = []
            act = []
            # input layer
            fc.append(nn.Linear(in_dim, hidden_dim))
            act.append(unit_activation())

            # hidden layers
            for _ in range(num_layers - 1):
                fc.append(nn.Linear(hidden_dim, hidden_dim))
                act.append(unit_activation())

            # output layer
            fc.append(nn.Linear(hidden_dim, out_dim))
            act.append(out_activation())

            # initialize output layer small to prevent saturation
            # FIXME: Fix this for discrete action
            if out_layer_initialize_small:
                fc[-1].weight.data.mul_(0.1)
                fc[-1].bias.data.mul_(0.1)

            # setup layer norms
            if layer_norm:
                ln = [nn.LayerNorm(hidden_dim) for _ in range(num_layers+1)]
            else:
                ln = [nn.Identity() for _ in range(num_layers+1)]

            # setup batch norms
            if batch_norm:
                bn = [nn.BatchNorm1d(hidden_dim) for _ in range(num_layers+1)]
            else:
                bn = [nn.Identity() for _ in range(num_layers+1)]

            # setup first layer batch norm
            if batch_norm_first_layer:
                bn_first_layer = nn.BatchNorm1d(in_dim)
            else:
                bn_first_layer = nn.Identity()

            # make the layers
            layers = [bn_first_layer]
            for i in range(num_layers):
                layers += [fc[i], ln[i], bn[i], act[i]]

            # remove unneeded Identity() layers
            layers = list(filter(lambda x: not isinstance(x, nn.Identity), layers))

            # make the output layer
            if last_layer_normed:   # has usage in MultiInputMLP
                layers += [fc[-1], ln[-1], bn[-1]]
                layers = list(filter(lambda x: not isinstance(x, nn.Identity), layers))
                layers += [act[-1]]
            else:
                layers += [fc[-1], act[-1]]

        self.pipeline = nn.Sequential(*layers)


    def forward(self, x):
        return self.pipeline(x)

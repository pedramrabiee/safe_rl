import torch
import torch.nn as nn
from networks.gaussian_processes import GP


class AffineInActionDynBase(nn.Module):
    def __init__(self, base_net_cls, obs_dim, ac_dim, net_kwargs=None):
        super(AffineInActionDynBase, self).__init__()
        self.obs_dim = obs_dim
        self.ac_dim = ac_dim
        self.out_dim = obs_dim * (1 + ac_dim) # obs_dim (for f) + obs_dim * ac_dim (for G)
        self.pipeline = base_net_cls(in_dim=obs_dim, out_dim=self.out_dim, **net_kwargs)

    def forward(self, x, split_return=False):
        raise NotImplementedError


class AffineInActionDeterministic(AffineInActionDynBase):
    def forward(self, x, split_return=False):
        obs, ac = torch.split(x, self.obs_dim, dim=-1)
        f, g = torch.split(self.pipeline(obs), self.obs_dim, dim=-1)
        g = g.view(-1, self.obs_dim, self.ac_dim)
        if split_return:
            return f, g
        else:
            return f + torch.matmul(g, torch.unsqueeze(ac, dim=-1)).squeeze()


class AffineInActionGaussian(AffineInActionDynBase):
    # def forward(self, x, split_return=False, ret_logvar=False):
    def forward(self, x, ret_logvar=False):
        # obs, ac = torch.split(x, self.obs_dim, dim=-1)
        obs = x     #TODO: clean this
        mu, var = self.pipeline(obs)

        mu_f, mu_g = torch.split(mu, [self.obs_dim, self.obs_dim * self.ac_dim], dim=-1)
        var_f, var_g = torch.split(var, [self.obs_dim, self.obs_dim * self.ac_dim], dim=-1)

        # if split_return:
        if ret_logvar:
            return mu_f, mu_g, torch.log(var_f), torch.log(var_g)
        return mu_f, mu_g, torch.sqrt(var_f), torch.sqrt(var_g)
        # else:
            # mu = mu_f + torch.matmul(mu_g, torch.unsqueeze(ac, dim=-1)).squeeze()
            # var = var_f + torch.matmul(var_g, torch.unsqueeze(ac.pow(2), dim=-1)).squeeze()
            # return mu, torch.log(var) if ret_logvar else var


class AffineInActionGP(GP):
    def __init__(self, obs_dim, ac_dim, net_kwargs=None):
        self.obs_dim = obs_dim
        self.ac_dim = ac_dim
        self.out_dim = obs_dim * (1 + ac_dim)

        super().__init__(in_dim=obs_dim, out_dim=self.out_dim, **net_kwargs)

    def forward(self, x, split_return=False):
        # TODO:  i removed ret_sample argument from this method and the forward method in GP since the forward method of the ExactGP can only return distribution
        obs, ac = torch.split(x, self.obs_dim, dim=-1)
        mvn = super().forward(obs)
        print(mvn)
        # you need the mvn before making it as multitakmultivariatenormal






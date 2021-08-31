from dynamics.gaussian_nn_dynamics import GaussianDynamics
from dynamics.deterministic_nn_dynamics import DeterministicDynamics
from dynamics.custom_dyns.affine_in_action import AffineInActionGaussian
from networks.gaussian_mlp import GaussianMLP
from torch.nn import Parameter
import torch
import torch.nn.functional as F
from utils.distribution_utils import bound_log_var
from utils.losses import nll_loss



class AffineInActionGaussianDynamics(GaussianDynamics):
    def initialize(self, params, init_dict=None):
        super(DeterministicDynamics, self).initialize(params, init_dict)
        self.model = AffineInActionGaussian(base_net_cls=GaussianMLP,
                                            obs_dim=self.obs_dim,
                                            ac_dim=self.ac_dim,
                                            net_kwargs=self.params.dynamics_net_kwargs)

        self.max_logvar_f = Parameter(0.5 * torch.ones(self.out_dim), requires_grad=True)
        self.min_logvar_f = Parameter(-10 * torch.ones(self.out_dim), requires_grad=True)
        self.max_logvar_g = Parameter(0.5 * torch.ones(self.out_dim * self.ac_dim), requires_grad=True)
        self.min_logvar_g = Parameter(-10 * torch.ones(self.out_dim * self.ac_dim), requires_grad=True)

        self.optimizer = self.params.optim_cls(self.model.parameters(), **self.params.optim_kwargs)

        self.optimizer.add_param_group({'params': self.min_logvar_f})
        self.optimizer.add_param_group({'params': self.max_logvar_f})
        self.optimizer.add_param_group({'params': self.min_logvar_g})
        self.optimizer.add_param_group({'params': self.max_logvar_g})

        self.extra_params = [self.max_logvar_f, self.max_logvar_g, self.min_logvar_f, self.min_logvar_g]
        self.extra_params_dict = dict(max_logvar_f=self.max_logvar_f,
                                      max_logvar_g=self.max_logvar_g,
                                      min_logvar_f=self.min_logvar_f,
                                      min_logvar_g=self.min_logvar_g)

    def _compute_output(self, inputs, split_return=False):
        obs, ac = torch.split(inputs, self.obs_dim, dim=-1)
        mu_f, mu_g, log_var_f, log_var_g = self.model(obs, ret_logvar=True)

        # bound log_var_f and log_var_g
        log_var_f = bound_log_var(log_var_f, self.max_logvar_f, self.min_logvar_f)
        log_var_g = bound_log_var(log_var_g, self.max_logvar_g, self.min_logvar_g)

        # convert log_var to var
        var_f = torch.exp(log_var_f)
        var_g = torch.exp(log_var_g)

        # reshape mu_g and var_g
        mu_g = mu_g.view(-1, self.obs_dim, self.ac_dim)
        var_g = var_g.view(-1, self.obs_dim, self.ac_dim)

        if split_return:
            return mu_f, mu_g, torch.sqrt(var_f), torch.sqrt(var_g)     # FIXME: return var instead of std to be consistent everywhere

        mu = mu_f + torch.matmul(mu_g, torch.unsqueeze(ac, dim=-1)).squeeze()
        var = var_f + torch.matmul(var_g, torch.unsqueeze(ac.pow(2), dim=-1)).squeeze()
        return mu, var

    def _compute_loss(self, inputs, targets):
        loss = nll_loss(targets, *self._compute_output(inputs))
        loss += 0.01 * (self.max_logvar_f.sum() + self.max_logvar_g.sum() - self.min_logvar_f.sum() - self.min_logvar_g.sum())
        return loss





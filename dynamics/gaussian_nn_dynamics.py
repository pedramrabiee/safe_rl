from dynamics.base_dynamics import BaseDynamics
from dynamics.deterministic_nn_dynamics import DeterministicDynamics
from networks.gaussian_mlp import GaussianMLP
import torch
from torch.nn import Parameter
from utils.misc import normalize, unnormalize
from utils.losses import nll_loss
import torch.nn.functional as F
from utils.distribution_utils import bound_log_var



class GaussianDynamics(DeterministicDynamics, BaseDynamics):
    def initialize(self, params, init_dict=None):
        super(DeterministicDynamics, self).initialize(params, init_dict)
        # if self.use_custom_dyn:
        #     self.model = self.params.custom_dyn_cls(base_net_cls=GaussianMLP,
        #                                             obs_dim=self.obs_dim,
        #                                             ac_dim=self.ac_dim,
        #                                             net_kwargs=self.params.dynamics_net_kwargs)
        # else:
        self.model = GaussianMLP(in_dim=self.obs_dim+self.ac_dim,
                                 out_dim=self.out_dim,
                                 **self.params.dynamics_net_kwargs)

        self.max_logvar = Parameter(0.5 * torch.ones(self.out_dim), requires_grad=True)
        self.min_logvar = Parameter(-10 * torch.ones(self.out_dim), requires_grad=True)

        self.optimizer = self.params.optim_cls(self.model.parameters(), **self.params.optim_kwargs)

        self.optimizer.add_param_group({'params': self.min_logvar})
        self.optimizer.add_param_group({'params': self.max_logvar})

        self.extra_params = [self.max_logvar, self.min_logvar]
        self.extra_params_dict = dict(max_logvar=self.max_logvar,
                                      min_logvar=self.min_logvar)

    @torch.no_grad()
    def _predict(self, obs, ac, stats, pred_dict=None, split_return=False):
        if split_return:
            dyn_bd = self._compute_output(torch.cat((obs, ac), dim=-1), split_return=True)
            return [x.detach().numpy() for x in dyn_bd]

        if self.params.normalized_io:
            obs_norm = normalize(data=obs, **stats.obs)
            ac_norm = normalize(data=ac, **stats.ac)
            delta_mu_norm, delta_var = self._compute_output(torch.cat((obs_norm, ac_norm), dim=-1))
            delta_mu_unnorm = unnormalize(data=delta_mu_norm, **stats.delta_obs)
            delta_sampled = delta_mu_unnorm + torch.normal(mean=0, std=1, size=delta_mu_unnorm.shape) * torch.sqrt(delta_var)
        else:
            delta_mu, delta_var = self._compute_output(torch.cat((obs, ac), dim=-1))
            delta_sampled = delta_mu + torch.normal(mean=0, std=1, size=delta_mu.shape) * torch.sqrt(delta_var)
        return delta_sampled.detach().numpy()

    def _compute_loss(self, inputs, targets):
        loss = nll_loss(targets, *self._compute_output(inputs))
        loss += 0.01 * (self.max_logvar.sum() - self.min_logvar.sum())
        return loss

    def _compute_output(self, inputs, split_return=False):
        mu, log_var = self.model(inputs, ret_logvar=True)
        log_var = bound_log_var(log_var, self.max_logvar, self.min_logvar)

        return mu, torch.exp(log_var)

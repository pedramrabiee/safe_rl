from shields.base_shield import BaseSheild
import torch
from torchdiffeq import odeint
from utils.torch_utils import softmin, softmax
from torch.autograd import grad
from utils.optim import make_box_constraints_from_bounds, solve_lp
from utils.cbf_utils import min_intervention_qp_box_constrained
import numpy as np
from dynamics.torch_dynamics import get_torch_dyn
import importlib
from attrdict import AttrDict

class BackupShield(BaseSheild):
    """
    Soft-Minimum and Soft-Maximum Barrier Functions Based on https://arxiv.org/abs/2305.10620
    """

    def initialize(self, params, init_dict=None):
        # TODO: Implement obs_dim
        self._obs_dim = self.obs_proc.obs_dim_processed()
        self.params = params
        self.dynamics = init_dict.dynamics_cls()
        self.backup_sets = init_dict.backup_sets
        self.backup_policies = init_dict.backup_policies
        self.safe_set = init_dict.safe_set
        # TODO: check if self.set and self.backup_sets are instance of SafeSetFromBarrieFunc
        assert len(self.backup_policies) == len(self.backup_sets),\
            'The number of backup sets does not match the number of backup policies'
        self._num_backup = len(self.backup_policies)
        self._backup_t_seq = params.backup_t_seq
        self._ac_bounds = _from_ac_lim_to_bounds_array(self._ac_lim)

    @torch.enable_grad()
    def shield(self, obs, ac, filter_dict=None):
        obs = torch.from_numpy(obs)

        h, h_grad, h_values, h_min_values, h_argmax = self._get_softmax_softmin_backup_h_grad_h(obs)
        # FIXME: MAKE ALL THE NUMPY TORCH CONVERSION EFFICIENT
        h = h.detach().numpy()

        u_b_select = self.backup_policies[h_argmax](obs) if h_min_values[h_argmax] <= 0 \
            else self._get_backup_blend(obs, h_min_values)
        u_b_select = u_b_select.detach().numpy()
        # FIXME: make qp problem formation smoother
        Lfh = np.atleast_1d(torch.dot(h_grad, self.dynamics.f(obs)).detach().numpy())
        Lgh = np.atleast_2d(torch.dot(h_grad, self.dynamics.g(obs)).detach().numpy())
        feas_fact = self._get_feasibility_factor(Lfh, Lgh, h)

        gamma = min((h - self.params.eps_buffer) / self.params.h_scale, feas_fact / self.params.feas_scale)
        if gamma <= 0:
            u = u_b_select
        else:
            # TODO: make sure ac_lim is in correct standard
            u, _ = min_intervention_qp_box_constrained(h=h - self.params.eps_buffer,
                                                       Lfh=Lfh, Lgh=Lgh,
                                                       alpha_func=lambda eta: self.params.alpha * eta,
                                                       u_des=ac,
                                                       u_bound=self._ac_bounds)

        beta = (1 if gamma >= 1 else gamma) if gamma > 0 else 0
        u = (1 - beta) * u_b_select + beta * u
        self.custom_plotter.filter_push_action((ac, u))
        return u

    def get_h(self, obs):
        obs = torch.from_numpy(obs).squeeze()
        return self._get_h(self._get_trajs(obs))

    def _get_softmax_softmin_backup_h_grad_h(self, obs):
        # TODO: obs probably is numpy array, you have to convert it
        obs.requires_grad_()
        # TODO: Currently, this doesn't support different t_seq corresponding to different policies. In order to do that,
        #  we have two options:
        #  1. If the timesteps are the same, forward propagate upto the max horizon, then trim each
        #  trajectory based on their required horizon.
        #  2. Do not stack dynamics, and deal with each fwd. prop. separately.
        trajs = self._get_trajs(obs)
        h, h_grad, h_values, h_min_values, h_argmax = self._get_h_grad_h(obs, trajs)
        # TODO: Convert obs back to numpy array if required
        obs.requires_grad_(requires_grad=False)
        return h, h_grad, h_values, h_min_values, h_argmax


    def _get_feasibility_factor(self, Lfh, Lgh, h):
        # TODO: you have to make sure _ac_lim is the format that make_box_constraints_from_bounds needs
        A_u, b_u = make_box_constraints_from_bounds(self._ac_bounds)
        u, optval = solve_lp(-Lgh, A_u, b_u)
        Lghu_max = -optval
        return Lfh[0] + Lghu_max + self.params.alpha * (h - self.params.eps_buffer)


    def _get_backup_blend(self, obs, h_min_values):
        # FIXME: make this function efficient
        mask = h_min_values >= self.params.eps_buffer
        u_b_valid = np.array(self.backup_policies)[mask.detach().numpy()].tolist()
        return torch.sum((h_min_values[mask] - self.params.eps_buffer) * torch.stack(
            [policy(obs) for policy in u_b_valid])) / torch.sum(h_min_values[mask] - self.params.eps_buffer)


    def _get_trajs(self, obs):
        trajs = odeint(
            lambda t, y: torch.cat([self.dynamics.dynamics(yy, policy(yy))
                                    for yy, policy in zip(y.split(self._obs_dim), self.backup_policies)], dim=0),
            obs.repeat(self._num_backup), self._backup_t_seq).split(self._obs_dim, dim=1)

        return trajs

    def _get_h_grad_h(self, obs, trajs):
        h, h_values, h_min_values = self._get_h(trajs)
        h_argmax = torch.argmax(h_min_values)
        h_argmax = h_argmax.item() if h_argmax.dim() == 0 else h_argmax[0].item()
        h_grad = grad(h, obs)[0]
        return h, h_grad, h_values, h_min_values, h_argmax


    def _get_h(self, trajs):
        h_list = [torch.cat((self.safe_set.des_safe_barrier(traj), backup_set.safe_barrier(traj[-1, :]).unsqueeze(0)))
                  for traj, backup_set in zip(trajs, self.backup_sets)]
        h_values = torch.stack([softmin(hh, self.params.softmin_gain) for hh in h_list])
        h_min_values = torch.stack([torch.min(hh) for hh in h_list])
        h = softmax(h_values, self.params.softmax_gain)
        return h, h_values, h_min_values


def get_backup_shield_info_from_env(env, env_info, obs_proc):
    return dict(dynamics_cls=get_torch_dyn(env_info),
                **get_backup_prerequisites(env=env, env_info=env_info, obs_proc=obs_proc))

def get_backup_prerequisites(env, env_info, obs_proc):
    # FIXME: Environment itself should be passed here, not the environment info
    module_name = _get_module_name(env_info)

    try:
        module = importlib.import_module(module_name)
        backup_sets_func = getattr(module, 'get_backup_sets')
        backup_policies_func = getattr(module, 'get_backup_policies')
        safe_set_func = getattr(module, 'get_safe_set')
        return dict(backup_sets=backup_sets_func(env=env, obs_proc=obs_proc),
                    backup_policies=backup_policies_func(),
                    safe_set=safe_set_func(env=env, obs_proc=obs_proc),
                    )
    except ImportError:
        # Handle cases where the module or class is not found
        raise ImportError('Module is not found')
    except AttributeError:
        # Handle cases where the class is not found in the module
        raise AttributeError('Method is not found')


def get_desired_policy(env_info):
    module_name = _get_module_name(env_info)
    try:
        module = importlib.import_module(module_name)
        return getattr(module, 'desired_policy')
    except ImportError:
        # Handle cases where the module or class is not found
        raise ImportError('Module is not found')
    except AttributeError:
        # Handle cases where the class is not found in the module
        raise AttributeError('Method is not found')


def _get_module_name(env_info):
    env_collection = env_info['env_collection']
    nickname = env_info['env_nickname']

    # Construct the module
    return f'envs_utils.{env_collection}.{nickname}.{nickname}_backup_shield'



def _from_ac_lim_to_bounds_array(ac_lim):
    # TODO: Fix this for multi-dimension action space. Start from agent factory where you populate ac_lim
    bounds = [[ac_lim['low'], ac_lim['high']]]
    return bounds

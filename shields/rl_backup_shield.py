import torch
from shields.backup_shield import BackupShield
from utils.torch_utils import softmax
from math import pi
from torchdiffeq import odeint, odeint_adjoint
from torch import nn
import numpy as np
from buffers.replay_buffer import ReplayBuffer
from logger import logger


class RLBackupShield(BackupShield):
    def initialize(self, params, init_dict=None):
        super().initialize(params, init_dict)
        self.rl_backup = init_dict.rl_backup

        self.models = [*self.rl_backup.models]
        self.optimizers = [*self.rl_backup.optimizers]
        self.models_dict = dict(rl_backup=self.rl_backup.models_dict)
        self.optimizers_dict = dict(rl_backup=self.rl_backup.optimizers_dict)

        self.agents = [self.rl_backup]
        #  Apppend backup sets with a set that is the softmax of all backup sets
        self.backup_sets.append(self.rl_backup_safe_set)
        #  Apppend backup policies with rl backup policy
        self.backup_policies.append(self.rl_backup_melt_into_backup_policies)
        # Increase number of backup policies by 1
        self._num_backup += 1
        # Make nn.Module dynamics for rl backup
        # TODO: Not sure if this is necessary. Check
        self.rl_backup_dyn = BackupDynamicsModule()
        self.rl_backup_dyn.init(self.dynamics, self.backup_policies[-1])
        assert isinstance(self.buffer, ReplayBuffer)

    def rl_backup_safe_set(self, obs):
        return softmax(torch.stack([backup_set.safe_set(obs) for backup_set in self.backup_sets]),
                       self.params.rl_backup_backup_set_softmax_gain)

    def rl_backup_melt_into_backup_policies(self, obs):
        """
           Perform RL backup by melting the contributions of backup policies based on a melt law.

           The function computes the values of backup policies, applies a melt law to determine
           their respective contributions, and combines them with the RL backup action.

           Parameters:
           - obs (torch.Tensor): Input tensor representing the observation.

           Returns:
           torch.Tensor: Melted RL backup result based on the contributions of backup policies.
           """
        backup_policies_vals = torch.vstack([policy(obs) for policy in self.backup_policies[:-1]])
        beta = self.melt_law(torch.stack([backup_set.safe_set(obs) for backup_set in self.backup_sets]))
        return (1 - torch.sum(beta)) * self.rl_backup.act(obs) + backup_policies_vals * beta.view(-1, 1)


    def melt_law(self, h):
        """
            Compute a function based on the given h values.

            The melt law is defined as follows:
            1                                   when h >= 0,
            0.5 * (1 - cos(pi * (a * h - 1)))   when -1/a < h < 0,
            0                                   when h <= -1/a.

            Parameters:
            - h (torch.Tensor): Input tensor containing h values.

            Returns:
            torch.Tensor: Output tensor containing computed values based on the law.
        """
        return torch.where(
            h >= 0,                     # Condition for the first case
            torch.tensor(1.0),          # Result if the condition is True
            0.5 * (1 - torch.cos(pi * (self.params.melt_law_gain * h - 1))) * (h > -1/self.params.melt_law_gain))

    def optimize_agent(self, samples, optim_dict=None):
        logger.log('Training RL Backup Agent...')
        rl_backup_loss = self.rl_backup.optimize_agent(samples, optim_dict)
        return {"RL_Backup_Loss": rl_backup_loss}

    def _get_softmax_softmin_backup_h_grad_h(self, obs):
        obs.requires_grad_()
        trajs = self._get_trajs(obs)
        h, h_grad, h_values, h_min_values, h_argmax = self._get_h_grad_h(obs, trajs)

        # Add rl-backup trajectory data to buffer
        traj_len = self._backup_t_seq.size()[0]
        acs = self.rl_backup(trajs[-1]).detach().numpy()
        rews = np.ones(traj_len) * h_values[-1].detach().numpy()
        next_obs = np.concatenate((trajs[-1][1:], self._fwd_prop_for_one_timestep(trajs[-1][-1])), axis=0)
        done = np.zeros(traj_len)
        done[-1] = 1
        trajs[-1].detach().numpy()
        self.push_to_buffer((trajs[-1], acs, rews,
                             next_obs, done, np.array([None] * traj_len)))

        # TODO: Convert obs back to numpy array if required
        obs.requires_grad_(requires_grad=False)
        return h, h_grad, h_values, h_min_values, h_argmax

    def _get_trajs(self, obs):
        # Find the trajectories under all backup policies except for the RL backup using odeint
        # TODO: Test if rl backup can be treated like the other backup policies or not?
        trajs = odeint(
            func=lambda t, y: torch.cat([self.dynamics(yy, policy(yy))
                                         for yy, policy in zip(y.split(self._obs_dim), self.backup_policies[:-1])],
                                        dim=0),
            y0=obs.repeat(self._num_backup),
            t=self._backup_t_seq
        ).split(self._obs_dim, dim=1)
        traj_rl = odeint_adjoint(func=self.rl_backup_dyn, y0=obs, t=self._backup_t_seq)
        # Append trajectories with the rl trajecory
        trajs += (traj_rl,)
        return trajs

    def _fwd_prop_for_one_timestep(self, last_obs):
        # Forward propagate for one time step to get the next observation for the last observation in the trajectory
        return odeint_adjoint(func=self.rl_backup_dyn, y0=last_obs, t=self._backup_t_seq[:2]).detach().numpy()


class BackupDynamicsModule(nn.Module):
    def init(self, dynamics, u_b):
        self.dynamics = dynamics
        self.u_b = u_b

    def forward(self, t, y):
        return self.dynamics(y, self.u_b(y))


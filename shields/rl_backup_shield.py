import torch
from shields.backup_shield import BackupShield
from utils.torch_utils import softmax
from math import pi
from torchdiffeq import odeint, odeint_adjoint
from torch import nn
import numpy as np
from buffers.replay_buffer import ReplayBuffer
from logger import logger
import torch.nn.functional as F
from utils.misc import torchify
from utils.safe_set import SafeSetFromBarrierFunction
from copy import copy

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
        self.rl_backup_safe_set = RLBackupBackupSet(env=None, obs_proc=self.obs_proc)
        self.rl_backup_safe_set.initialize(init_dict=dict(backup_sets=copy(self.backup_sets),
                                                          params=self.params))
        self.backup_sets.append(self.rl_backup_safe_set)
        #  Apppend backup policies with rl backup policy
        self.backup_policies.append(self.rl_backup_melt_into_backup_policies)
        # Increase number of backup policies by 1
        self._num_backup += 1
        # Make nn.Module dynamics for rl backup
        # TODO: Not sure if this is necessary. Check
        # self.rl_backup_dyn = BackupDynamicsModule()
        # self.rl_backup_dyn.init(self.dynamics, self.backup_policies[-1], self.obs_proc)
        assert isinstance(self.buffer, ReplayBuffer)

    # def rl_backup_safe_set(self, obs):
    #     return softmax(torch.stack([backup_set.safe_set(obs) for backup_set in self.backup_sets]),
    #                    self.params.rl_backup_backup_set_softmax_gain)

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
        beta = self.melt_law(h=torch.stack([backup_set.safe_barrier(obs) for backup_set in self.backup_sets[:-1]]))
        # TODO: Check if we need to call act here instead of step
        rl_backup_ac, _ = self.rl_backup.step(self.obs_proc.proc(obs, proc_key='backup_policy', reverse=True))
        # TODO: This is not correct when multiple backups have nonzero beta values
        return ((1 - torch.sum(beta)) * rl_backup_ac + torch.sum(backup_policies_vals * beta.view(-1, 1))).squeeze()

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

        return {"Loss/Policy": rl_backup_loss}


    def pre_train(self, samples, pre_train_dict=None):
        # Pretrain by imitation learning
        # TODO: Check pretraining. Make sure samples are converted to torch tensor on time
        self.rl_backup.policy_optimizer.zero_grad()
        acs_predicted, _ = self.rl_backup.step(samples)
        # acs_predicted = torchify(acs_predicted, dtype=torch.float64, requires_grad=True)

        samples_processed = self.obs_proc.proc(samples, proc_key='backup_set')
        samples_processed = torchify(samples_processed, dtype=torch.float64)
        backup_barrier_vals = torch.vstack([backup_set.safe_barrier(samples_processed)
                                            for backup_set in self.backup_sets[:-1]])
        max_barrier_indices = torch.argmax(backup_barrier_vals, dim=0)
        # TODO: Make acs_backup correct

        acs_backup = torch.vstack([self.backup_policies[idx.item()](samples_processed[i])
                                   for i, idx in enumerate(max_barrier_indices)])       # acs_backup will be tensor

        loss_rl_backup = F.mse_loss(acs_predicted, acs_backup)
        loss_rl_backup.backward()
        self.rl_backup.policy_optimizer.step()

        optim_info = {"Loss/Policy": loss_rl_backup.cpu().data.numpy()}
        logger.add_tabular(optim_info, cat_key='iteration')
        return optim_info


    def _get_softmax_softmin_backup_h_grad_h(self, obs):
        obs.requires_grad_()
        trajs = self._get_trajs(obs)
        h, h_grad, h_values, h_min_values, h_argmax = self._get_h_grad_h(obs, trajs)

        # Add rl-backup trajectory data to buffer
        traj_len = self._backup_t_seq.size()[0]
        rl_obs = self.obs_proc.proc(trajs[-1], proc_key='backup_policy', reverse=True)
        last_obs = trajs[-1][-1]

        # TODO: step vs act
        acs, _ = self.rl_backup.step(rl_obs)
        acs = acs.detach().numpy()
        rews = np.full(traj_len, h_values[-1].item())
        next_ob = self._fwd_prop_for_one_timestep(last_obs)
        next_ob = self.obs_proc.proc(next_ob, proc_key='backup_policy', reverse=True)
        rl_obs = rl_obs.detach().numpy()
        next_obs = np.vstack((rl_obs[1:], next_ob))
        done = np.zeros(traj_len)
        done[-1] = 1
        self.push_to_buffer((rl_obs, acs, rews.reshape(1, -1),
                             next_obs, done.reshape(1, -1), np.array([None] * traj_len)))

        # TODO: Convert obs back to numpy array if required
        obs.requires_grad_(requires_grad=False)
        return h, h_grad, h_values, h_min_values, h_argmax

    def _get_trajs(self, obs):
        # Find the trajectories under all backup policies except for the RL backup using odeint
        trajs = odeint(
            func=lambda t, y: torch.cat([self.dynamics.dynamics(yy, policy(yy))
                                         for yy, policy in zip(y.split(self._obs_dim_backup_policy), self.backup_policies)],
                                        dim=0),
            y0=obs.repeat(self._num_backup),
            t=self._backup_t_seq
        ).split(self._obs_dim_backup_policy, dim=1)
        return trajs

    def _fwd_prop_for_one_timestep(self, last_obs):
        # Forward propagate for one time step to get the next observation for the last observation in the trajectory
        return odeint(func=lambda t, y: self.dynamics.dynamics(y, self.backup_policies[-1](y)), y0=last_obs, t=self._backup_t_seq[:2])[-1].detach().numpy()


# class BackupDynamicsModule(nn.Module):
#     def init(self, dynamics, u_b, obs_proc):
#         self.dynamics = dynamics
#         self.u_b = u_b
#         self.obs_proc = obs_proc
#
#     def forward(self, t, y):
#         return self.dynamics.dynamics(y, self.u_b(y))

class RLBackupBackupSet(SafeSetFromBarrierFunction):
    def initialize(self, init_dict=None):
        self.params = init_dict['params']
        self.backup_sets = init_dict['backup_sets']

    def safe_barrier(self, obs):
        return softmax(torch.stack([backup_set.safe_barrier(obs) for backup_set in self.backup_sets]),
                       self.params.rl_backup_backup_set_softmax_gain)






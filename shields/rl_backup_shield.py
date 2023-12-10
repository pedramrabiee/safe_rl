import torch
from shields.backup_shield import BackupShield
from utils.torch_utils import softmin, softmax
from math import pi
from torchdiffeq import odeint
import numpy as np
from buffers.replay_buffer import ReplayBuffer
from logger import logger
import torch.nn.functional as F
from utils.misc import torchify
from utils.safe_set import SafeSetFromBarrierFunction
from copy import copy
from utils.scale import action2newbounds, action2oldbounds
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader
from utils.seed import rng
from attrdict import AttrDict
import matplotlib.pyplot as plt


class RLBackupShield(BackupShield):
    def initialize(self, params, init_dict=None):
        super().initialize(params, init_dict)
        self.rl_backup = init_dict.rl_backup
        if self.params.add_remained_time_to_obs:
            if self.rl_backup.init_dict is not None:
                rl_backup_init_dict = {**self.rl_backup.init_dict,
                                       'obs_dim': self.rl_backup.obs_dim + 1}
            else:
                rl_backup_init_dict = {'obs_dim': self.rl_backup.obs_dim + 1}

            self.rl_backup.initialize(params=self.rl_backup.params, init_dict=rl_backup_init_dict)

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

        assert isinstance(self.buffer, ReplayBuffer)
        self._reset_buffer_queue()

        self.include_rl_backup_in_h = True
        self._backup_horizon = self._backup_t_seq[-1]
        self._plotter_counter = 0
        self._rl_backup_explore = False


    def rl_backup_melt_into_backup_policies(self, obs, **kwargs):
        """
           Perform RL backup by melting the contributions of backup policies based on a melt law.

           The function computes the values of backup policies, applies a melt law to determine
           their respective contributions, and combines them with the RL backup action.

           Parameters:
           - obs (torch.Tensor): Input tensor representing the observation.

           Returns:
           torch.Tensor: Melted RL backup result based on the contributions of backup policies.
           """
        backup_policies_vals = [policy(obs) for policy in self.backup_policies[:-1]]
        h = [backup_set.safe_barrier(obs) for backup_set in self.backup_sets[:-1]]
        beta = self.melt_law(h)
        if self.params.add_remained_time_to_obs:
            if 'traj_time' in kwargs:
                # normalize traj time by horizon
                remained_time = 1 - kwargs['traj_time'] / self._backup_horizon
            else:
                remained_time = torch.tensor(1)

            new_obs = self.obs_proc.proc(obs, proc_key='backup_policy', reverse=True)
            if new_obs.ndim == 1:
                new_obs = torch.hstack((new_obs, remained_time))
            else:
                if remained_time.ndim == 0:
                    new_obs = torch.cat((new_obs, torch.full((new_obs.size(0), 1), remained_time)), dim=1)
                else:
                    new_obs = torch.hstack((new_obs, remained_time.unsqueeze(1) if remained_time.ndim == 1
                    else remained_time))
            rl_backup_ac, _ = self.rl_backup.step(obs=new_obs, explore=self._rl_backup_explore)
        else:
            rl_backup_ac, _ = self.rl_backup.step(obs=self.obs_proc.proc(obs, proc_key='backup_policy', reverse=True),
                                                  explore=self._rl_backup_explore)

        # it is assumed that rl backup outputs actions in new bounds, scale it back to old bounds
        rl_backup_ac = action2oldbounds(rl_backup_ac)

        weighted_bpks = [b * bkp for b, bkp in zip(beta, backup_policies_vals)]
        stacked_weighted_bkps = torch.stack(weighted_bpks, dim=0)
        blended_bkps = torch.sum(stacked_weighted_bkps, dim=0)
        # TODO: This is not correct when multiple backups have nonzero beta values
        ac = (1 - torch.sum(torch.hstack(beta), dim=-1).unsqueeze(dim=-1)) * rl_backup_ac + blended_bkps
        if self._rl_backup_explore:
            self._push_ac_to_queue(ac)
        return ac

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
        return [torch.where(
            hh >= 0,  # Condition for the first case
            torch.tensor(1.0),  # Result if the condition is True
            0.5 * (1 - torch.cos(pi * (self.params.melt_law_gain * hh - 1))) * (hh > -1 / self.params.melt_law_gain)).unsqueeze(dim=-1)
            for hh in h]

    def optimize_agent(self, samples, optim_dict=None):
        if not self.params.rl_backup_train:
            return
        logger.log('Training RL Backup Agent...')
        rl_backup_loss = self.rl_backup.optimize_agent(samples, optim_dict)

        return rl_backup_loss

    def pre_train(self, samples, pre_train_dict=None):
        if not self.params.rl_backup_train:
            return
        samples = torch.from_numpy(samples)

        samples_processed = self.obs_proc.proc(samples, proc_key='backup_set')
        samples_processed = torchify(samples_processed, dtype=torch.float64)

        # backup_barrier_vals = torch.vstack([backup_set.safe_barrier(samples_processed)
        #                                     for backup_set in self.backup_sets[:-1]])

        barriers_vals = torch.vstack([self.get_h_per_id_from_batch_of_obs(samples_processed, idx)
                        for idx in range(self._num_backup - 1)])

        max_barrier_indices = torch.argmax(barriers_vals, dim=0)

        # TODO: Make acs_backup correct
        acs_backup = torch.vstack([self.backup_policies[idx.item()](samples_processed[i])
                                   for i, idx in enumerate(max_barrier_indices)])  # acs_backup will be tensor

        acs_backup_scaled = action2newbounds(acs_backup)
        if self.params.add_remained_time_to_obs:
            samples = torch.hstack((samples, torch.ones(samples.shape[0], 1)))
        train_dataloader = DataLoader(TensorDataset(samples, acs_backup_scaled),
                                      batch_size=self.params.rl_backup_pretrain_batch_size, shuffle=True)

        batch_to_sample_ratio = self.params.rl_backup_pretrain_batch_size / samples.shape[0]
        max_epoch = int(self.params.rl_backup_pretrain_epoch / batch_to_sample_ratio)
        pbar = tqdm(total=max_epoch, desc='Filter Pretraining Progress')

        for _ in range(max_epoch):
            batch_samples, batch_labels = next(iter(train_dataloader))

            # Pretrain by imitation learning
            # TODO: Check pretraining. Make sure samples are converted to torch tensor on time
            self.rl_backup.policy_optimizer.zero_grad()
            loss_rl_backup = self._compute_pretrain_loss(samples=batch_samples, labels=batch_labels)
            loss_rl_backup.backward()
            self.rl_backup.policy_optimizer.step()

            optim_info = {"Loss/RLBackup": loss_rl_backup.cpu().data.numpy()}
            logger.add_tabular(optim_info, cat_key='rl_backup_pretrain_epoch')

            logger.dump_tabular(cat_key='rl_backup_pretrain_epoch', log=False, wandb_log=True, csv_log=False)
            pbar.update(1)

        pbar.close()

    def _compute_pretrain_loss(self, samples, labels):
        acs_predicted, _ = self.rl_backup.step(samples)
        loss_rl_backup = F.mse_loss(acs_predicted, labels)
        return loss_rl_backup


    def _get_softmax_softmin_backup_h_grad_h(self, obs):
        obs.requires_grad_()
        # # Reset buffer queue mainly to clear _rl_backup_ac_queue
        # self._reset_buffer_queue()

        # RESET ACTION QUEUE
        self._reset_temp_ac_queue()
        # calls to rl_melt_into_backup by odeint is appended to _rl_backup_ac_queue
        trajs = self._get_trajs(obs)

        h, h_grad, h_values, h_min_values, h_argmax = self._get_h_grad_h(obs, trajs)

        # Add rl-backup trajectory data to buffer
        self._process_backup_traj_push_to_queue(trajs=trajs, h_values=h_values)

        # TODO: Convert obs back to numpy array if required
        obs.requires_grad_(requires_grad=False)
        return h, h_grad, h_values, h_min_values, h_argmax

    def _get_trajs(self, obs):
        # Find the trajectories under all backup policies except for the RL backup using odeint
        if self.params.add_remained_time_to_obs:
            backup_trajs = odeint(
                func=lambda t, y: torch.cat([self.dynamics.dynamics(yy, policy(yy))
                                             for yy, policy in zip(y.split(self._obs_dim_backup_policy), self.backup_policies[:-1])],
                                            dim=0),
                y0=obs.repeat(self._num_backup - 1),
                t=self._backup_t_seq
            ).split(self._obs_dim_backup_policy, dim=1)

            # TODO: Check and Test this case
            rl_backup_trajs = odeint(
                func=lambda t, y: self.dynamics.dynamics(y, self.backup_policies[-1](y, traj_time=t)),
                y0=obs,
                t=self._backup_t_seq)

            trajs = (*backup_trajs, rl_backup_trajs)
        else:
            if self._rl_backup_explore:
                backup_trajs = odeint(
                    func=lambda t, y: torch.cat([self.dynamics.dynamics(yy, policy(yy))
                                                 for yy, policy in zip(y.split(self._obs_dim_backup_policy), self.backup_policies[:-1])],
                                                dim=0),
                    y0=obs.repeat(self._num_backup - 1),
                    t=self._backup_t_seq
                ).split(self._obs_dim_backup_policy, dim=1)

                rl_backup_trajs = odeint(
                    func=lambda t, y: self.dynamics.dynamics(y, self.backup_policies[-1](y)),
                    y0=obs,
                    t=self._backup_t_seq,
                    method=self.params.ode_method_for_explor_mode,
                )
                trajs = (*backup_trajs, rl_backup_trajs)
            else:
                trajs = super()._get_trajs(obs)
        return trajs

    def _fwd_prop_for_one_timestep_per_id(self, last_obs, id):
        if id != self._num_backup - 1:
            return odeint(func=lambda t, y: self.dynamics.dynamics(y, self.backup_policies[id](y)), y0=last_obs,
                          t=self._backup_t_seq[:2])[-1].detach().numpy()
        else:
            if self.params.add_remained_time_to_obs:
                # Forward propagate for one time step to get the next observation for the last observation in the trajectory
                return odeint(func=lambda t, y:
                self.dynamics.dynamics(y, self.backup_policies[id](y, traj_time=self._backup_t_seq[-1] + t)),
                              y0=last_obs,
                              t=self._backup_t_seq[:2])[-1].detach().numpy()
            else:
                return odeint(func=lambda t, y: self.dynamics.dynamics(y, self.backup_policies[id](y)), y0=last_obs,
                              t=self._backup_t_seq[:2],
                              method=self.params.ode_method_for_explor_mode if self._rl_backup_explore else None)[-1].detach().numpy()

    def _process_backup_traj_push_to_queue(self, trajs, h_values):
        if not self.params.rl_backup_train:
            return

        traj_len = self._backup_t_seq.size()[0]

        obs_list = []
        next_obs_list = []
        rew_list = []
        done_list = []

        for id, (traj, h_value) in enumerate(zip(trajs, h_values)):

            # Obtain obs from trajectory and detach
            obs = traj.detach()
            next_ob = self._fwd_prop_for_one_timestep_per_id(last_obs=torch.as_tensor(obs[-1]), id=id)
            next_obs = np.vstack((obs[1:], next_ob))


            obs = obs.numpy()

            # make reward
            # rews = self._get_reward(traj.detach(), id)
            rews = np.full(traj_len, h_value.item())

            done = np.zeros(traj_len)
            done[-1] = 1

            # if id == self._num_backup - 1 and self.params.add_remained_time_to_obs:
            #     # t_seq = 1 - self._backup_t_seq.unsqueeze(dim=1).numpy() / self._backup_t_seq[-1]
            #     t_seq = self._backup_t_seq.unsqueeze(dim=1).numpy()
            #     obs = np.concatenate((obs, t_seq), axis=1)
            #     next_t = 2 * self._backup_t_seq[-1] - self._backup_t_seq[-2]
            #     new_t_seq = np.vstack((t_seq[1:], next_t))
            #     next_obs = np.concatenate((next_obs, new_t_seq), axis=1)

            obs_list.append(obs)
            next_obs_list.append(next_obs)
            rew_list.append(rews.reshape(-1, 1))
            done_list.append(done.reshape(-1, 1))
            # TODO: remove the following

        if self._plotter_counter % 20 == 0:
            self.plot_traj(obs_list[-1])

        self._plotter_counter += 1

        # PROCESS ACTION QUEUE AND ADD TO PERMANENT ACTION QUEUE
        self._process_temp_ac_queue_push_ac_to_perm_queue()

        # push data to buffer queue
        self._push_to_queue(obs_list, next_obs_list, rew_list, done_list)

    def _get_reward(self, traj, id):

        h_b = self.backup_sets[id].safe_barrier(traj)
        h_s = self.safe_set.des_safe_barrier(traj)

        return torch.where(h_s >= 0,
                           h_b,
                           h_b + 1000 * h_s).numpy()


    def _reset_buffer_queue(self):
        self._obs_buf = None
        self._next_obs_buf = None
        self._rews_buf = None
        self._done_buf = None
        self._rl_backup_acs_buf = dict(temp=[], perm=None)

    def _push_to_queue(self, obs, next_obs, rews, done, **kwargs):
        if not self.params.rl_backup_train:
            return

        if obs[0].ndim == 1:
            obs.unsqueeze(dim=0)
            next_obs.unsqueeze(dim=0)

        if self._obs_buf is None:
            self._obs_buf = obs
            self._next_obs_buf = next_obs
            self._rew_buf = rews
            self._done_buf = done
        else:
            self._obs_buf = [np.concatenate((arr1, arr2), axis=0) for arr1, arr2 in zip(self._obs_buf, obs)]
            self._next_obs_buf = [np.concatenate((arr1, arr2), axis=0) for arr1, arr2 in zip(self._next_obs_buf, next_obs)]
            self._rew_buf = [np.concatenate((arr1, arr2), axis=0) for arr1, arr2 in zip(self._rew_buf, rews)]
            self._done_buf = [np.concatenate((arr1, arr2), axis=0) for arr1, arr2 in zip(self._done_buf, done)]

    def _reset_temp_ac_queue(self):
        self._rl_backup_acs_buf['temp'] = []

    def _reset_perm_ac_queue(self):
        self._rl_backup_acs_buf['perm'] = None

    def _process_temp_ac_queue_push_ac_to_perm_queue(self):
        acs = torch.vstack(self._rl_backup_acs_buf['temp']).detach().numpy()

        # RK45 make 5 calls per timestep to rl_backup_melt_into_backup_policies.
        # Only the first call needs to be stored in the buffer
        cand_acs = acs[np.arange(acs.shape[0]) % 1 == 0]
        if self._rl_backup_acs_buf['perm'] is None:
            self._rl_backup_acs_buf['perm'] = cand_acs
        else:
            self._rl_backup_acs_buf['perm'] = np.concatenate((self._rl_backup_acs_buf['perm'], cand_acs), axis=0)
        # reset temporary rl backup acs buffer
        self._reset_temp_ac_queue()

    def _push_ac_to_queue(self, ac):
        if not self.params.rl_backup_train:
            return

        self._rl_backup_acs_buf['temp'].append(ac)

        # if ac.ndim == 1:
        #     ac.unsqueeze(dim=0)
        #
        # if self._rl_backup_acs_buf is None:
        #     self._rl_backup_acs_buf = ac
        # else:
        #     self._rl_backup_acs_buf = np.concatenate((self._rl_backup_acs_buf, ac.detach().numpy), axis=0)

    @torch.no_grad()
    def compute_ac_push_to_buffer(self, episode):
        if not self.params.rl_backup_train:
            return
        # Get actions by getting query from the rl_backup_melt_into_backup_policies at rl_obs
        # Scale to new bounds
        for id, (obs, next_obs, rews, done) in enumerate(zip(self._obs_buf, self._next_obs_buf, self._rew_buf, self._done_buf)):
            if not self.params.add_backup_trajs_to_buf:
                if id != self._num_backup - 1:
                    continue

            obs_tensor = torch.as_tensor(obs)
            if id != self._num_backup - 1:
                # it is assumed that backup policies other than the rl backup policy output actions in new action bounds
                acs = self.backup_policies[id](obs_tensor).detach().numpy()
                acs = action2newbounds(acs)

                # epsilon = np.random.randn(*acs.shape) * 0.05
                # epsilon = np.clip(epsilon, -0.02, 0.02)
                # acs = np.clip(acs + epsilon, self._ac_lim['low'], self._ac_lim['high'])

                traj_len = rews.shape[0]
                obs = self.obs_proc.proc(obs, proc_key='backup_policy', reverse=True)
                next_obs = self.obs_proc.proc(next_obs, proc_key='backup_policy', reverse=True)
                self.push_to_buffer((obs, acs, rews, next_obs, done, np.array([None] * traj_len)))
            else:

                if self.params.add_remained_time_to_obs:
                    raise NotImplementedError
                    # acs = action2newbounds(self.rl_backup_melt_into_backup_policies(obs=obs_tensor[:, :-1],
                    #                                                                 traj_time=obs_tensor[:, -1].reshape(-1, 1)))
                    # acs = acs.detach().numpy()
                    #
                    # obs = np.concatenate(
                    #     (self.obs_proc.proc(obs[:, :-1], proc_key='backup_policy', reverse=True),
                    #      1 - obs[:, -1].reshape(-1, 1) / self._backup_t_seq[-1]),
                    #     axis=1)
                    # next_obs = np.concatenate(
                    #     (self.obs_proc.proc(next_obs[:, :-1], proc_key='backup_policy', reverse=True),
                    #      1 - next_obs[:, -1].reshape(-1, 1) / self._backup_t_seq[-1]),
                    #     axis=1)
                else:
                    # acs = action2newbounds(self.rl_backup_melt_into_backup_policies(obs_tensor))
                    # acs = acs.detach().numpy()

                    acs = self._rl_backup_acs_buf['perm']

                    # compute next observation from the last observation of rl obs and make next rl obs
                    # next_ob = self._fwd_prop_for_one_timestep_per_id(last_obs=torch.as_tensor(obs[-1]), id=id)
                    # next_obs = np.vstack((obs[1:], next_ob))

                    # Convert rl_obs to
                    obs = self.obs_proc.proc(obs, proc_key='backup_policy', reverse=True)
                    next_obs = self.obs_proc.proc(next_obs, proc_key='backup_policy', reverse=True)
                    # Only add actions that are inside safe set to buffer

                traj_len = rews.shape[0]

                if self.params.add_unsafe_data_to_buf:
                    self.push_to_buffer((obs, acs, rews,
                                         next_obs, done, np.array([None] * traj_len)))
                else:
                    mask = self.safe_set.is_des_safe(obs_tensor)
                    self.push_to_buffer((obs[mask], acs[mask], rews[mask],
                                         next_obs[mask], done[mask], np.array([None] * traj_len)[mask]))


        # Reset buffer queue
        self._reset_buffer_queue()
        self._reset_perm_ac_queue()

    def get_h_from_batch_of_obs(self, obs):
        if not torch.is_tensor(obs):
            obs = torch.from_numpy(obs)
        h_list = [self.get_h_per_id_from_batch_of_obs(obs, id) for id in range(self._num_backup if
                                                                         self.include_rl_backup_in_h
                                                                         else self._num_backup - 1)]
        return self._get_softmax_from_h(h_list)

    def get_customized_samples(self, batch_size, device='cpu'):
        samples = dict()
        # First get the entire buffer so that you can remove
        buffer = self.get_buffer(to_tensor=True, device=device)

        buffer.done = buffer.done.view(-1, 1)
        del buffer.info

        while True:
            buffer_size = buffer.rew.shape[0] if buffer is not None else 0
            inds = rng.choice(np.arange(buffer_size),
                              size=min(batch_size*2, buffer_size),
                              replace=False)

            obs = buffer.obs[inds]
            if len(samples) > 0:
                if samples['rew'].shape[0] >= batch_size:
                    break
            if buffer.rew.size(0) == 0:
                break
            #
            # TODO: Do observation process
            # mask = self.get_h_per_id_from_batch_of_obs(obs=self.obs_proc.proc(obs, proc_key='safe_set'), id=self._num_backup - 1) >= self.params.sampling_h_cutoff
            # mask = np.array([True] * inds.shape[0])
            # masked_inds = inds[mask]
            # Add observation which satisfy the criteria to samples[i]

            for k in buffer:
                # samples[k] = torch.vstack((samples[k], buffer[k][masked_inds])) if k in samples \
                #     else buffer[k][masked_inds]
                samples[k] = torch.vstack((samples[k], buffer[k][inds])) if k in samples \
                    else buffer[k][inds]

                # Remove the ones that are added to samples for unique sampling
                buffer[k] = np.delete(buffer[k], inds, axis=0)
        for k in samples:
            samples[k] = samples[k][:batch_size]

        return AttrDict(samples)

    def set_rl_backup_explore(self, status):
        self._rl_backup_explore = status

    def set_desired_policy(self, desired_policy, is_mf_policy):
        self.desired_policy = desired_policy
        self.is_mf_desired_policy = is_mf_policy

    @property
    def rl_backup_id(self):
        return self._num_backup - 1

    def add_batch_of_data_to_buffer_from_obs(self, obs):

        # TURN ON EXPLORATION
        self.set_rl_backup_explore(True)
        t_seq = torch.hstack((self._backup_t_seq, 2 * self._backup_t_seq[-1] - self._backup_t_seq[-2]))

        # MODIFY get_trajs_per_id_from_batch_of_obs TO SAMPLE WITH EXPLORATION

        self._reset_temp_ac_queue()
        obs = torch.as_tensor(obs).flatten()
        obs.requires_grad_(False)

        trajs = odeint(
            lambda t, y: self.dynamics.dynamics_flat_return(
                y.view(-1, self._obs_dim_backup_policy),
                self.backup_policies[self.rl_backup_id](y.view(-1, self._obs_dim_backup_policy))),
            obs,
            t_seq,
            method=self.params.ode_method_for_explor_mode)

        trajs = trajs.detach()          # #timesteps+1 by batch_size * obs_dim

        # observation and next observation values
        obs = trajs[:-1, :]             # #timesteps by batch_size * obs_dim
        next_obs = trajs[1:, :]         # #timesteps by batch_size * obs_dim
        traj_len = obs.size(0)


        stacked_obs = torch.vstack(torch.split(obs, self._obs_dim_backup_policy, dim=1))
        stacked_next_obs = torch.vstack(torch.split(next_obs, self._obs_dim_backup_policy, dim=1)).numpy()

        # Compute h values
        h_s = self.safe_set.des_safe_barrier(stacked_obs)
        h_s = torch.vstack(h_s.split(traj_len)).t()       # #timesteps by batch_size
        h_b = self.backup_sets[self.rl_backup_id].safe_barrier(obs[-1, :].view(-1, self._obs_dim_backup_policy)).view(1, -1)    # 1 by batch_size
        h = softmin(torch.vstack((h_s, h_b)), self.params.softmin_gain)     # batch_size

        acs = torch.hstack(self._rl_backup_acs_buf['temp']).t().detach()    # #timesteps by batch_size * ac_dim

        # RK45 make 5 calls per timestep to rl_backup_melt_into_backup_policies.
        # Only the first call needs to be stored in the buffer
        cand_acs = acs[np.arange(acs.shape[0]) % 1 == 0]

        stacked_cand_acs = torch.vstack(torch.split(cand_acs, self._ac_dim, dim=1)).numpy()


        # reset temporary rl backup acs buffer
        self._reset_temp_ac_queue()

        # make reward
        rews = torch.tile(h.view(1, -1), (traj_len, 1))
        stacked_rews = torch.vstack(torch.split(rews, 1, dim=1)).numpy()

        # make done
        done = torch.zeros_like(rews)
        done[-1] += 1
        stacked_done = torch.vstack(torch.split(done, 1, dim=1)).numpy()

        info = np.array([None] * stacked_rews.shape[0])

        stacked_obs = self.obs_proc.proc(stacked_obs.numpy(), proc_key='backup_policy', reverse=True)
        stacked_next_obs = self.obs_proc.proc(stacked_next_obs, proc_key='backup_policy', reverse=True)

        # push data to buffer
        self.push_to_buffer((stacked_obs, stacked_cand_acs, stacked_rews,
                             stacked_next_obs, stacked_done, info))

        # TURN OFF EXPLORATION
        self.set_rl_backup_explore(False)


    # DEBUGGING METHODS

    def plot_traj(self, obs):
        # Convert the torch tensor to a numpy array

        # Extract the first and second columns
        column1 = obs[:, 0]
        column2 = obs[:, 1]

        # Plot the first column vs the second column
        plt.plot(column1, column2)

        # Mark the initial point with 'o' or 'x'
        plt.scatter(column1[0], column2[0], color='red', marker='o')

        # Set axis limits
        plt.xlim(-3, 3)
        plt.ylim(-3, 3)

        # Add labels and title
        plt.xlabel(r'$\theta$')
        plt.ylabel(r'$\dot \theta$')

        # Show the plot
        plt.show()



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
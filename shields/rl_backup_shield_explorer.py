from shields.rl_backup_shield import RLBackupShield
from torch import linspace
from utils.scale import action2newbounds, action2oldbounds
from torchdiffeq import odeint
import torch
import numpy as np
from utils.seed import rng


class RLBackupShieldExplorer(RLBackupShield):
    def initialize(self, params, init_dict=None):
        super().initialize(params, init_dict)
        if self.params.add_remained_time_to_obs:
            raise NotImplementedError

        # make rl backup t sequence
        rl_backup_horizon = self.params.horizon + self.params.des_policy_horizon
        self._rl_backup_t_seq = linspace(0,
                                         rl_backup_horizon,
                                         int(rl_backup_horizon / self.params.backup_timestep) + 1)

        self.backup_dilusion_rate = 1.0
        self.dilusion_rate = 1.0

    def _get_t_switch(self):
        horizon = self.params.des_policy_horizon
        if self.params.sample_t_switch:
            batch_size = 1
            horizon = torch.as_tensor(rng.uniform(low=0.0, high=self.params.des_policy_horizon, size=batch_size))

        t_switch = (1 - self.params.des_policy_melt_region_ratio) * horizon

        return t_switch, horizon
    def rl_backup_melt_from_des_policy(self, t, obs, **kwargs):
        t_switch = kwargs['t_switch']
        horizon = kwargs['horizon']

        u_des = lambda x: action2oldbounds(self.desired_policy.step(self.obs_proc.proc(x, proc_key='backup_policy',
                                                                                       reverse=True), explore=False)[0])\
            if self.is_mf_desired_policy else self.desired_policy.act(x)

        if t.ndim == 0:
            if t < t_switch:
                ac = u_des(obs)
            elif t < horizon:
                a1 = t - t_switch
                a2 = horizon - t
                ac = (self._rl_backup_melt_into_backup_policies(obs, **kwargs) * a1 +
                        u_des(obs) * a2) / (horizon - t_switch)
            else:
                ac = self._rl_backup_melt_into_backup_policies(obs, **kwargs)

            if self._rl_backup_explore:
                self._push_ac_to_queue(ac)
            return ac

        a1 = t - t_switch
        a2 = self.params.des_policy_horizon - t

        result = torch.where(t < t_switch,
                             u_des(obs),
                             torch.where(
                                 t < horizon,
                                 (self._rl_backup_melt_into_backup_policies(obs, **kwargs) * a1 +
                                  u_des(obs) * a2) / (horizon - t_switch),
                                 self._rl_backup_melt_into_backup_policies(obs, **kwargs)
                             ))

        if self._rl_backup_explore:
            self._push_ac_to_queue(result)
        return result

    def _get_softmax_softmin_backup_h_grad_h(self, obs):
        obs.requires_grad_()

        # RESET ACTION QUEUE
        self._reset_temp_ac_queue()
        # calls to rl_melt_into_backup by odeint is appended to _rl_backup_ac_queue
        trajs = self._get_trajs_explorer(obs)

        h, h_grad, h_values, h_min_values, h_argmax = self._get_h_grad_h(obs, trajs)

        # Add rl-backup trajectory data to buffer
        self._process_backup_traj_push_to_queue(trajs=trajs, h_values=h_values)

        # TODO: Convert obs back to numpy array if required
        obs.requires_grad_(requires_grad=False)
        return h, h_grad, h_values, h_min_values, h_argmax


    def _get_trajs_explorer(self, obs):
        # Find the trajectories under all backup policies except for the RL backup using odeint
        backup_trajs = odeint(
            func=lambda t, y: torch.cat([self.dynamics.dynamics(yy, policy(yy))
                                         for yy, policy in zip(y.split(self._obs_dim_backup_policy), self.backup_policies[:-1])],
                                        dim=0),
            y0=obs.repeat(self._num_backup - 1),
            t=self._backup_t_seq,
        ).split(self._obs_dim_backup_policy, dim=1)

        t_switch, horizon = self._get_t_switch()
        rl_backup_trajs = odeint(
            func=lambda t, y: self.dynamics.dynamics(y, self.rl_backup_melt_from_des_policy(t, y,
                                                                                            t_switch=t_switch,
                                                                                            horizon=horizon)),
            y0=obs,
            t=self._rl_backup_t_seq,
            method=self.params.ode_method_for_explor_mode if self._rl_backup_explore else None
        )

        trajs = (*backup_trajs, rl_backup_trajs)

        return trajs

    @torch.no_grad()
    def compute_ac_push_to_buffer(self, episode):
        if not self.params.rl_backup_train:
            return
        # Get actions by getting query from the rl_backup_melt_into_backup_policies at rl_obs
        # Scale to new bounds
        for id, (obs, next_obs, rews, done, traj_time) in enumerate(zip(self._obs_buf, self._next_obs_buf,
                                                                        self._rew_buf, self._done_buf, self._t_buf)):

            obs_tensor = torch.as_tensor(obs)
            if id != self.rl_backup_id:
                if not self.params.add_backup_trajs_to_buf:
                    continue

                # it is assumed that backup policies other than the rl backup policy output actions in new action bounds
                acs = self.backup_policies[id](obs_tensor).detach().numpy()
                acs = action2newbounds(acs)

                # epsilon = np.random.randn(*acs.shape) * 0.05
                # epsilon = np.clip(epsilon, -0.02, 0.02)
                # acs = np.clip(acs + epsilon, self._ac_lim['low'], self._ac_lim['high'])

                traj_len = rews.shape[0]
                obs = self.obs_proc.proc(obs, proc_key='backup_policy', reverse=True)
                next_obs = self.obs_proc.proc(next_obs, proc_key='backup_policy', reverse=True)
                mask = np.random.choice(obs.shape[0], int(obs.shape[0] * self.dilusion_rate * self.backup_dilusion_rate ** episode), replace=False)
                self.push_to_buffer((obs[mask], acs[mask], rews[mask], next_obs[mask], done[mask], np.array([None] * traj_len)[mask]))
            else:
                # acs = action2newbounds(self.rl_backup_melt_from_des_policy(torch.as_tensor(traj_time), obs_tensor))
                # acs = acs.detach().numpy()

                acs = self._rl_backup_acs_buf['perm']

                # Only add actions that are inside safe set to buffer
                mask = self.safe_set.is_des_safe(obs_tensor)
                obs = obs[mask]
                next_obs = next_obs[mask]
                rews = rews[mask]
                done = done[mask]

                # mask = np.random.choice(obs.shape[0],
                #                         int(obs.shape[0] * self.dilusion_rate),
                #                         replace=False)
                # obs = obs[mask]
                # next_obs = next_obs[mask]
                # rews = rews[mask]
                # done = done[mask]

                # Convert rl_obs to
                obs = self.obs_proc.proc(obs, proc_key='backup_policy', reverse=True)
                next_obs = self.obs_proc.proc(next_obs, proc_key='backup_policy', reverse=True)

                traj_len = rews.shape[0]

                # self.push_to_buffer((obs[mask], acs[mask], rews[mask],
                #                      next_obs[mask], done[mask], np.array([None] * traj_len)[mask]))

                self.push_to_buffer((obs, acs, rews,
                                     next_obs, done, np.array([None] * traj_len)))


        # Reset buffer queue
        self._reset_buffer_queue()
        self._reset_perm_ac_queue()

    def _process_backup_traj_push_to_queue(self, trajs, h_values):
        if not self.params.rl_backup_train:
            return

        traj_len = self._backup_t_seq.size()[0]
        rl_traj_len = self._rl_backup_t_seq.size()[0]

        obs_list = []
        next_obs_list = []
        rew_list = []
        done_list = []
        traj_time_list = []

        for id, (traj, h_value) in enumerate(zip(trajs, h_values)):

            # Obtain obs from trajectory and detach
            obs = traj.detach()

            # compute next observation from the last observation of rl obs and make next rl obs
            next_ob = self._fwd_prop_for_one_timestep_per_id(last_obs=obs[-1], id=id)
            next_obs = np.vstack((obs[1:], next_ob))

            obs = obs.numpy()

            # make reward
            if id == self.rl_backup_id:
                traj_time = self._rl_backup_t_seq.numpy()
                rews = np.full(rl_traj_len, h_value.item())
                done = np.zeros(rl_traj_len)
            else:
                traj_time = self._backup_t_seq.numpy()
                rews = np.full(traj_len, h_value.item())
                done = np.zeros(traj_len)
            done[-1] = 1


            # rews = self._get_reward(traj.detach(), id)

            obs_list.append(obs)
            next_obs_list.append(next_obs)
            rew_list.append(rews.reshape(-1, 1))
            done_list.append(done.reshape(-1, 1))
            traj_time_list.append(traj_time.reshape(-1, 1))

            # TODO: remove the following

            # if self._plotter_counter % 10 == 0:
            #     self.plot_traj(rl_obs)

            # self._plotter_counter += 1

        # PROCESS ACTION QUEUE AND ADD TO PERMANENT ACTION QUEUE
        self._process_temp_ac_queue_push_ac_to_perm_queue()

        self._push_to_queue(obs_list, next_obs_list, rew_list, done_list, traj_time=traj_time_list)

    # def _fwd_prop_for_one_timestep_per_id(self, last_obs, id):
    #     if id != self.rl_backup_id:
    #         return odeint(func=lambda t, y: self.dynamics.dynamics(y, self.backup_policies[id](y)), y0=last_obs,
    #                       t=self._backup_t_seq[:2])[-1].detach().numpy()
    #     else:
    #         return odeint(func=lambda t, y: self.dynamics.dynamics(y, self.rl_backup_melt_into_backup_policies(self._rl_backup_t_seq[-1], y)), y0=last_obs,
    #                       t=self._rl_backup_t_seq[:2])[-1].detach().numpy()

    def _push_to_queue(self, obs, next_obs, rews, done, **kwargs):
        super()._push_to_queue(obs, next_obs, rews, done)
        if self._t_buf is None:
            self._t_buf = kwargs['traj_time']
        else:
            self._t_buf = [np.concatenate((arr1, arr2), axis=0) for arr1, arr2 in zip(self._t_buf, kwargs['traj_time'])]

    def _reset_buffer_queue(self):
        super()._reset_buffer_queue()
        self._t_buf = None


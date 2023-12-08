from utils.safe_set import SafeSetFromBarrierFunction
import torch
from math import pi
from attrdict import AttrDict
import numpy as np
from envs_utils.gym.pendulum.pendulum_configs import env_config, safe_set_dict
from utils.seed import rng
from scipy.stats import truncnorm
from envs_utils.gym.pendulum.pendulum_obs_proc import PendulumObsProc
from utils.custom_plotter import CustomPlotter
from logger import logger
from utils.plot_utils import plot_zero_level_sets


class PendulumBackupSet(SafeSetFromBarrierFunction):
    def initialize(self, init_dict=None):
        self.p = torch.tensor(init_dict.p)
        self.c = init_dict.c
        self.center = torch.tensor(init_dict.center)

    # def safe_barrier(self, obs):
    #     # TODO: make obs and center matrix to handle batch
    #     if not torch.is_tensor(obs):
    #         obs = torch.from_numpy(obs)
    #     result = torch.matmul(torch.matmul(obs - self.center, self.p), (obs - self.center).t())
    #     return 1 - result / self.c if result.dim() == 0 else 1 - result.diag() / self.c

    def safe_barrier(self, obs):
        if not torch.is_tensor(obs):
            obs = torch.from_numpy(obs)

        diff = obs - self.center

        if obs.dim() == 1:
            # Unsqueeze to add batch dimension
            result = torch.dot(diff, torch.mv(self.p, diff))
            result = result.squeeze()

        elif obs.dim() == 2:
            einsum_str = 'bi,ij,bj->b'
            result = torch.einsum(einsum_str, diff, self.p, diff)
        else:
            raise ValueError('obs must be 1D or 2D tensor')

        res = 1 - result / self.c
        return torch.where(res >= 0, res, res / 550)

class PendulumSafeSet(SafeSetFromBarrierFunction):
    def initialize(self, init_dict=None):
        self.bounds = torch.tensor(init_dict.bounds).unsqueeze(dim=0)
        self.center = torch.tensor(init_dict.center).unsqueeze(dim=0)
        self.p_norm = init_dict.p_norm

    def late_initialize(self, init_dict=None):
        self.backup_agent = init_dict.backup_agent

    def des_safe_barrier(self, obs):
        if not torch.is_tensor(obs):
            obs = torch.from_numpy(obs)
        return 1 - torch.norm((torch.atleast_1d(obs) - self.center) / self.bounds, p=self.p_norm, dim=1)

    def safe_barrier(self, obs):
        if obs.ndim == 1:
            h, _, _ = self.backup_agent.get_h(obs)
        else:
            h = self.backup_agent.get_h_from_batch_of_obs(obs)
        return h

    def _get_obs(self, batch_size):
        # max_speed = self.env.observation_space.high[2]
        max_speed = safe_set_dict.bounds[1]
        theta = rng.uniform(low=-safe_set_dict.bounds[0], high=safe_set_dict.bounds[0], size=batch_size)
        thetadot = truncnorm.rvs(-1, 1, scale=max_speed, size=batch_size) if env_config.sample_velocity_gaussian \
            else rng.uniform(low=-max_speed, high=max_speed, size=batch_size)
        return np.array([np.cos(theta), np.sin(theta), thetadot]) if batch_size == 1 \
            else np.vstack([np.cos(theta), np.sin(theta), thetadot]).T


class PendulumBackupControl:
    def __init__(self, gain, center, ac_lim):
        self.gain = torch.tensor(gain)
        self.center = torch.tensor(center)
        self.ac_lim = ac_lim
        self.u_eq = self.get_u_equilibrium()

    def __call__(self, obs):
        if obs.ndim == 1:
            ac = self.ac_lim * torch.tanh((self.u_eq + torch.dot(self.gain, obs)) / self.ac_lim)
            return ac.unsqueeze(dim=0)
        else:
            ac = self.ac_lim * torch.tanh((self.u_eq + torch.mv(obs, self.gain)) / self.ac_lim)
            return ac.unsqueeze(dim=1)

    def get_u_equilibrium(self):
        return self.ac_lim * torch.atanh(
            (-env_config.m * env_config.g * env_config.l * torch.sin(self.center[0]) / 2) / self.ac_lim) - self.gain[
            0] * self.center[0]


_backup_sets_dict = dict(c=[0.02, 0.02, 0.02],
                         p=[
                             [[0.625, 0.125], [0.125, 0.125]],

                             # u_max = 1.5
                             [[0.650, 0.150], [0.150, 0.240]],
                             [[0.650, 0.150], [0.150, 0.240]],
                             # u_max = 6.5
                             # [[0.585353535353535, 0.0853535353535354], [0.0853535353535354, 0.114494439342924]],
                             # [[0.585353535353535, 0.0853535353535354], [0.0853535353535354, 0.114494439342924]],
                             # u_max = 8
                             # [[0.584656084656085, 0.084656084656085], [0.084656084656085, 0.113322695333277]],
                             # [[0.584656084656085, 0.084656084656085], [0.084656084656085, 0.113322695333277]],
                             ],
                         center=[
                             [0.0, 0.0],
                             [pi/2, 0.0],
                             [-pi/2, 0.0]
                         ]
)

_num_backup_sets_to_consider = 1
# _backup_set_order = [1, 2, 3]
# _backup_set_order = [1, 2, 3]
_backup_set_order = [1]
def get_backup_sets(env, obs_proc):
    backup_sets = [PendulumBackupSet(env, obs_proc) for _ in range(_num_backup_sets_to_consider)]
    for i in range(len(backup_sets)):
        backup_set_id = _backup_set_order[i] - 1
        backup_sets[i].initialize(init_dict=AttrDict(c=_backup_sets_dict['c'][backup_set_id],
                                                     p=_backup_sets_dict['p'][backup_set_id],
                                                     center=_backup_sets_dict['center'][backup_set_id]))

    return backup_sets


def get_safe_set(env, obs_proc):
    safe_set = PendulumSafeSet(env, obs_proc)
    safe_set.initialize(init_dict=safe_set_dict)
    return safe_set


# TODO: fix ac_lim
_backup_policies_dict = dict(
    gain=[
        [-3.0, -3.0],
        [-3.0, -3.0],
        [-3.0, -3.0],
    ],
    center=[
        [0.0, 0.0],
        [pi/2, 0.0],
        [-pi/2, 0.0],
    ],
    ac_lim=env_config.max_torque
)

def get_backup_policies():
    bkp_plcs = []
    for i in range(_num_backup_sets_to_consider):
        backup_policy_id = _backup_set_order[i] - 1
        bkp_plcs.append(
            PendulumBackupControl(
                gain=_backup_policies_dict['gain'][backup_policy_id],
                center=_backup_policies_dict['center'][backup_policy_id],
                ac_lim=_backup_policies_dict['ac_lim'])
        )
    return bkp_plcs

class PendulumDesiredPolicy:
    def act(self, obs):
        return torch.tensor([0.0]) if obs.ndim == 1 else torch.zeros(obs.size(0), 1)

# class PendulumDesiredPolicy:
#     def act(self, obs):
#         if obs.ndim == 1:
#             return torch.tensor([env_config.max_torque]) if obs[0] >= 0 else torch.tensor([-env_config.max_torque])
#         return torch.where(obs[:, 0] >= 0,
#                            env_config.max_torque,
#                            -env_config.max_torque
#                            ).view(-1, 1)

class PendulumObsProcBackupShield(PendulumObsProc):
    def __init__(self, env):
        super().__init__(env)

        self.env = env

        self._proc_keys_indices = dict(
            backup_set='_trig_to_theta',
            safe_set='_trig_to_theta',
            backup_policy='_trig_to_theta',
            shield='_trig_to_theta',
        )


class PendulumPlotter(CustomPlotter):

    def __init__(self, obs_proc):
        super().__init__(obs_proc)
        self._plot_schedule_by_episode = {'1': ['state_action_plot']}
        self._plot_schedule_by_itr = None
        self._plot_schedule_by_itr = {
            '0': ['h_contours'],
            '50': ['h_contours']
        }


    def _prep_obs(self, obs):
        if obs.shape[1] == 3:
            theta = np.arctan2(obs[..., 1], obs[..., 0])
            obs = np.hstack([theta, obs[..., 2]])
        return np.atleast_2d(obs)
        # logger.push_plot(np.concatenate((state.reshape(1, -1), ac.reshape(1, -1) * scale.ac_old_bounds[1]), axis=1), plt_key="sampler_plots")
        # logger.push_plot(state.reshape(1, -1), plt_key="sampler_plots", row_append=True)

    # def filter_push_action(self, ac):
    #     ac, ac_filtered = ac
    #     logger.push_plot(np.concatenate((ac.reshape(1, -1), ac_filtered.reshape(1, -1)), axis=1), plt_key="sampler_plots")

    def dump_state_action_plot(self, dump_dict):
        if 'u_backup' in self._data:
            data = self._make_data_array(['obs', 'ac', 'u_des', 'u_backup'])
            logger.dump_plot_with_key(
                data=data,
                custom_col_config_list=[[0], [1], [2, 3, 4]],
                plt_key="states_action_plots",
                filename='states_action_episode_%d' % dump_dict['episode'],
                columns=['theta', 'theta_dot', 'ac', 'u_des', 'u_backup'],
                plt_info=dict(
                    xlabel=r'Timestep',
                    ylabel=[r'$\theta$',
                            r'$\dot \theta$',
                            r'$u$'],
                    legend=[None,
                            None,
                            [r'$u$', r'$u_{\rm d}$', r'$u_{\rm b}$']
                            ]
                ))
            self._empty_data_by_key_list(['obs', 'ac', 'u_des', 'u_backup'])
            return
        data = self._make_data_array(['obs', 'ac', 'u_des'])
        logger.dump_plot_with_key(
            data=data,
            custom_col_config_list=[[0], [1], [2, 3]],
            plt_key="states_action_plots",
            filename='states_action_episode_%d' % dump_dict['episode'],
            columns=['theta', 'theta_dot', 'ac', 'u_des'],
            plt_info=dict(
                xlabel=r'Timestep',
                ylabel=[r'$\theta$',
                        r'$\dot \theta$',
                        r'$u$'],
                legend=[None,
                        None,
                        [r'$u$', r'$u_{\rm d}$']
                        ]
            ))
        self._empty_data_by_key_list(['obs', 'ac', 'u_des'])

    def dump_h_contours(self, dump_dict):
        backup_set_funcs = dump_dict['backup_set_funcs']
        safe_set_func = dump_dict['safe_set_func']
        viability_kernel_funcs = dump_dict['viability_kernel_funcs']

        S_b_label = r'\mathcal{S}_{\rm b'

        plot_zero_level_sets(
            functions=[safe_set_func, *backup_set_funcs,
                       *viability_kernel_funcs],
            funcs_are_torch=True,
            mesh_density=30,
            bounds=(-pi-0.1, pi+0.1)
            # legends=[r'$S_s$',
            #            *[fr'$S_b_{str(i+1)}$' for i in range(len(backup_set_funcs))],
            #            *[fr'$h_{str(i+1)}$' for i in range(len(backup_set_funcs))]],
            # legends=[r'\mathcal{S}_{\rm s}',
            #          *[fr'{S_b_label}_{str(i+1)} }}' for i in range(len(backup_set_funcs))],
            #          *[fr'h_{str(i+1)}' for i in range(len(viability_kernel_funcs))],
            #          # r'\mathcal{S}'
            #          ],
            )
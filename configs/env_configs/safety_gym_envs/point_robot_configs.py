import numpy as np
from utils import scale
from configs.env_configs.safety_gym_envs.safety_gym_configs import SafetyGymSafeSetFromData, SafetyGymSafeSetFromCriteria
from utils.process_observation import ObsProc
from scipy.signal import lfilter
import torch
from scipy.spatial.transform import Rotation as R
from dynamics.nominal_dynamics import NominalDynamics
from attrdict import AttrDict
from math import ceil

config = AttrDict(
    do_obs_proc=True,
    safe_reset=False,
    w_0=0.1,
    w_m=0.1,
    ext=0.5,
    max_speed=10.0,
    sample_velocity_gaussian=True        # velocity distribution will be Gaussian with std = max_speed / 3
)

env_config = {
    'robot_base': 'xmls/point_m.xml',
    'sensors_obs': ['accelerometer', 'velocimeter', 'gyro', 'magnetometer', 'framepos', 'framequat', 'framexaxis', 'frameangvel'],
    'task': 'goal',
    'observe_goal_lidar': True,
    # 'observe_pillars': True,
    'observe_hazards': True,
    'observation_flatten': False,
    'lidar_max_dist': 3,
    'lidar_type': 'natural',
    # 'pillars_num': 10,
    'hazards_num': 7,
    'lidar_num_bins': 40,
    'render_lidar_size': 0.01,
    'hazards_keepout': 0.5,
    'randomize_layout': False   # TODO: you may need to add another key to Engine to randomize_goals on each reset while keeping the layout the same
}


class PointRobotSafeSetFromData(SafetyGymSafeSetFromData):
    def get_safe_action(self, obs):
        # expect obs: [pos_x, pos_y, pos_z, cos(theta), sin(theta), vel_x, vel_y, theta_dot, pts_on_obstacle (x,y,z)))
        # TODO: this can be wrong. check the implementation in FromCriteria
        pos = obs[:, :3]
        vec = obs[3:5]
        xy_vel = obs[5:7]
        vel = np.concatenate((xy_vel, np.zeros(xy_vel.shape[0])), axis=-1)
        pts_on_obstacles = obs[:, -3:]
        r = pos - pts_on_obstacles
        torque_dir = np.cross(r, vel)[:, -1]
        torque_dir = np.sign(torque_dir)
        force_dir = -np.sign(np.vdot(vec, r))
        ac_lim_high = scale.ac_old_bounds[1]

        force = (ac_lim_high[0] * force_dir).reshape(-1, 1)
        torque = (ac_lim_high[1] * torque_dir).reshape(-1, 1)
        ac = np.concatenate((force, torque), axis=-1)
        return ac.reshape(-1, 2, 1)

    def _is_ss_safe(self, r, v):
        # TODO: check this
        return (r[:, :2] * v[:, :2]).sum(-1) <= 0.0

    def _is_ss_unsafe(self, r, v):
        # TODO: check this
        return (r[:, :2] * v[:, :2]).sum(-1) > 0.0

class PointRobotSafeSetFromCriteria(SafetyGymSafeSetFromCriteria):
    def _get_obs(self):
        xmin, ymin, xmax, ymax = self.env.placements_extents
        xy = np.array([np.random.uniform(xmin - config.ext, xmax + config.ext),
                       np.random.uniform(ymin - config.ext, ymax + config.ext)])
        theta = np.random.uniform(low=-np.pi, high=np.pi)
        vec = np.array([np.cos(theta), np.sin(theta)])
        num_velocity = self.env.sim.model.nv
        if config.sample_velocity_gaussian:
            vel = np.random.normal(loc=0, scale=config.max_speed / 3, size=num_velocity)
        else:
            vel = np.random.uniform(low=-config.max_speed, high=config.max_speed, size=num_velocity)
        # FIXME: fix the order of terms
        return np.hstack((xy, vec, vel))

    def is_ss_safe(self, obs):
        if torch.is_tensor(obs):
            obs = obs.numpy()
        if obs.squeeze().ndim == 1:
            obs = obs.squeeze()
            r = self._get_vec_to_obstacle(obs[:2])
            v = obs[4:6]
            return (r * v).sum(-1) <= 0.0
        else:
            return [self.is_ss_safe(o) for o in obs]

    def get_safe_action(self, obs):
        # expect obs: [pos_x, pos_y, cos(theta), sin(theta), vel_x, vel_y, theta_dot))

        pos = obs[:, :2]
        vec = obs[:, 4:6]
        xy_vel = obs[:, 5:7]
        vel = np.concatenate((xy_vel, np.zeros((xy_vel.shape[0], 1))), axis=-1)
        r = self._get_vec_to_obstacle(pos)
        r = np.concatenate((r, np.zeros((r.shape[0], 1))), axis=-1)
        torque_dir = np.cross(r, vel)[:, -1]
        torque_dir = np.sign(torque_dir)
        force_dir = -np.sign(np.einsum('ij, ij -> i',vec, r[:, :2]))
        ac_lim_high = scale.ac_old_bounds[1]

        force = (ac_lim_high[0] * force_dir).reshape(-1, 1)
        torque = (ac_lim_high[1] * torque_dir).reshape(-1, 1)
        ac = np.concatenate((force, torque), axis=-1)
        return ac.reshape(-1, 2, 1)

    def _get_vec_to_obstacle(self, pos):
        if pos.squeeze().ndim == 1:
            pos = pos.squeeze()
            d = self.get_dist_to_obst(pos)
            closest_idx = np.argmin(d - self.obstacles['size'])
            closest_obst_pos = self.obstacles['pos'][closest_idx, :]
            r = closest_obst_pos[:-1] - pos
            r_norm = np.linalg.norm(r)
            r *= (r_norm - self.obstacles['size'][closest_idx]) / r_norm
            return r
        else:
            return np.vstack([self._get_vec_to_obstacle(p) for p in pos])

class PointObsProc(ObsProc):
    def __init__(self, env):
        super().__init__(env)
        self._proc_keys = ['buffer', 'mf', 'mb', 'filter']
        # self._unproc_keys = ['buffer']

        # make buffer keys: keys from observation that needs to be saved in the buffer
        sensor_keys = env_config['sensors_obs']
        obs_keys = self.env.observation_space.spaces.keys()
        lidar_keys = list(filter(lambda x: 'lidar' in x, obs_keys))
        self._buffer_keys = [*sensor_keys, *lidar_keys]     # buffer_keys is the list of all the observation keys that we want to save in buffer\
        self._save_buffer_indices_per_key()

        self._mf_keys = [*['accelerometer', 'velocimeter', 'gyro', 'magnetometer'], *lidar_keys] # train mf agent on sensor data
        self._mb_keys = ['framepos', 'framequat', 'velocimeter', 'frameangvel']
        self._filter_keys = [*['framepos', 'framequat'], *lidar_keys]
        # self._safe_set_keys = [*['framepos', 'framequat'], *lidar_keys]

    def obs_dim(self, proc_key=None):
        if proc_key == 'mb' or proc_key == 'filter':
            return int(7)
        else:
            return int(self._obs_dim_by_keys(getattr(self, f'_{proc_key}_keys')))

    # def _proc_for_buffer(self, obs, proc_dict=None):
    #     """ used in sampler to convert observation after calling env.step to numpy array saved in the buffer"""
    #     return self._flatten_obs_by_keys(obs, self._buffer_keys)

    def _proc_for_buffer(self, obs, proc_dict=None):
        """
        in the case where the safety-gym environment outputs observation as gym.spaces.Dict,
        there is no need to process the observation before pushing it to buffer.
        Observations are saved as is in the buffer.
        """
        return obs

    def _proc_for_mf(self, obs, proc_dict=None):
        """
        process observation for model-free training if observations are an instance of gym.spaces.Dict.
        Otherwise, it is assumed that the observation is already processed or it is in the processes form.
        """
        if self._is_dict_obs(obs):
            # if you made any modification to this so that the dimension of the processed observation is changed,
            # then you need to modify the observation_dim for mf in the obs_dim method above
            return self._flatten_obs_by_keys(obs, self._mf_keys)
        return obs

    def _proc_for_mb(self, obs, proc_dict=None):
        """
        process observation for model-based training if observations are an instance of gym.spaces.Dict.
        Otherwise, it is assumed that the observation is already processed or it is in the processes form.
        """

        if self._is_dict_obs(obs):
            # TODO: correct this
            obs_proc = self._flatten_obs_by_keys(obs, self._mb_keys)
            # obs_proc: [xyz_pos (3), quat (4), vel (3), angvel(3)]
            pos_xy = obs_proc[..., :2]
            quat = obs_proc[..., 3:7]
            theta = self._quat2theta(quat)
            theta_vec = np.array([np.cos(theta), np.sin(theta)]).reshape(theta.shape[0], 2)
            linvel_xy = obs_proc[..., 7:9]
            angvel_z = obs_proc[..., -1:]
            out = np.hstack([pos_xy, theta_vec, linvel_xy, angvel_z])
            return out
        return obs

    def _proc_for_filter(self, obs, proc_dict=None):
        """
        process observation for filter training if observations are an instance of gym.spaces.Dict.
        Otherwise, it is assumed that the observation is already processed or it is in the processes form.
        """
        return self._proc_for_mb(obs)

    # Helper functions
    def _filter_obs_by_keys(self, obs, keys):
        samples = []
        for k in sorted(keys):
            start = self._buffer_indices[k][0]
            end = self._buffer_indices[k][1]
            samples.append(obs[:, start:end])
        return np.hstack(samples)

    def _flatten_obs_by_keys(self, obs, keys):
        return np.hstack([obs[k] for k in keys])

    def _save_buffer_indices_per_key(self):
        obs_space_dict = self.env.observation_space.spaces
        sizes = [np.prod(obs_space_dict[k].shape) for k in sorted(self._buffer_keys)]
        indices = lfilter([1], [1, -1.0], sizes, axis=-1)
        start = np.zeros(indices.shape[0] + 1)
        start[1:] = indices[:]
        end = np.zeros(indices.shape[0] + 1)
        end[:-1] =indices[:]
        indices = np.vstack([start.reshape(1,-1), end.reshape(1,-1)]).astype(int)
        indices = indices.T
        self._buffer_indices = {k: indices[i, :] for i, k in enumerate(sorted(self._buffer_keys))}

    def _obs_dim_by_keys(self, keys):
        obs_space_dict = self.env.observation_space.spaces
        return sum(np.prod(obs_space_dict[k].shape) for k in keys)

    @staticmethod
    def _quat2theta(quat):
        r = R.from_quat(quat)
        vec = r.as_euler('zyx')
        theta = vec[..., -1]
        return theta

    @staticmethod
    def _is_dict_obs(obs):
        if isinstance(obs, dict):
            return True
        if isinstance(obs, list) or isinstance(obs, np.ndarray):
            if isinstance(obs[0], dict):
                return True
        return False


class PointRobotNominalDynamics(NominalDynamics):
    def initialize(self, params, init_dict=None):
        #TODO: You should link this with the params, so that if you are normalizing the observation or action, apply the same thing here
        self.m = 1.0        # FIXME: get this from env
        self.I = 1.0        # FIXME: get this from env
        self.cs = 0.01      # FIXME: get this from env
        self.cr = 0.005     # FIXME: get this from env
        self.com_loc = 0.01   # FIXME: get this from env

        self.continous_time = init_dict['is_continuous']

    def _predict(self, obs, ac, split_return=False):
        m = self.m
        I = self.I
        cs = self.cs
        cr = self.cr
        r = self.com_loc
        dt = self.timestep

        # TODO: Fix and check this: currently under the assumption that obs vector is [x, y, theta, x_dot, y_dot, theta_dot]
        x1, x2, x3, x4, x5, x6, x7 = None, None, None, None, None, None, None
        if obs.shape[-1] == 6:
            # [x, y, theta, x_dot, y_dot, theta_dot]
            x1 = obs[..., 0]  # x
            x2 = obs[..., 1]  # y
            x3 = np.cos(obs[..., 2])  # cos(theta)
            x4 = np.sin(obs[..., 2])  # sin(theta)
            x5 = obs[..., 3]  # x_dot
            x6 = obs[..., 4]  # y_dot
            x7 = obs[..., 5]  # theta_dot

        elif obs.shape[-1] == 7:
            # [x, y, cos(theta), sin(theta), x_dot, y_dot, theta_dot]

            x1 = obs[..., 0]  # x
            x2 = obs[..., 1]  # y
            x3 = obs[..., 2]  # cos(theta)
            x4 = obs[..., 3]  # sin(theta)
            x5 = obs[..., 4]  # x_dot
            x6 = obs[..., 5]  # y_dot
            x7 = obs[..., 6]  # theta_dot


        if not self.continous_time:
            raise NotImplementedError
        else:
            f_func = lambda x3, x4, x5, x6, x7: \
                np.array([
                    x5,
                    x6,
                    -x7 * x4,
                    x7 * x3,
                    -(cs / m) * (x5 - r * x7 * x4),
                    -(cs / m) * (x6 + r * x7 * x3),
                    -(cr + cs * r ** 2) / I * x7 - r * cs / I * (-x5 * x4 + x6 * x3)
                ], dtype=np.float32)
            G_func = lambda x3, x4: \
                np.array([
                    [0.0, 0.0],
                    [0.0, 0.0],
                    [0.0, 0.0],
                    [0.0, 0.0],
                    [x3 / m, 0.0],
                    [x4 / m, 0.0],
                    [0.0, 1 / I]
                ], dtype=np.float32)

            f = np.stack(list(map(f_func, x3, x4, x5, x6, x7)), axis=0)
            G = np.stack(list(map(G_func, x3, x4)), axis=0)
        return (f, G) if split_return else f + np.matmul(G, ac).squeeze(-1)


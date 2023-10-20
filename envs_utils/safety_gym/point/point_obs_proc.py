import numpy as np
from scipy.signal import lfilter
from scipy.spatial.transform import Rotation as R
from utils.process_observation import ObsProc
from envs_utils.safety_gym.point.point_configs import engine_config


class PointObsProc(ObsProc):
    def __init__(self, env):
        super().__init__(env)
        self._proc_keys = ['buffer', 'mf', 'mb', 'filter']
        # self._unproc_keys = ['buffer']

        # make buffer keys: keys from observation that needs to be saved in the buffer
        sensor_keys = engine_config['sensors_obs']
        obs_keys = self.env.observation_space.spaces.keys()
        lidar_keys = list(filter(lambda x: 'lidar' in x, obs_keys))
        self._buffer_keys = [*sensor_keys, *lidar_keys]     # buffer_keys is the list of all the observation keys that we want to save in buffer\
        self._save_buffer_indices_per_key()

        # self._mf_keys = [*['accelerometer', 'velocimeter', 'gyro', 'magnetometer'], *lidar_keys] # train mf agent on sensor data
        self._mf_keys = ['accelerometer', 'velocimeter', 'gyro', 'magnetometer', 'goal_lidar'] # train mf agent on sensor data
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
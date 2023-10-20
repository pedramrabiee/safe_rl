import numpy as np
import torch
from envs_utils.safety_gym.point.point_configs import env_config
from envs_utils.safety_gym.safety_gym_utils import SafetyGymSafeSetFromData, SafetyGymSafeSetFromCriteria
from utils import scale
from utils.seed import rng


class PointRobotSafeSetFromData(SafetyGymSafeSetFromData):
    def __init__(self, env, obs_proc):
        super().__init__(env, obs_proc)
        self.w_o = env_config.w_o
        self.w_m = env_config.w_m
        self.robot_keepout = env_config.robot_keepout
        self.max_speed = env_config.max_speed

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
        ac_lim_high = scale.ac_new_bounds[1]

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
    def __init__(self, env, obs_proc):
        super().__init__(env, obs_proc)
        self.w_o = env_config.w_o
        self.w_m = env_config.w_m
        self.robot_keepout = env_config.robot_keepout
        self.max_speed = env_config.max_speed

    def _get_obs(self):
        xmin, ymin, xmax, ymax = self.env.placements_extents
        xy = np.array([rng.uniform(xmin - env_config.ext, xmax + env_config.ext),
                       rng.uniform(ymin - env_config.ext, ymax + env_config.ext)])
        theta = rng.uniform(low=-np.pi, high=np.pi)
        vec = np.array([np.cos(theta), np.sin(theta)])
        num_velocity = self.env.sim.model.nv
        if env_config.sample_velocity_gaussian:
            vel = rng.normal(loc=0, scale=self.max_speed / 3, size=num_velocity)
        else:
            vel = rng.uniform(low=-self.max_speed, high=self.max_speed, size=num_velocity)
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
        force_dir = -np.sign(np.einsum('ij, ij -> i', vec, r[:, :2]))
        ac_lim_high = scale.ac_new_bounds[1]

        force = (ac_lim_high * force_dir).reshape(-1, 1)
        torque = (ac_lim_high * torque_dir).reshape(-1, 1)
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
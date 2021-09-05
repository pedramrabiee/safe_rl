import numpy as np
from utils.safe_set import SafeSetFromData, SafeSetFromCriteria
from utils.safety_gym_utils import *
from utils.misc import e_or
import torch


# TODO: add num_samples for each set in config
# TODO: add in_hazards somewhere
# TODO: add repmat to make multiple experience per position based on different velocity and/or orientations
# TODO: The indexing there is for the point robot


class SafetyGymSafeSetFromData(SafeSetFromData):
    def __init__(self, env, obs_proc):
        super().__init__(env, obs_proc)
        self.w_o = None             # populated in the specific robot __init__ method
        self.w_m = None             # populated in the specific robot __init__ method
        self.robot_keepout = None   # populated in the specific robot __init__ method
        self.max_speed = None       # populated in the specific robot __init__ method

    def make_in_safe(self, data):
        pos_samples = sample_on_rays_upto_stop(lidar_obs=data.lidar,
                                               num_samples=100,
                                               robot_pos=data.robot_pos,
                                               robot_mat=data.robot_mat,
                                               max_lidar_dist=3.0,
                                               stop=self.w_o+self.w_m+0.02)
        pos_samples = np.vstack(pos_samples)
        theta_samples = self._get_orientation(batch_size=pos_samples.shape[0])
        vel_samples = self._get_velocity(batch_size=pos_samples.shape[0])
        return np.concatenate((pos_samples, theta_samples, vel_samples), axis=-1)

    def make_mid_safe(self, data):
        pos_samples, _ = self._make_mid_pos(data)

        pos_samples = np.vstack(pos_samples)
        theta_samples = self._get_orientation(batch_size=pos_samples.shape[0])
        vel_samples = self._get_velocity(batch_size=pos_samples.shape[0])
        return np.concatenate((pos_samples, theta_samples, vel_samples), axis=-1)

    def make_unsafe(self, data):
        # TODO: only the case where not in hazard is implemented, implement in_hazard case later when you implemented in_hazard
        pos_samples = sample_on_rays_from_start_to_end(lidar_obs=data.lidar,
                                                       num_samples=100,
                                                       robot_pos=data.robot_pos,
                                                       robot_mat=data.robot_mat,
                                                       start=0.0,
                                                       end=1e-12)
        pos_samples = np.vstack(pos_samples)
        theta_samples = self._get_orientation(batch_size=pos_samples.shape[0])
        vel_samples = self._get_velocity(batch_size=pos_samples.shape[0])
        return np.concatenate((pos_samples, theta_samples, vel_samples), axis=-1)

    def make_out_cond_safe(self, data):
        pos_samples, pts_on_obstacles = self._make_out_pos(data)
        return self._make_cond(criterion=self._is_ss_safe, pos_samples=pos_samples, pts_on_obstacles=pts_on_obstacles)

    def make_out_cond_usafe(self, data):
        pos_samples, pts_on_obstacles = self._make_out_pos(data)
        return self._make_cond(criterion=self._is_ss_unsafe, pos_samples=pos_samples, pts_on_obstacles=pts_on_obstacles)

    def make_mid_cond_safe(self, data):
        pos_samples, pts_on_obstacles = self._make_mid_pos(data)
        return self._make_cond(criterion=self._is_ss_unsafe, pos_samples=pos_samples, pts_on_obstacles=pts_on_obstacles)

    # Helper Functions

    def _make_cond(self, criterion, pos_samples, pts_on_obstacles):
        num_samples = pos_samples.shape[0]
        r = pos_samples - pts_on_obstacles
        theta_samples = self._get_orientation(batch_size=num_samples)

        samples = []
        points = []
        leftovers = np.concatenate([pos_samples, theta_samples], axis=-1)
        while True:
            num_samples = leftovers.shape[0]
            vel_samples = self._get_velocity(batch_size=num_samples)
            leftovers = np.concatenate([leftovers, vel_samples], axis=-1)
            safe_mask = criterion(r, vel_samples)
            samples.append(np.concatenate([leftovers[safe_mask, ...], pts_on_obstacles[safe_mask, ...]], axis=-1))

            leftovers = leftovers[1 - safe_mask, ...]
            r = r[1 - safe_mask]
            if leftovers.shape[0] == 0:
                break

        return np.vstack(samples)

    def _make_out_pos(self, data):
        pos_samples, pts_on_obstacles = sample_point_on_normal(lidar_obs=data.lidar,
                                                               num_samples=10,
                                                               robot_pos=data.robot_pos,
                                                               robot_mat=data.robot_mat,
                                                               start=0.0 + 1e-12,
                                                               end=self.w_o,
                                                               output_points_on_obstacle=True)

        pos_samples = np.vstack(pos_samples)
        pts_on_obstacles = np.vstack(pts_on_obstacles)
        return pos_samples, pts_on_obstacles

    def _make_mid_pos(self, data):
        pos_samples, pts_on_obstacles = sample_point_on_normal(lidar_obs=data.lidar,
                                                               num_samples=10,
                                                               robot_pos=data.robot_pos,
                                                               robot_mat=data.robot_mat,
                                                               start=self.w_o+1e-12,
                                                               end=self.w_o+self.w_m,
                                                               output_points_on_obstacle=True)

        pos_samples = np.vstack(pos_samples)
        pts_on_obstacles = np.vstack(pts_on_obstacles)
        return pos_samples, pts_on_obstacles

    def _get_velocity(self, batch_size):
        num_velocity = self.env.sim.model.nv
        return np.random.uniform(low=-self.max_speed, high=self.max_speed, size=(batch_size, num_velocity))

    def _get_orientation(self, batch_size):
        theta = np.random.uniform(low=-np.pi, high=np.pi, size=(batch_size, 1))
        return np.hstack([np.cos(theta), np.sin(theta)])

class SafetyGymSafeSetFromCriteria(SafeSetFromCriteria):
    def __init__(self, env, obs_proc):
        super().__init__(env, obs_proc)

        # TODO: Add these items to the obstacle list and modify the criteria methods to work with these
        #  kind of obstacles too:
        #  walls: not circular
        #  vases: not circular. you may need to consider a circle that encompasses the vases
        #  gremlins: these are moving objects. Not considered in the current RL-CBF implementations
        obstacles_names = ['hazards', 'pillars']

        # make a list of obstacles info
        self.obstacles = self._make_obstacles(obstacles_names)
        self.w_o = None             # populated in the specific robot __init__ method
        self.w_m = None             # populated in the specific robot __init__ method
        self.robot_keepout = None   # populated in the specific robot __init__ method
        self.max_speed = None       # populated in the specific robot __init__ method


    def _make_obstacles(self, obstacles_names):
        obst_pos = []
        obst_size = []
        for obst_name in obstacles_names:
            obst_num = getattr(self.env, f'{obst_name}_num')
            if obst_num:
                obst_pos.append(getattr(self.env, f'{obst_name}_pos'))
                obst_size.append(getattr(self.env, f'{obst_name}_size') * np.ones(obst_num))

        obst_pos = np.vstack(obst_pos)
        obst_size = np.hstack(obst_size)
        return dict(pos=obst_pos, size=obst_size)

    @staticmethod
    def _get_dist_to_obst(xy, obst_xy):
        return np.linalg.norm(obst_xy - xy, axis=-1)

    def get_dist_to_obst(self, xy):
        return self._get_dist_to_obst(xy, self.obstacles['pos'][:, :-1])

    def is_in_safe(self, obs):  # geometrically inside the inner safe section
        return self._check_criterion(obs,
                                     lambda d: all(d - self.obstacles['size'] - self.robot_keepout - self.w_o - self.w_m > 0))

    def is_mid_safe(self, obs):  # geometrically inside the middle safe section
        return self._check_criterion(obs,
                                     lambda d: all(d - self.obstacles['size'] - self.robot_keepout - self.w_o > 0) and
                                               any(d - self.obstacles['size'] - self.robot_keepout - self.w_o - self.w_m <= 0))

    def is_out_safe(self, obs):  # geometrically inside the outer safe section
        return self._check_criterion(obs,
                                     lambda d: all(d - self.obstacles['size'] - self.robot_keepout > 0) and
                                               any(d - self.obstacles['size'] - self.robot_keepout - self.w_o <= 0))

    def is_unsafe(self, obs):
        return self._check_criterion(obs,
                                     lambda d: any(d - self.obstacles['size'] <= 0))

    def is_geo_safe(self, obs):
        return e_or(self.is_in_safe(obs), self.is_mid_safe(obs), self.is_out_safe(obs))

    def _check_criterion(self, obs, criterion):
        if torch.is_tensor(obs):
            obs = obs.numpy()
        if obs.squeeze().ndim == 1:
            obs = obs.squeeze()
            d = self.get_dist_to_obst(obs[:2])
            return criterion(d)
        else:
            return [self._check_criterion(o, criterion) for o in obs]
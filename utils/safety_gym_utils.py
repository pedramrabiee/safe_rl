from utils.misc import theta2vec
import numpy as np
import os.path as osp
import shutil
from utils.seed import rng


def polar2xy(r_list, theta):
    return [theta2vec(theta, add_z_axis=True) * r for r in r_list]


def lidar2polar(lidar_obs, inc_unpop_rays=False):
    lidar_thetas = np.linspace(0, 2 * np.pi, lidar_obs.shape[0], endpoint=False)

    # unnormalize distances
    if inc_unpop_rays:
        lidar_dist = -np.log(lidar_obs+1e-24)
    else:
        populated_mask = lidar_obs > 0.0
        lidar_dist = -np.log(lidar_obs[populated_mask]+1e-24)
        lidar_thetas = lidar_thetas[populated_mask]
    return lidar_dist, lidar_thetas


def lidar2localpos(lidar_obs, inc_unpop_rays=False):
    lidar_dist, lidar_thetas = lidar2polar(lidar_obs, inc_unpop_rays)

    # egocenteric position of objects
    pos = [theta2vec(theta, add_z_axis=True) * dist for theta, dist in zip(lidar_thetas, lidar_dist)]
    return pos


def egocen_pos2global_pos(pos, robot_pos, robot_mat):
    return [np.matmul(robot_mat, p) + robot_pos for p in pos]


def lidar2globalpos(lidar_obs, robot_pos, robot_mat):
    pos = lidar2localpos(lidar_obs)
    # objects' positions in global frame
    global_pos = egocen_pos2global_pos(pos=pos,
                                       robot_pos=robot_pos,
                                       robot_mat=robot_mat)
    return global_pos


def sample_on_rays_from_start_to_end(lidar_obs, num_samples,
                                     robot_pos, robot_mat,
                                     start=0.0, end=0.0):
    lidar_dist, lidar_thetas = lidar2polar(lidar_obs, inc_unpop_rays=False)

    samples = []
    num_acceptable_rays = lidar_dist.shape[0]
    if num_acceptable_rays:
        num_samples_per_ray = int(num_samples / num_acceptable_rays)
        for i in range(num_acceptable_rays):
            if lidar_dist[i] - end > 0:
                sample_section = rng.rand(num_samples_per_ray) * (end - start) + start
                dist_on_ray = lidar_dist[i] - sample_section
                samples += polar2xy(dist_on_ray, lidar_thetas[i])

    global_pos = egocen_pos2global_pos(pos=samples,
                                       robot_pos=robot_pos,
                                       robot_mat=robot_mat)

    return global_pos

def sample_on_rays_upto_stop(lidar_obs, num_samples,
                             robot_pos, robot_mat,
                             max_lidar_dist, stop=0.0):
    num_rays = lidar_obs.shape[0]
    lidar_dist, lidar_thetas = lidar2polar(lidar_obs, inc_unpop_rays=True)

    samples = []
    min_lidar_dist = lidar_dist.min()
    min_lidar_dist = min(min_lidar_dist, max_lidar_dist)

    num_samples_per_ray = int(num_samples / num_rays)
    if min_lidar_dist - stop >= 0:
        for i in range(num_rays):
            sample_section = rng.rand(num_samples_per_ray) * (min_lidar_dist - stop) + stop
            dist_on_ray = min_lidar_dist - sample_section
            samples += polar2xy(dist_on_ray, lidar_thetas[i])

    global_pos = egocen_pos2global_pos(pos=samples,
                                       robot_pos=robot_pos,
                                       robot_mat=robot_mat)
    return global_pos


def sample_on_rays_upto_obstacle(lidar_obs, num_samples, robot_pos, robot_mat):
    num_rays = lidar_obs.shape[0]

    lidar_dist, lidar_thetas = lidar2polar(lidar_obs, inc_unpop_rays=True)

    samples = []

    num_samples_per_ray = int(num_samples / num_rays)
    for i in range(num_rays):
        dist_on_ray = rng.rand(num_samples_per_ray) * lidar_dist[i]
        samples += polar2xy(dist_on_ray, lidar_thetas[i])

    global_pos = egocen_pos2global_pos(pos=samples,
                                       robot_pos=robot_pos,
                                       robot_mat=robot_mat)
    return global_pos


def get_normal_to_obstacles_from_lidar(lidar_obs, robot_pos, robot_mat, max_dist_to_find_normal=1.0,
                                       max_gap_between_pts_on_obstacle=0.1):
    num_rays = lidar_obs.shape[0]
    lidar_thetas = np.linspace(0, 2 * np.pi, num_rays, endpoint=False)

    # unnormalize distances
    populated_mask = lidar_obs > 0.0
    lidar_dist = -np.log(lidar_obs+1e-24)
    # lidar_thetas = lidar_thetas[populated_mask]
    normals = []
    points = []
    # num_pts = lidar_dist.shape[0]


    for i in range(num_rays):
        next_id = i+1 if (i+1) < num_rays else 0
        # initialize the normal to be None
        # We are looking for two successive point on the same object,
        # if there is an unpopulated rays in between two rays,
        # it means that the two points are not on the same object
        if not populated_mask[i] or not populated_mask[next_id]:
            continue

        # The greater the distance to obstacle, the larger the error in the normal direciton
        # Hence, we filter out points with distance > max_dist_to_find_normal
        if lidar_dist[i] > max_dist_to_find_normal or lidar_dist[next_id] > max_dist_to_find_normal:
            continue

        # if the gap between to successive pts on lidar > max_gap_between_pts_on_obstacle
        if np.abs(lidar_dist[i] - lidar_dist[next_id]) > max_gap_between_pts_on_obstacle:
            continue
        # TODO: Clean this part
        # Find the normal direction
        p1 = theta2vec(lidar_thetas[i], add_z_axis=True) * lidar_dist[i]
        p2 = theta2vec(lidar_thetas[next_id], add_z_axis=True) * lidar_dist[next_id]

        p1 = egocen_pos2global_pos(pos=[p1],
                                   robot_pos=robot_pos,
                                   robot_mat=robot_mat)[0]

        p2 = egocen_pos2global_pos([p2],
                                   robot_pos=robot_pos,
                                   robot_mat=robot_mat)[0]

        v = p2 - p1
        n = np.array([-v[1], v[0]])
        n /= np.linalg.norm(n)
        n = n if np.dot(n, robot_pos[:2] - p1[:2]) >= 0 else -n

        # overwrite normals
        normals.append(n)
        points.append(p1)

    return normals, points


def is_in_hazards(env, robot_pos):
    if env.hazards_num > 0:
        robot_pos = robot_pos[:-1]
        hazards_size = env.hazards_size
        hazards_pos = np.array(env.hazards_pos)[:, :-1]
        dist_to_hazards = np.linalg.norm(hazards_pos - robot_pos, axis=-1)
        if dist_to_hazards.min() < hazards_size:
            return True
    return False


def sample_on_normal(direction, point, num_sample, start, end):
    dist_on_dir = rng.rand(num_sample) * (end - start) + start
    out = np.zeros((num_sample, 3))
    out[:, 0] = dist_on_dir * direction[0] + point[0]
    out[:, 1] = dist_on_dir * direction[1] + point[1]
    out[:, 2] = point[2]

    return out


def sample_point_on_normal(lidar_obs,
                           num_samples,
                           robot_pos,
                           robot_mat,
                           end,
                           start=0.0,
                           max_dist_to_find_normal=2.0,
                           max_gap_between_pts_on_obstacle=0.1,
                           output_points_on_obstacle=False):

    normals, pts_on_obstacle = get_normal_to_obstacles_from_lidar(lidar_obs, robot_pos, robot_mat,
                                                                  max_dist_to_find_normal=max_dist_to_find_normal,
                                                                  max_gap_between_pts_on_obstacle=max_gap_between_pts_on_obstacle)

    samples = []
    points = []

    for n, p in zip(normals, pts_on_obstacle):
        sample = sample_on_normal(direction=n,
                                  point=p,
                                  num_sample=num_samples,
                                  start=start,
                                  end=end)
        samples.append(sample)
        points.append(np.tile(p, [num_samples, 1]))

    if output_points_on_obstacle:
        return samples, points
    else:
        return samples

def make_obstacles_location_dict(env):
        if hasattr(env, 'robot'):
            pass
        elif hasattr(env.unwrapped, 'robot'):   # used in safety_gym environments
            env = env.unwrapped

        def xy_from_pos(pos):
            return [p[:-1] for p in pos]

        locations = {}
        if env.hazards_num:
            locations.update({'hazards_locations': xy_from_pos(env.hazards_pos)})
        if env.vases_num:
            locations.update({'vases_locations': xy_from_pos(env.vases_pos)})
        if env.pillars_num:
            locations.update({'pillars_locations': xy_from_pos(env.vases_pos)})
        if env.gremlins_num:
            locations.update({'gremlins_locations': xy_from_pos(env.gremlins_pos)})
        if env.walls_num:
            locations.update({'walls_locations': xy_from_pos(env.walls_pos)})

        return locations

def save_mujoco_xml_file(xml_path, save_dir):
    import safety_gym
    BASE_DIR = osp.dirname(safety_gym.__file__)
    source_file = osp.join(BASE_DIR, xml_path)
    filename = xml_path[5:]     # xml_path starts with xmls/
    filename = osp.join(save_dir, filename)
    shutil.copyfile(source_file, filename)



def get_engine_config(env_id):
    if env_id == 'Point':
        from envs_utils.safety_gym.point_robot_configs import engine_config
    else:
        raise NotImplementedError
    return engine_config

from attrdict import AttrDict
from point_utils import moment_of_inertia

config = {
    'init':{
        'max_episode_time': 3.0,
        'max_episode_time_eval': 2.0,
        'plot_custom_figs': False,
        'save_custom_figs_data': False,
        'episode_steps_per_itr': 10,
        'n_training_episode': 19,
        'do_evaluation': True,
        'n_episodes_evaluation': 1,
        'num_evaluation_sessions': 4,
        'n_video_save_per_evaluation': 1,
        'wandb_project_name': 'point',
        'save_models': False,
    },
    'sf_params':{
        'filter_pretrain_sample_size': 500,
        'mf_update_freq': 10,
        'safety_filter_is_on': True,
        'filter_pretrain_is_on': True,
        'filter_train_is_on': True,
    },
    'cbf_params': {
        'pretrain_max_epoch': 1e2,
        'train_on_jacobian': True,
        'pretrain_batch_to_sample_ratio': 0.2,
        # losses weights
        'safe_loss_weight': 1.0,
        'unsafe_loss_weight': 1.0,
        'deriv_loss_weight': 0.1,
        'safe_deriv_loss_weight': 1.0,
        'u_max_weight_in_deriv_loss': 1.0,
        'deriv_loss_version': 3,
        'loss_tanh_normalization': False
    }
}

env_config = AttrDict(
    do_obs_proc=True,
    safe_reset=True, # if true, robot is located in in_safe set
    timestep=0.002,
    integrator='RK4',
    w_o=0.1,
    w_m=0.1,
    robot_keepout=0.15,
    ext=0.5,
    max_speed=10.0,
    sample_velocity_gaussian=True,        # velocity distribution will be Gaussian with std = max_speed / 3
    use_same_layout_for_eval=True,
    c_slide=0.01,
    c_hinge=0.005,
    density=1.0,
    r_sphere=0.1,
    l_cube=2*0.05,
)


obstacle_config = AttrDict(

    )

engine_config = dict(
    robot_base= 'xmls/point_m.xml',
    # sensors_obs= ['accelerometer', 'velocimeter', 'gyro', 'magnetometer', 'framepos', 'framequat', 'framexaxis', 'frameangvel'],
    task='goal',
    hazards_num=6,
    hazards_size=0.3,     # is this the radius? yes ok man miram unvar safe set besazam
    hazards_pos=[[0.2,0.7],
                 [4.8,4.5],
                 [2.3,1.1],
                 [1.6,3.9],
                 [4.9,0.2],
                 [2.4,3.6]],
    hazards_keepout=0.5,
    randomize_layout=False,
    # observe_goal_lidar=True,
    # observe_hazards=True,
    # observation_flatten=False,
    lidar_max_dist=3,
    lidar_type='natural',
    lidar_num_bins=16,
    render_lidar_size=0.01,
    frameskip_binom_n=10,
    num_steps=1000000,

)




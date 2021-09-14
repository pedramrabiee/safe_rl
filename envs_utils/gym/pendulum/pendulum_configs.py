from attrdict import AttrDict

config = {
    'init': {
        'use_custom_env': True,
        'max_episode_time': 10.0,
        'max_episode_time_eval': 10.0,
        'plot_custom_figs': True,
        'save_custom_figs_data': True,
        'episode_steps_per_itr': 1,
        'n_training_episode': 39,
        'do_evaluation': False,
        'n_episodes_evaluation': 5,
        'num_evaluation_sessions': 4,
        'n_video_save_per_evaluation': 3,
        'wandb_project_name': 'pendulum',
        'save_models': True,
    },
    'sf_params': {
        'filter_pretrain_sample_size': 500,
        'mf_update_freq': 1,
        'filter_training_stages': dict(stages=[5000, 10000, 15000],
                                       freq=[5000, 4000, 4000]),
        'safety_filter_is_on': True,
        'filter_pretrain_is_on': True,
        'filter_train_is_on': True,
    },
    'cbf_params': {
        'pretrain_max_epoch': 1e5,
        'stop_criteria_eps': 0.0,
        'train_on_jacobian': True,
        'pretrain_batch_to_sample_ratio': 0.2,
        # losses weights
        'safe_loss_weight': 1.0,
        'unsafe_loss_weight': 1.0,
        'deriv_loss_weight': 0.001,
        'safe_deriv_loss_weight': 1.0,
        'u_max_weight_in_deriv_loss': 1.0,
        'deriv_loss_version': 3,
        'loss_tanh_normalization': False,
        'gamma_safe': 0.0,
        'gamma_unsafe': 0.0,
        'gamma_dh': 0.0
    }
}

env_config = AttrDict(
    do_obs_proc=False,
    safe_reset=True,
    timestep=0.01,
    # Safe set width
    half_wedge_angle=1.5,
    # mid_safe_set_width=0.2,
    outer_safe_set_width=0.2,
    # Pendulum dynamics parameters
    g=10.0,
    m=1.0,
    l=1.0,
    max_torque=60.0,
    max_speed=8.0,
    sample_velocity_gaussian=True       # velocity distribution will be truncated normal distribution
)
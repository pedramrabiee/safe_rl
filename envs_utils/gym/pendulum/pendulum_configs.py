from attrdict import AttrDict
import numpy as np


config = {
    'init': {
        'debugging_mode': False,
        'use_custom_env': True,
        'max_episode_time': 5.0,
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
        'load_models': False,
        'load_buffer': True,
        'n_episode_init_phase': 4,
        'step_save_freq': 1,
        'load_run_name': 'run-20210914_160746-a7mp6vyc',
        'load_timestamp': '20210914_161734_ep0',

    },
    'ddpg_params': {
        'init_phase_coef': 0.5
    },
    'sf_params': {
        'filter_pretrain_sample_size': 20000,
        'mf_update_freq': 1,
        'filter_training_stages': dict(stages=[4, 10, 20],
                                       freq=[1, 1, 1]),
        'safety_filter_is_on': True,
        'filter_pretrain_is_on': True,
        'filter_train_is_on': False,
        'save_filter_and_buffer_after_pretraining': True,
        'ep_to_start_appending_cbf_deriv_loss_data': 4,
    },
    'cbf_params': {
        'pretrain_max_epoch': 1e5,
        'stop_criteria_eps': 0.0,
        'train_on_jacobian': True,
        'pretrain_batch_to_sample_ratio': 0.2,
        # losses weights
        'safe_loss_weight': 1.0,
        'unsafe_loss_weight': 1.0,
        'deriv_loss_weight': 0.1,
        'safe_deriv_loss_weight': 0.5,
        'u_max_weight_in_deriv_loss': 1.0,
        'deriv_loss_version': 3,
        'loss_tanh_normalization': False,
        'gamma_safe': 0.0,
        'gamma_unsafe': 0.0,
        'gamma_dh': 0.0,
        'use_filter_just_for_plot': False,
        'train_on_max': True
    }
}

env_config = AttrDict(
    do_obs_proc=False,
    safe_reset=False,
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
    max_speed=np.inf,
    max_speed_for_safe_set_training=20.0,
    max_T_for_safe_set=100,
    sample_velocity_gaussian=True       # velocity distribution will be truncated normal distribution
)
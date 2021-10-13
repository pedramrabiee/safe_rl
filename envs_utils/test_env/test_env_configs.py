from attrdict import AttrDict
import numpy as np
import torch.nn as nn


config = {
    'init': {
        'debugging_mode': False,
        'max_episode_time': 20.0,
        'plot_custom_figs': True,
        'save_custom_figs_data': True,
        'episode_steps_per_itr': 2000,
        'n_training_episode': 20,
        'do_evaluation': False,
        'n_episodes_evaluation': 5,
        'num_evaluation_sessions': 4,
        'n_video_save_per_evaluation': 3,
        'wandb_project_name': 'cbf_test',
        'save_models': True,
        'save_buffer': True,
        'step_save_freq': 1,
        'load_models': True,
        'load_buffer': True,
        'load_run_name': 'run-20211005_144305-1149d4zz',
        'load_timestamp': '20211005_145551_ep17',
        'custom_load_list': ['safety_filter'],

    },
    'sf_params': {
        'mf': 'cbftest_agent',
        'filter_pretrain_sample_size': 2000,
        'mf_update_freq': 2000,
        'filter_training_stages': dict(stages=[1, 10, 20],
                                       freq=[1, 1, 1]),
        'safety_filter_is_on': True,
        'filter_pretrain_is_on': False,
        'filter_train_is_on': False,
        'mf_train_is_on': False,
        'ep_to_start_appending_cbf_deriv_loss_data': 0,
    },
    'cbf_params': {
        'eta': 100.0,
        'pretrain_max_epoch': 1000,
        'max_epoch': 100,
        'stop_criteria_eps': 0.0,
        'train_on_jacobian': True,
        'pretrain_batch_to_sample_ratio': 0.2,
        # losses weights
        'safe_loss_weight': 1.0,
        'unsafe_loss_weight': 2.0,
        'deriv_loss_weight': 0.0,
        'safe_deriv_loss_weight': 1.0,
        'u_max_weight_in_deriv_loss': 1.0,
        'deriv_loss_version': 3,
        'loss_tanh_normalization': False,
        'use_filter_just_for_plot': False,
        'gamma_safe': 0.1,
        'gamma_unsafe': 0.1,
        'gamma_dh': 0.0,
        'constrain_control': True,
        'train_on_max': True,
        'k_epsilon': 1e48,

    }
}

env_config = AttrDict(
    max_episode_time=20.0,
    do_obs_proc=False,
    safe_reset=True,
    timestep=0.01,
    max_u=20.0,
    max_u_for_safe_set=20.0,
    max_speed=np.inf,
    max_speed_for_safe_set_training=20.0,
    max_x=40.0,
    max_x_for_safe_set=12,
    min_x_for_safe_set=-4.0,
    max_x_safe=5.5,
    min_x_safe=2.5,
    out_width_up=0.3,
    out_width_down=0.45,
    m=1.0,
    k=1.0,
    c=1.0,
    omega=0.5,
    command_amplitude=2.0,
    max_T_for_safe_set=100,
    fixed_reset=False
)
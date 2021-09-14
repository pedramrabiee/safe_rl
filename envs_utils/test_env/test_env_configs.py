from attrdict import AttrDict

config = {
    'init': {
        'max_episode_time': 20.0,
        'plot_custom_figs': False,
        'save_custom_figs_data': False,
        'episode_steps_per_itr': 2000,
        'n_training_episode': 9,
        'do_evaluation': False,
        'n_episodes_evaluation': 5,
        'num_evaluation_sessions': 4,
        'n_video_save_per_evaluation': 3,
        'wandb_project_name': 'cbf_test',
        'save_models': False,
    },
    'sf_params': {
        'mf': 'cbftest_agent',
        'filter_pretrain_sample_size': 1000,
        'mf_update_freq': 2000,
        'filter_training_stages': dict(stages=[2000, 10000, 15000],
                                       freq=[2000, 2000, 2000]),
        'safety_filter_is_on': False,
        'filter_pretrain_is_on': False,
        'filter_train_is_on': False,
        'mf_train_is_on': False
    },
    'cbf_params': {
        'pretrain_max_epoch': 1e4,
        'stop_criteria_eps': 0.0,
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
    max_episode_time=20.0,
    do_obs_proc=False,
    safe_reset=True,
    timestep=0.01,
    max_speed=2.0,
    max_x=3.0,
    max_x_safe=1.1,
    out_width=0.2,
    max_u=10.0,
    m=1.0,
    k=1.0,
    c=1.0)
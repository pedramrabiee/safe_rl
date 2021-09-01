import torch.nn as nn
import torch.optim as optim
from controller.random_controller import RandomController
from controller.random_shoot_controller import RandomShootController
from controller.cem_controller import CEMController
from attrdict import AttrDict


class Config:
    def __init__(self):
        self.setup = None       # populated in base_trainer __init__
        self.env_spec_config = None     # populated in base_trainer __init__: modify this in the config dictionary at the beginning of environment specific config file

        # ENVIRONMENT
        self.ac_lim = (-1.0, 1.0)
        self.use_custom_env = True              # implement your customized(/modified) env in utils/make_env

        # UNCOMMENT line below for custom max_episode_len
        self.max_episode_time = 25.0
        self.max_episode_time_eval = 2.0

        # MODE
        self.resume = False
        self.benchmark = False
        self.evaluation_mode = False
        self.debugging_mode = False             # Turns wandb logging off/ saves nothing to files
        self.plot_custom_figs = False
        self.save_custom_figs_data = False

        assert (self.resume + self.benchmark + self.evaluation_mode) < 2, 'Only one of resume, benchmark, or evaluation mode can be True'

        # TRAINER
        self.episode_steps_per_itr = 1      # number of timesteps to collect data between model updates
        self.n_training_episode = 9
        self.seed = 2 ** 32 + 475896325
        self.buffer_size = 1e6

        # TRAINING
        self.n_training_processes = 1
        self.training_device = 'cpu'

        # SAMPLER
        self.sampling_batch_size = 'all'                    # TODO: you need to remove this and move the sampling batch_size into method parameters. Like in the case of sf agent and trainer
        self.n_episode_initial_data_collection = 1

        self.sampler_device = 'cpu'
        self.episodes_per_return_log = 1
        self.add_noise_when_buffering = AttrDict(is_true=False,
                                                 data=['obs', 'next_obs'],
                                                 noise_to_signal=0.01)

        # EVALUATION
        self.do_evaluation = False
        self.n_episodes_evaluation = 5
        self.num_evaluation_sessions = 1    # number of evaluation sessions
        self.n_video_save_per_evaluation = 2
        self.n_evaluation_processes = 1
        self.evaluation_device = 'cpu'
        self.n_serial_envs_evaluation = 8  # for use in DummyVecEnv, only used when n_sampler_processes = 1
        self.save_video = True and not self.debugging_mode  # do not need to save video on debugging mode

        # LOG AND SAVE
        self.results_dir = 'results'
        self.use_wandb = True  # Enable wandb
        self.wandb_project_name = "cbf"  # wandb project name
        self.save_models = False and not self.debugging_mode    # Enable to save models every SAVE_FREQUENCY episodes (do not need to save on debugging mode)
        self.save_buffer = False
        self.num_save_sessions = 10
        self.add_timestamp = True  # Add timestamp to console's

        # LOAD MODELS
        self.load_models = False
        self.load_buffer = False
        self.load_run_name = 'run-20210628_125253-37ijc4uh'
        self.load_run_id = self.load_run_name[-8:]
        self.overwrite_config = False            # overwrite config file, with the loaded model config file
        # self.load_timestamp = '20210110_131125'
        self.load_timestamp = 'last'

        # Custom model loader:
        # Make a list of dict keys that you want to load from agent's models.
        # To load all models, set this to None
        self.custom_load_list = None
        # self.custom_load_list = ['mf_agent']

    ##############################
    ###### Agent params
    ##############################
    # Model-based
    def get_mb_params(self):
        from dynamics.gaussian_nn_dynamics import GaussianDynamics
        from dynamics.deterministic_nn_dynamics import DeterministicDynamics
        from dynamics.gaussian_processes_dynamics import GPDynamics
        from explorations.rand_noise import RandNoise
        from buffers.replay_buffer import ReplayBuffer
        from dynamics.custom_dyns.affine_in_action import AffineInActionDeterministic, AffineInActionGaussian, AffineInActionGP
        from dynamics.affine_in_action_gaussian_dynamics import AffineInActionGaussianDynamics


        controller_cls = [RandomController, CEMController]
        replay_buffer_cls = ReplayBuffer
        self.mb_params = AttrDict(
            # dynamics
            replay_buffer_cls=replay_buffer_cls,
            is_ensemble=False,
            dynamics_cls=AffineInActionGaussianDynamics,
            dynamics_params=AttrDict(holdout_ratio=0.1,
                                     epochs=20,
                                     batch_size=512,
                                     is_bootstrap=False,
                                     dynamics_net_kwargs=dict(hidden_dim=200,
                                                              num_layers=2,
                                                              unit_activation=nn.SiLU,
                                                              out_activation=nn.Identity,
                                                              batch_norm=False,
                                                              layer_norm=False,
                                                              batch_norm_first_layer=False),
                                     optim_cls=optim.Adam,
                                     optim_kwargs=dict(lr=1e-3,
                                                       weight_decay=1e-2),
                                     ensemble_size=5,  # only used if is_ensemble is True
                                     is_probabilistic=False,
                                     delta_output=True,
                                     normalized_io=False,
                                     gp_train_freq=5,
                                     # Nominal dynamics settings
                                     use_nominal_dyn=True,
                                     # Custom dynamics architecture settings
                                     use_custom_dyn=True,
                                     custom_dyn_cls=AffineInActionGaussian,
                                     # Continuou/Discrete-time dynamics settings
                                     train_continuous_time=True,
                                     ),
            # Preprocess Observation
            # set this to None, if you don't want to preprocess observation for dynamics training
            controller_cls=controller_cls,
            controller_params=self.get_controller_params(controller_cls),
            exp_strategy_cls=RandNoise,
            exp_kwargs=dict(dist='normal',
                            dist_kwargs=dict(loc=0, scale=0.1)),
            phase_2_usage_frac=0.9,
        )
        return self.mb_params

    # DDPG
    def get_ddpg_params(self):
        from explorations.ou_noise import OUNoise
        self.ddpg_params = AttrDict(
            tau=0.001,
            gamma=0.99,
            exp_strategy_cls=OUNoise,
            n_exploration_episode=1000,
            exp_kwargs=dict(theta=0.15,
                            sigma=0.2,
                            mu=0.0),
            init_noise_scale=0.3,
            final_noise_scale=0.0,
            # policy network
            pi_net_kwargs=dict(hidden_dim=128,
                               num_layers=2,
                               unit_activation=nn.ReLU,
                               out_activation=nn.Tanh,
                               batch_norm=False,
                               layer_norm=True,
                               batch_norm_first_layer=False,
                               out_layer_initialize_small=True),
            pi_optim_cls=optim.Adam,
            pi_optim_kwargs=dict(lr=1e-4,
                                 weight_decay=0),
            # grad clip
            use_clip_grad_norm=True,
            clip_max_norm=0.5,
            # q network
            q_net_kwargs=dict(hidden_dim=128,
                              num_layers=2,
                              unit_activation=nn.ReLU,
                              out_activation=nn.Identity,
                              batch_norm=False,
                              layer_norm=True,
                              batch_norm_first_layer=False,
                              out_layer_initialize_small=True),
            q_optim_cls=optim.Adam,
            q_optim_kwargs=dict(lr=1e-3,
                                weight_decay=0),
            multi_in_critic=True,
            multi_in_critic_kwargs=dict(in2_cat_layer=1),
            net_updates_per_iter=5,     # currently only used in ddpg_trainer
        )
        return self.ddpg_params

    # Safety Filter
    def get_sf_params(self):
        self.sf_params = AttrDict(
            # models
            mf='ddpg',
            mb='mb',
            filter='cbf',
            # batch sizes
            mf_train_batch_size=128,
            mb_train_batch_size='all',
            filter_train_batch_size=1024,
            filter_initial_training_batch_size=6000,
            # update frequencies
            mf_update_freq=1,
            mb_update_freq=10000,
            filter_update_freq=200,     # this option is not currently used in the sf_trainer, use filter_training_stages instead
            filter_training_stages=dict(stages=[5000, 10000, 20000],
                                        freq=[2500, 500, 1000]),
            # misc.
            safety_filter_is_on=True,
            filter_pretrain_is_on=True,
            filter_train_is_on=False,
            dyn_train_is_on=False,
            mf_train_is_on=True,

        )
        return self.sf_params

    ##############################
    ###### Filters
    ##############################
    # CBF Filter
    def get_cbf_filter_params(self):
        self.cbf_params = AttrDict(
            # filter network
            filter_net_kwargs=dict(hidden_dim=128,
                                   num_layers=2,
                                   unit_activation=nn.ReLU,
                                   out_activation=nn.Identity,
                                   batch_norm=False,
                                   layer_norm=False,
                                   batch_norm_first_layer=False,
                                   out_layer_initialize_small=True),
            filter_optim_cls=optim.Adam,
            filter_optim_kwargs=dict(lr=1e-4,
                                     weight_decay=0),
            # Preprocess Observation
            k_epsilon=1e24,  # slack variable weight
            k_delta=1.6,  # for confidence interval k_delta * std
            eta=0.99,  # alpha function coeficient: alpha(x) = eta * h(x)
            stop_criteria_eps=5e-4,  # stop criteria for unsafe experience loss all being negative
            max_epoch=10,
            gamma_dh=0.0,  # saftey threshold in loss
            gamma_safe=0.0,
            gamma_unsafe=0.0,
            train_on_jacobian=True,
            use_trained_dyn=False,
            pretrain_max_epoch=1e4,
            safe_loss_weight=1.0,
            unsafe_loss_weight=1.0,
            ss_safe_loss_weight=1.0,
            ss_unsafe_loss_weight=1.0,
            deriv_loss_weight=1.0,
            deriv_loss_version=2,
            # set this to None, if you don't want to preprocess observation for dynamics training
        )
        return self.cbf_params

    ##############################
    ###### Controllers params
    ##############################
    # Random Shooter
    def get_random_shooter_params(self):
        return AttrDict(
            horizon=10,
            num_particles=1000,
            gamma=0.9,
        )

    # Random Controller
    def get_random_controller_params(self):
        return AttrDict(
            horizon=10
        )

    # CEM
    def get_cem_params(self):
        return AttrDict(
            horizon=25,
            cem_itr=5,
            elites_fraction=0.1,
            gamma=0.9,
            num_particles=400,
            lr=0.1,
            var_threshold=0.001
        )

    def get_controller_params(self, controller_cls):
        controller_params = []
        for i in range(len(controller_cls)):
            if controller_cls[i] == RandomController:
                controller_params.append(self.get_random_controller_params())
            elif controller_cls[i] == RandomShootController:
                controller_params.append(self.get_random_shooter_params())
            elif controller_cls[i] == CEMController:
                controller_params.append(self.get_cem_params())
        return controller_params







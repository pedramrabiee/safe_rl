import torch.nn as nn
import torch.optim as optim
from controller.random_controller import RandomController
from controller.random_shoot_controller import RandomShootController
from controller.cem_controller import CEMController
from attrdict import AttrDict
from utils.misc import deep_update
import importlib
from torch import linspace

class Config:
    def __init__(self, config_override_dict=None):
        self.config_override_dict = config_override_dict
        self.setup = None       # populated in base_trainer __init__
        self.env_spec_config = None     # populated in base_trainer __init__: modify this in the env_config dictionary at the beginning of environment specific env_config file

        # ENVIRONMENT
        self.ac_lim = (-1.0, 1.0)
        self.use_custom_env = True              # implement your customized(/modified) env in utils/make_env

        # UNCOMMENT line below for custom max_episode_len
        self.max_episode_time = 10.0
        self.max_episode_time_eval = 10.0

        # MODE
        self.resume = False
        self.benchmark = False
        self.evaluation_mode = False
        self.debugging_mode = True             # Turns wandb logging off/ saves nothing to files
        self.plot_custom_figs = False
        self.save_custom_figs_data = False

        assert (self.resume + self.benchmark + self.evaluation_mode) < 2, 'Only one of resume, benchmark, or evaluation mode can be True'

        # TRAINER
        # To change random seed number refer to seed_ in utils.seed
        self.episode_steps_per_itr = 1      # number of timesteps to collect data between model updates
        self.n_training_episode = 50
        self.buffer_size = 1e6

        # TRAINING
        self.n_training_processes = 1
        self.training_device = 'cpu'

        # SAMPLER
        self.sampling_batch_size = 'all'                    # you need to remove this and move the sampling batch_size into method parameters. Like in the case of sf agent and trainer
        self.n_episode_initial_data_collection = 1
        self.n_episode_init_phase = 0                       # number of episodes to use init phase for action
        # number of environment steps to skip when saving to buffer
        # (e.g. env_step = 0.001, step_save_freq = 10 -> save to buffer every 0.01s)
        self.step_save_freq = 1

        self.sampler_device = 'cpu'
        self.episodes_per_return_log = 1
        self.add_noise_when_buffering = AttrDict(is_true=False,
                                                 data=['obs', 'next_obs'],
                                                 noise_to_signal=0.01)

        # EVALUATION
        self.do_evaluation = True
        self.n_episodes_evaluation = 5
        self.num_evaluation_sessions = 4    # number of evaluation sessions
        self.n_video_save_per_evaluation = 3
        self.n_evaluation_processes = 1
        self.evaluation_device = 'cpu'
        self.n_serial_envs_evaluation = 8  # for use in DummyVecEnv, only used when n_sampler_processes = 1
        self.save_video = True and not self.debugging_mode  # do not need to save video on debugging mode

        # LOG AND SAVE
        self.results_dir = 'results'
        self.use_wandb = True  # Enable wandb
        self.wandb_project_name = "point"  # wandb project name
        self.save_models = False and not self.debugging_mode    # Enable to save models every SAVE_FREQUENCY episodes (do not need to save on debugging mode)
        self.save_buffer = True
        self.num_save_sessions = 5
        self.add_timestamp = True  # Add timestamp to console's

        # LOAD MODELS
        self.load_models = False
        self.load_buffer = True
        self.load_run_name = 'run-20210914_145325-3nrejh0p'
        self.load_run_id = self.load_run_name[-8:]
        self.overwrite_config = False            # overwrite config file, with the loaded model env_config file
        self.load_timestamp = '20210914_145340_ep0'
        # self.load_timestamp = 'last'

        # Custom model loader:
        # Make a list of dict keys that you want to load from agent's models.
        # To load all models, set this to None
        self.custom_load_list = None
        # self.custom_load_list = ['mf_agent']

        # update attribute values using config override dict
        self.update_attr(self.config_override_dict.get('init', {}))


    ##############################
    ###### Agent params
    ##############################

    def get_agent_params(self, key):
        get_params_func_name = f"_get_{key}_params"
        params_attr_name = f"{key}_params"

        if hasattr(self, get_params_func_name):
            get_params_func = getattr(self, get_params_func_name)
            params = deep_update(get_params_func(), self.config_override_dict.get(params_attr_name))
            setattr(self, params_attr_name, params)
            return params
        else:
            # Handle cases where the key is not found
            raise ValueError(f"Unsupported key: {key}")


    # Model-based
    def _get_mb_params(self):
        from explorations.rand_noise import RandNoise
        from buffers.replay_buffer import ReplayBuffer
        from dynamics.affine_in_action import AffineInActionGaussian
        from dynamics.affine_in_action_gaussian_dynamics import AffineInActionGaussianDynamics

        controller_cls = [RandomController, CEMController]
        replay_buffer_cls = ReplayBuffer
        mb_params = AttrDict(
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
        return mb_params

    # DDPG
    def _get_ddpg_params(self):
        from explorations.ou_noise import OUNoise
        ddpg_params = AttrDict(
            tau=0.001,
            gamma=0.99,
            exp_strategy_cls=OUNoise,
            n_exploration_episode=self.n_training_episode,
            init_phase_coef=0.5,
            exp_kwargs=dict(theta=0.15,         # exploration parameters
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
        return ddpg_params

    def _get_maddpg_params(self):
        ddpg_params = self._get_ddpg_params()
        madddpg_specific_params = AttrDict()
        self.maddpg_params = AttrDict(**ddpg_params, **madddpg_specific_params)
        return self.maddpg_params

    # Safety Filter
    def _get_sf_params(self):
        sf_params = AttrDict(
            # models
            mf='ddpg',
            mb='mb',
            filter='cbf',
            # batch sizes
            mf_train_batch_size=128,
            mb_train_batch_size='all',
            filter_train_batch_size=1024,
            filter_pretrain_sample_size=500,
            # update frequencies
            mf_update_freq=1,
            mb_update_freq=10000,
            filter_update_freq=200,     # this option is not currently used in the sf_trainer, use filter_training_stages instead
            filter_training_stages=dict(stages=[5000, 10000, 15000],
                                        freq=[5000, 4000, 4000]),
            ep_to_start_appending_cbf_deriv_loss_data=4,         # you shouldn't contaminate the data with the data collected under untrained RL policy during first episodes
            # misc.
            safety_filter_is_on=True,
            filter_pretrain_is_on=True,
            filter_train_is_on=True,
            dyn_train_is_on=False,
            mf_train_is_on=True,
            add_cbf_pretrain_data_to_buffer=True,
            save_filter_and_buffer_after_pretraining=True,
        )
        return sf_params

    def _get_bus_params(self):
        bus_params = AttrDict(
            to_shield=True,
        )
        return bus_params

    def _get_rlbus_params(self):
        rlbus_params = AttrDict(
            # rl backup policy
            rl_backup_pretrain_is_on=True,
            rl_backup_pretrain_sample_size=500,
            rl_backup_train_is_on=True,
            rl_backup_update_freq=1,
            rl_backup_train_batch_size=128,
            to_shield=True,
            # desired policy
            use_mf_desired_policy=True,
            desired_policy_agent='ddpg',
            desired_policy_train_in_on=True,
            desired_policy_update_freq=1,
            desired_policy_train_batch_size=128,
        )
        return rlbus_params
    # ========================================
    #              Shields
    # ========================================
    # Backup Shield
    def _get_backup_shield_params(self):
        bus_params = AttrDict(
            eps_buffer=0.0,
            softmin_gain=100,
            softmax_gain=300,
            h_scale=0.05,
            feas_scale=0.05,
            alpha=1.0,
            horizon=1.5,
            backup_timestep=0.1
        )
        num_backup_steps = int(bus_params.horizon/bus_params.backup_timestep) + 1
        backup_t_seq = linspace(0,
                                bus_params.horizon,
                                num_backup_steps)
        bus_params.backup_t_seq = backup_t_seq
        return bus_params


    def _get_rl_backup_shield_params(self):
        rlbus_params = AttrDict(
            eps_buffer=0.0,
            softmin_gain=100,
            softmax_gain=100,
            h_scale=0.05,
            feas_scale=0.05,
            alpha=1.0,
            horizon=2.5,
            melt_law_gain=1000,
            pretraining_melt_law_gain=0.05,
            rl_backup_backup_set_softmax_gain=100,
            backup_timestep=0.1,
            rl_backup_agent='ddpg',
            rl_backup_train=True,
            saturate_rl_backup_reward=True,
            saturate_rl_backup_at=0.0,
            discount_rl_backup_reward=False,
            discount_ratio_rl_backup_reward=0.98,
            rl_backup_pretrain_epoch=50,
            rl_backup_pretrain_batch_size=64,
        )
        num_backup_steps = int(rlbus_params.horizon / rlbus_params.backup_timestep) + 1
        backup_t_seq = linspace(0,
                                rlbus_params.horizon,
                                num_backup_steps)
        rlbus_params.backup_t_seq = backup_t_seq
        return rlbus_params

    # CBF Shield
    def _get_cbf_params(self):
        cbf_params = AttrDict(
            # filter network
            filter_net_kwargs=dict(hidden_dim=512,
                                   num_layers=2,
                                   unit_activation=nn.Tanh,
                                   out_activation=nn.Tanh,
                                   batch_norm=False,
                                   layer_norm=False,
                                   batch_norm_first_layer=False,
                                   out_layer_initialize_small=False),
            filter_optim_cls=optim.Adam,
            filter_optim_kwargs=dict(lr=1e-4,
                                     weight_decay=0),
            # Preprocess Observation
            k_epsilon=1e24,  # slack variable weight
            k_delta=1.6,  # for confidence interval k_delta * std
            eta=0.99,  # alpha function coeficient: alpha(x) = eta * h(x)
            diff_exe_eta=False, # use different eta for execution
            tau=0.1,  # used for polyak update
            stop_criteria_eps=5e-5,  # stop criteria for unsafe experience loss all being negative
            max_epoch=10,
            gamma_dh=0.0,  # saftey threshold in loss
            gamma_safe=0.0,
            gamma_unsafe=0.0,
            train_on_jacobian=True,
            use_trained_dyn=False,
            pretrain_max_epoch=1e5,
            pretrain_batch_to_sample_ratio=0.2,
            # losses weights
            safe_loss_weight=1.0,
            unsafe_loss_weight=1.0,
            deriv_loss_weight=0.1,
            safe_deriv_loss_weight=1.0,
            u_max_weight_in_deriv_loss=1.0,
            deriv_loss_version=2,
            loss_tanh_normalization=False,
            pretrain_holdout_ratio=0.1,
            use_filter_just_for_plot=False,
            train_on_max=True,
            constrain_control=True,

            # set this to None, if you don't want to preprocess observation for dynamics training
        )
        return cbf_params

    # ==============================
    #     Controllers params
    # ==============================
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

    def update_attr(self, overrides):
        for k, v in overrides.items():
            assert hasattr(self, k), f'Invalid key {k}'
            setattr(self, k, v)


def get_config_override(train_env):
    env_collection = train_env['env_collection']
    nickname = train_env['env_nickname']

    # Construct the module and class names
    module_name = f'envs_utils.{env_collection}.{nickname}.{nickname}_configs'

    try:
        config_module = importlib.import_module(module_name)
        config_dict = getattr(config_module, 'config')
        return config_dict
    except ImportError:
        # Handle cases where the module or class is not found
        return NotImplementedError
    except AttributeError:
        # Handle cases where the class is not found in the module
        return NotImplementedError


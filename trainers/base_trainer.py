from config import Config
from config import get_config_override
from utils.seed import set_env_seed, get_seed
from utils.make_env import make_env
from agents.agent_factory import AgentFactory
from utils.misc import *
from samplers.sampler import Sampler
from logger import logger
import torch
from utils import scale
from utils.safe_set import get_safe_set
from utils.process_observation import NeutralObsProc
from copy import copy
from envs_utils.get_env_spec_config import get_env_spec_config


class BaseTrainer:
    def __init__(self, setup, root_dir):
        """
        Initialize the base trainer with the given setup and root directory.

        Args:
            setup (AttrDict): The setup configuration.
            root_dir (str): The root directory for logging and saving data.
        """

        # instantiate config, override config, add setup to config, and load env specific config
        self.load_config(setup)

        # initialize Logger
        logger.initialize(self.config, root_dir)

        # instantiate training environment, set train_iter
        self.make_train_env(setup)

        # instantiate observation processor
        self.setup_obs_proc(setup)

        # instantiate custom_plotter
        self.setup_custom_plotter(setup)

        # instantiate and initialize agent
        self.agent = self.initialize_agent(setup)

        # initialize scale
        scale.initialize(self.env.action_bounds)

        # instantiate evaluation environment
        self.make_eval_env(setup)

        # make safe_sets
        self.make_safe_sets(setup)

        # instantiate and initialize sampler
        self.sampler = Sampler(self.config)
        self.sampler.initialize(env=self.env,
                                env_eval=self.env_eval,
                                agent=self.agent,
                                obs_proc=self.obs_proc,
                                custom_plotter=self.custom_plotter,
                                safe_set=self.safe_set,
                                safe_set_eval=self.safe_set_eval)

        # load params if model loading enables
        self.load_model_params()

        # save config, config override, env specific config, Mujoco xml file, Engine config
        self.save_configs()

        # run custom trainer initialization
        self.initialize()

        logger.log("Trainer initialized...")

    def load_config(self, setup):
        """
            Instantiate configurations based on the given setup.

            Args:
                setup (AttrDict): The setup configuration.
        """
        # instantiate Config
        self.config_override = get_config_override(setup['train_env'])

        if setup.load_config_path is None:
            self.config = Config(self.config_override)
        else:
            self.config = load_config_from_py(setup.load_config_path)

        if (self.config.load_models and self.config.overwrite_config) or self.config.resume:
            new_config = get_loaded_config(project_dir=osp.join(self.config.results_dir,
                                                                self.config.wandb_project_name),
                                           load_run_name=self.config.load_run_name)
            self._overwrite_config(new_config=new_config)

        self.config.setup = setup
        self.config.env_spec_config = get_env_spec_config(setup['train_env'])

    def make_train_env(self, setup):
        """
           Create and initialize the training environment based on the given setup.

           Args:
               setup (AttrDict): The setup configuration.

       """
        self.env, env_info = make_env(env_id=setup['train_env']['env_id'],
                                      env_nickname=setup['train_env']['env_nickname'],
                                      collection=setup['train_env']['env_collection'],
                                      ac_lim=self.config.ac_lim,
                                      max_episode_time=self.config.max_episode_time,
                                      use_custom_env=self.config.use_custom_env)
        # self.env.seed(seed)     # set environment seed
        set_env_seed(self.env, get_seed())

        # Get max episode length
        self.config.max_episode_len = env_info['max_episode_len']
        self.train_iter = int(
            self.config.n_training_episode * env_info['max_episode_len'] / self.config.episode_steps_per_itr / self.config.step_save_freq)

        if self.config.setup.train_env['env_collection'] == 'safety_gym':
            # reset environment to create layout (to get obstacle positions)
            _ = self.env.reset()
            # store obstacle position to use in evaluation environment
            if self.config.env_spec_config.use_same_layout_for_eval:
                from utils.safety_gym_utils import make_obstacles_location_dict
                self.obstacle_locations = make_obstacles_location_dict(self.env)

    def setup_obs_proc(self, setup):
        """
        Set up the observation processor based on the given setup.

        Args:
            setup (AttrDict): The setup configuration.

        """
        self.obs_proc = setup['obs_proc_cls'](self.env) if self.config.env_spec_config.do_obs_proc else NeutralObsProc(self.env)
        self.obs_proc.initialize()
        self.config.setup['obs_proc'] = self.obs_proc

    def setup_custom_plotter(self, setup):
        """
         Set up a custom plotter based on the given setup.

         Args:
             setup (AttrDict): The setup configuration.

         """
        # instantiate CustomPlotter
        self.custom_plotter = setup['custom_plotter_cls'](self.obs_proc)
        self.config.setup['custom_plotter'] = self.custom_plotter

    def initialize_agent(self, setup):
        """
        Create and initialize an agent based on the given setup.

        Args:
            setup (AttrDict): The setup configuration.

        Returns:
            object: The initialized agent.
        """
        agent_factory = AgentFactory(self.env, self.config)
        return agent_factory(setup['agent'])

    def make_eval_env(self, setup):
        """
        Create and initialize the evaluation environment based on the given setup.

        Args:
            setup (AttrDict): The setup configuration.

        """
        # instantiate evaluation environment
        if self.config.save_video:
            video_dict = {}
            video_save_dir = osp.join(logger.logdir, 'videos')
            # save n last videos per evaluation
            video_callable = n_last_eval_video_callable(n=self.config.n_video_save_per_evaluation,
                                                        value=int(self.config.n_episodes_evaluation))

            video_dict['video_save_dir'] = video_save_dir
            video_dict['video_callable'] = video_callable
        else:
            video_dict = None

        self.env_eval, env_eval_info = make_env(env_id=setup['eval_env']['env_id'],
                                                env_nickname=setup['eval_env']['env_nickname'],
                                                collection=setup['eval_env']['env_collection'],
                                                video_dict=video_dict,
                                                ac_lim=self.config.ac_lim,
                                                max_episode_time=self.config.max_episode_time_eval,
                                                use_custom_env=self.config.use_custom_env,
                                                make_env_dict=getattr(self, 'obstacle_locations', None))

        self.eval_seed = get_seed() + 123123
        set_env_seed(self.env_eval, self.eval_seed)
        # Get max episode length for evaluation
        self.config.max_episode_len_eval = env_eval_info['max_episode_len']

        if self.config.setup.train_env['env_collection'] == 'safety_gym':
            _ = self.env_eval.reset()

    def make_safe_sets(self, setup):
        """
        Create safe sets for the training and evaluation environments.

        Args:
            setup (AttrDict): The setup configuration.

        """
        # instantiate safe_set
        self.safe_set = get_safe_set(env_info=setup.train_env,
                                     env=self.env,
                                     obs_proc=self.obs_proc,
                                     seed=get_seed())

        # instantiate safe_set for evaluation enviornment
        self.safe_set_eval = get_safe_set(env_info=setup.eval_env,
                                          env=self.env_eval,
                                          obs_proc=self.obs_proc,
                                          seed=self.eval_seed)

    def train(self):
        """
        Start the training process.
        """
        logger.log("Training started...")
        # run training loop
        for itr in range(self.train_iter):
            logger.log("Training at iteration %d" % itr)

            # train
            self._train(itr)

            # save checkpoint based on num_save_session
            if self.config.save_models:
                self._save_checkpoint(itr)

            # evaluate based on num_evaluation_session
            if self.config.do_evaluation:
                self._evaluate(itr)

    def load_model_params(self):
        """
         Load model parameters if the model loading is enabled.
        """
        if self.config.load_models or self.config.resume or self.config.benchmark or self.config.evaluation_mode:
            if self.config.evaluation_mode:
                logger.log("Loading models and optimizers for evaluation...", color='cyan')
            else:
                logger.log("Loading models and optimizers...")

            self._load()

    def save_configs(self):
        """
        Save various configuration settings, environment specifics, and related files.
        """
        # save config as a py
        if not self.config.debugging_mode:
            save_config_as_py(logger.logdir)
            if self.config.setup.train_env['env_collection'] == 'safety_gym':
                # save mujoco xml file
                from utils.safety_gym_utils import save_mujoco_xml_file
                save_mujoco_xml_file(xml_path=self.config.env_spec_config['robot_base'],
                                     save_dir=logger.logdir)

        # save env config as json
        logger.dump_dict2json(self.config.env_spec_config, 'env_spec_config')
        logger.dump_dict2json(self.config_override, 'config_override')

    def initialize(self):
        """
        Initialize the custom trainer. This method can be overridden in derived classes.
        """
        pass

    def evaluate(self):
        """
        Start the evaluation process.
        """
        logger.log("Evaluation started...")
        # run evaluation loop
        self._evaluate(itr=0)

    def _train(self, itr):
        """
        Perform the training loop for a single iteration.

        Args:
            itr (int): The current training iteration.
        """
        raise NotImplementedError

    def _prep_optimizer_dict(self):
        """
        Prepare an optimizer dictionary for use. This method can be overridden in derived classes.

        Returns:
            dict: The optimizer dictionary.
        """
        return dict()

    def _save_checkpoint(self, itr):
        """
        Save a checkpoint at the specified iteration.

        Args:
            itr (int): The current iteration.
        """
        if itr % int(self.train_iter / self.config.num_save_sessions) == 0 or itr >= self.train_iter:
            logger.log('Saving checkpoint at episode %d...' % self.sampler.episode_completed)
            self._save(itr=itr, episode=self.sampler.episode_completed)

    def _evaluate(self, itr):
        """
        Perform the evaluation process.

        Args:
            itr (int): The current iteration.
        """
        if self.config.evaluation_mode or itr % int(self.train_iter / self.config.num_evaluation_sessions) == 0 or itr >= self.train_iter:
            # set the plotter in evaluation mode: push_plot calls won't work
            logger.set_plotter_in_eval_mode()
            self.sampler.evaluate()
            logger.set_plotter_in_train_mode()
            logger.dump_tabular(cat_key='evaluation_episode', wandb_log=True, csv_log=False)

    def _save(self, itr, episode):
        """
        Save a checkpoint with the specified iteration and episode.

        Args:
            itr (int): The current iteration.
            episode (int): The current episode.
        """
        filename = get_save_checkpoint_name(logger.logdir, episode)
        states = {}
        states['itr'] = itr
        states['episdoe'] = episode
        states = self.agent.get_params()
        if self.config.save_buffer:
            num_buffer = self.agent.num_buffer
            if num_buffer == 1:
                states['buffer'] = self.agent.get_buffer()
            else:
                cur_id = self.agent.curr_buf_id
                states['buffer'] = []
                for i in range(num_buffer):
                    self.agent.curr_buf_id = i
                    states['buffer'].append(self.agent.get_buffer())
                self.agent.curr_buf_id = cur_id
        torch.save(states, filename)
        logger.log('Checkpoint saved: %s' % filename)

    def _load(self):
        """
        Load model parameters and optionally buffers if loading is enabled.
        """
        path = get_load_checkpoint_name(current_root=logger.logdir,
                                        load_run_name=self.config.load_run_name,
                                        timestamp=self.config.load_timestamp)
        checkpoint = torch.load(path)
        self.agent.load_params(checkpoint=checkpoint,
                               custom_load_list=self.config.custom_load_list)
        if self.config.load_buffer and 'buffer' in checkpoint.keys():
            if isinstance(checkpoint['buffer'], list):
                cur_id = self.agent.curr_buf_id
                for i, buffer_data in enumerate(checkpoint['buffer']):
                    self.agent.curr_buf_id = i
                    self.agent.init_buffer(buffer_data)
                self.agent.curr_buf_id = cur_id
            else:
                self.agent.init_buffer(checkpoint['buffer'])

    def _overwrite_config(self, new_config):
        """
        Overwrite the current configuration with a new configuration.

        Args:
            new_config (Config): The new configuration.
        """
        # overwrite loaded config from the loaded run with the current config.
        # However, revert the changes for loading setting to continue with loading model in _load method
        load_models = copy(self.config.load_models)
        load_run_name = copy(self.config.load_run_name)
        load_run_id = copy(self.config.load_run_id)

        self.config = new_config
        self.config.load_models = load_models
        self.config.load_run_name = load_run_name
        self.config.load_run_id = load_run_id

    def _obs_proc_from_samples_by_key(self, samples, proc_key):
        """
        Process observations in the samples by a specified processing key.

        Args:
            samples (dict): A dictionary of samples.
            proc_key (str): The key for observation processing.

        Returns:
            dict: The samples with processed observations.
        """
        samples['obs'] = self.obs_proc.proc(samples['obs'], proc_key=proc_key)
        samples['next_obs'] = self.obs_proc.proc(samples['next_obs'], proc_key=proc_key)
        return samples
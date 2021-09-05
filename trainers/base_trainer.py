from config import Config
from utils.seed import set_seed, set_env_seed
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
from configs.get_env_spec_config import get_env_spec_config
from safety_gym.envs.engine import Engine

class BaseTrainer:
    def __init__(self, setup, root_dir):
        # instantiate Config
        if setup.load_config_path is None:
            self.config = Config()
        else:
            self.config = load_config_from_py(setup.load_config_path)

        if (self.config.load_models and self.config.overwrite_config) or self.config.resume:
            new_config = get_loaded_config(project_dir=osp.join(self.config.results_dir,
                                                                self.config.wandb_project_name),
                                           load_run_name=self.config.load_run_name)
            self._overwrite_config(new_config=new_config)

        self.config.setup = setup
        self.config.env_spec_config = get_env_spec_config(setup['train_env'])

        # Set random.seed, np.random.seed, torch.manual_seed, torch.cuda.manual_seed_all
        seed = set_seed(self.config.seed)

        # instantiate training environment
        self.env, env_info = make_env(env_id=setup['train_env']['env_id'],
                                      collection=setup['train_env']['env_collection'],
                                      ac_lim=self.config.ac_lim,
                                      max_episode_time=self.config.max_episode_time,
                                      use_custom_env=self.config.use_custom_env)
        # self.env.seed(seed)     # set environment seed
        set_env_seed(self.env, seed)

        # Get max episode length
        self.config.max_episode_len = env_info['max_episode_len']
        self.train_iter = int(self.config.n_training_episode * env_info['max_episode_len'] / self.config.episode_steps_per_itr)

        if self.config.setup.train_env['env_collection'] == 'safety_gym':
            _ = self.env.reset()
            if self.config.env_spec_config.use_same_layout_for_eval:
                from utils.safety_gym_utils import make_obstacles_location_dict
                self.obstacle_locations = make_obstacles_location_dict(self.env)

        # instantiate observation processor
        # TODO: add boolean for this in config OR NOT (I can have the switch inside the agent themselves)
        self.obs_proc = self.config.setup['obs_proc_cls'](self.env) if self.config.env_spec_config.do_obs_proc else NeutralObsProc(self.env)
        self.obs_proc.initialize()
        self.config.setup['obs_proc'] = self.obs_proc


        # instantiate CustomPlotter
        self.custom_plotter = self.config.setup['custom_plotter_cls'](self.obs_proc)
        self.config.setup['custom_plotter'] = self.custom_plotter

        # instantiate safe_set
        self.safe_set = get_safe_set(env_id=setup.train_env.env_id,
                                     env=self.env,
                                     obs_proc=self.obs_proc,
                                     seed=seed)

        # instantiate and initialize agent
        self.agent = self._initialize_agent(setup)

        # initialize Logger
        logger.initialize(self.config, root_dir)

        # initialize scale
        scale.initialize(self.env.action_bounds)

        # save config as a pickle file
        if not self.config.debugging_mode:
            save_config_as_py(logger.logdir)

        # save env config as json
        logger.dump_dict2json(self.config.env_spec_config, 'env_spec_config')

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
                                                collection=setup['eval_env']['env_collection'],
                                                video_dict=video_dict,
                                                ac_lim=self.config.ac_lim,
                                                max_episode_time=self.config.max_episode_time_eval,
                                                use_custom_env=self.config.use_custom_env,
                                                make_env_dict=getattr(self, 'obstacle_locations', None))

        eval_seed = seed + 123123
        set_env_seed(self.env_eval, eval_seed)
        # Get max episode length for evaluation
        self.config.max_episode_len_eval = env_eval_info['max_episode_len']

        if self.config.setup.train_env['env_collection'] == 'safety_gym':
            _ = self.env_eval.reset()

        # instantiate safe_set for evaluation enviornment
        self.safe_set_eval = get_safe_set(env_id=setup.eval_env.env_id,
                                          env=self.env_eval,
                                          obs_proc=self.obs_proc,
                                          seed=eval_seed)

        # instantiate Sampler
        self.sampler = Sampler(self.config)
        # initialize sampler
        self.sampler.initialize(env=self.env,
                                env_eval=self.env_eval,
                                agent=self.agent,
                                obs_proc=self.obs_proc,
                                custom_plotter=self.custom_plotter,
                                safe_set=self.safe_set,
                                safe_set_eval=self.safe_set_eval)

        # load params if model loading enables
        if self.config.load_models or self.config.resume or self.config.benchmark or self.config.evaluation_mode:
            if self.config.evaluation_mode:
                logger.log("Loading models and optimizers for evaluation...", color='cyan')
            else:
                logger.log("Loading models and optimizers...")

            self._load()

        # run custom trainer initialization
        self.initialize()


        logger.log("Trainer initialized...")

    def initialize(self):
        pass

    def train(self):
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

    def evaluate(self):
        logger.log("Evaluation started...")
        # run evaluation loop
        self._evaluate(itr=0)


    def _train(self, itr):
        raise NotImplementedError

    def _prep_optimizer_dict(self):
        return dict()

    # helper functions
    def _initialize_agent(self, setup):
        agent_factory = AgentFactory(self.env, self.config)
        return agent_factory(setup['agent'])

    def _save_checkpoint(self, itr):
        if itr % int(self.train_iter / self.config.num_save_sessions) == 0 or itr >= self.train_iter:
            logger.log('Saving checkpoint at episode %d...' % self.sampler.episode_completed)
            self._save(itr=itr, episode=self.sampler.episode_completed)

    def _evaluate(self, itr):
        if self.config.evaluation_mode or itr % int(self.train_iter / self.config.num_evaluation_sessions) == 0 or itr >= self.train_iter:
            # set the plotter in evaluation mode: push_plot calls won't work
            logger.set_plotter_in_eval_mode()
            self.sampler.evaluate()
            logger.set_plotter_in_train_mode()
            logger.dump_tabular(cat_key='evaluation_episode', wandb_log=True, csv_log=False)

    def _save(self, itr, episode):
        filename = get_save_checkpoint_name(logger.logdir, episode)
        states = {}
        states['itr'] = itr
        states['episdoe'] = episode
        states = self.agent.get_params()
        if self.config.save_buffer:
            states['buffer'] = self.agent.get_buffer()
        torch.save(states, filename)
        logger.log('Checkpoint saved: %s' % filename)

    def _load(self):
        path = get_load_checkpoint_name(current_root=logger.logdir,
                                        load_run_name=self.config.load_run_name,
                                        timestamp=self.config.load_timestamp)
        checkpoint = torch.load(path)
        self.agent.load_params(checkpoint=checkpoint,
                               custom_load_list=self.config.custom_load_list)
        if self.config.load_buffer and 'buffer' in checkpoint.keys():
            self.agent.init_buffer(checkpoint['buffer'])

    def _overwrite_config(self, new_config):
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
        samples['obs'] = self.obs_proc.proc(samples['obs'], proc_key=proc_key)
        samples['next_obs'] = self.obs_proc.proc(samples['next_obs'], proc_key=proc_key)
        return samples
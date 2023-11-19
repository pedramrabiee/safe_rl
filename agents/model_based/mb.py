from agents.base_agent import BaseAgent
from torch.utils.data import TensorDataset, DataLoader
from time import time
from tqdm import tqdm
from utils.misc import *
from utils.normalizer import *
from logger import logger


class MBAgent(BaseAgent):
    def initialize(self, params, init_dict=None):
        self.params = params
        self.ac_bounds = init_dict['bounds']

        # get the observation dim from observation process class
        self._obs_dim = self.obs_proc.obs_dim(proc_key='mb')

        # instantiate and initialize dynamics
        if self.params.is_ensemble:
            from dynamics.ensemble_nn_dynamics import EnsembleDynamics
            self.dynamics = EnsembleDynamics(obs_dim=self._obs_dim,
                                             ac_dim=self._ac_dim,
                                             out_dim=self._obs_dim,
                                             timestep=self._timestep,
                                             obs_proc=self.obs_proc)
            self.dynamics.initialize(params=self.params.dynamics_params,
                                     init_dict=AttrDict(dynamics_cls=self.params.dynamics_cls,
                                                        env_bounds=self.ac_bounds))
        else:
            self.dynamics = self.params.dynamics_cls(obs_dim=self._obs_dim,
                                                     ac_dim=self._ac_dim,
                                                     out_dim=self._obs_dim,
                                                     timestep=self._timestep,
                                                     obs_proc=self.obs_proc)
            self.dynamics.initialize(params=self.params.dynamics_params,
                                     init_dict=AttrDict(env_bounds=self.ac_bounds))

        # instantiate and initialize planner
        self.reward_gen = init_dict['reward_gen']
        self.init_controller(idx=0)

        # instantiate action exploration
        self.exploration = self.params.exp_strategy_cls(ac_dim=self._ac_dim, **self.params.exp_kwargs)

        # initialize normalizer
        self._stats = init_stats(self._obs_dim, self._ac_dim)

        self.normalized_io = self.params.dynamics_params.normalized_io
        self.delta_output = self.params.dynamics_params.delta_output # FIXME: remove delta_output if you're not using it

        # list models
        if hasattr(self.dynamics, 'ensemble'):
            self.models = [dyn.model for dyn in self.dynamics.ensemble]
            self.optimizers = [dyn.optimizer for dyn in self.dynamics.ensemble]
            self.models_dict = {f'dyn_model_{k}': dyn.model for k, dyn in enumerate(self.dynamics.ensemble)}
            self.optimizers_dict = {f'dyn_optim_{k}': dyn.optimizer for k, dyn in enumerate(self.dynamics.ensemble)}
            if hasattr(self.dynamics, 'extra_params'):
                self.extra_params = list(np.hstack([dyn.extra_params for dyn in self.dynamics.ensemble]))
                self.extra_params_dict = {f'dyn_extra_param_{k}': dyn.extra_params_dict for k, dyn in enumerate(self.dynamics.ensemble)}
        else:
            self.models = [self.dynamics.model]
            self.optimizers = [self.dynamics.optimizer]
            self.models_dict = dict(dyn_model=self.dynamics.model)
            self.optimizers_dict = dict(dyn_optim=self.dynamics.optimizer)
            if hasattr(self.dynamics, 'extra_params'):
                self.extra_params = self.dynamics.extra_params
                self.extra_params_dict = self.dynamics.extra_params_dict

    def reset(self):
        self.controller.reset()

    def init_controller(self, idx):
        self.controller = self.params.controller_cls[idx](dynamics=self.dynamics,
                                                          reward_gen=self.reward_gen,
                                                          bounds=self.ac_bounds,
                                                          ac_dim=self._ac_dim,
                                                          obs_proc=self.obs_proc)

        self.controller.initialize(self.params, init_dict=self.params.controller_params[idx])

    def step(self, obs, explore=False, init_phase=False):
        # process observation to match the models' input requirement
        obs = self.obs_proc.proc(obs, proc_key='mb')

        action = self.controller.act(obs, self._stats)
        if torch.is_tensor(action):
            if explore:
                action += torchify(self.exploration.noise())
            action = action.clamp(self._ac_lim['low'], self._ac_lim['high'])
        if isinstance(action, np.ndarray):
            if explore:
                action += self.exploration.noise()
            action = action.clip(self._ac_lim['low'], self._ac_lim['high'])
        return action, None

    def optimize_agent(self, samples, optim_dict=None):
        self._stats = self.buffer.get_stats(obs_preproc_func=self.obs_proc.proc if self.obs_proc else None)
        loss = self.dynamics.train(samples, train_dict=AttrDict(stats=self._stats,
                                                                itr=optim_dict['itr']))
        self.eval_mode()
        eval_loss = self.dynamics.eval()

        logger.add_tabular({'EvalLoss/Dynamics': eval_loss}, cat_key='iteration')

        return {"Loss": loss, "Eval_Loss": eval_loss}


    @property
    def stats(self):
        return self._stats




from agents.base_agent import BaseAgent
from utils.misc import *
from logger import logger


class SFAgent(BaseAgent):
    def initialize(self, params, init_dict=None):
        self.params = params

        # model-free and model-based agents are instantiated and initialized in agent factory
        self.mf_agent = init_dict.mf_agent
        self.mb_agent = init_dict.mb_agent
        self.safety_filter = init_dict.safety_filter

        # set dynamics predictor for safety_filter
        self.safety_filter.set_dyn_predictor(predictor=self.mb_agent.dynamics.predict)

        # list models and optimizers
        self.models = [*self.mf_agent.models, *self.mb_agent.models, *self.safety_filter.models]
        self.optimizers = [*self.mf_agent.optimizers, *self.mb_agent.optimizers, *self.safety_filter.optimizers]
        self.models_dict = dict(mf_agent=self.mf_agent.models_dict,
                                mb_agent=self.mb_agent.models_dict,
                                safety_filter=self.safety_filter.models_dict)

        self.optimizers_dict = dict(mf_agent=self.mf_agent.optimizers_dict,
                                    mb_agent=self.mb_agent.optimizers_dict,
                                    safety_filter=self.safety_filter.optimizers_dict)

        self.extra_params = self.mb_agent.extra_params
        self.extra_params_dict = dict(mb_agent=self.mb_agent.extra_params_dict)

        self.agents = [self.mf_agent, self.mb_agent, self.safety_filter]


    def step(self, obs, explore=False, init_phase=False):
        # call act method on model-free agent
        ac_mf, _ = self.mf_agent.act(obs, explore=explore, init_phase=init_phase)
        ac_mf = np.expand_dims(ac_mf, axis=0) if np.ndim(ac_mf) == 1 else ac_mf

        # get dynamics from model-based, ground truth and mu, std
        dyn_bd = self.mb_agent.dynamics.predict(obs=obs, ac=ac_mf, stats=self.mb_agent.stats, split_return=True)

        info = None
        # pass action, and dynamics to filter and get the filtered action
        if self.params.safety_filter_is_on:
            ac_filtered, dyn_out = self.safety_filter.filter(obs, ac_mf, filter_dict=dict(dyn_bd=dyn_bd))
            info = dict(dyn_out=dyn_out)    # FIXME: rename dyn_out to filter_info
        else:
            ac_filtered = ac_mf
        return ac_filtered, info

    def optimize_agent(self, samples, optim_dict=None):
        mf_loss = None
        mb_loss = None
        filter_loss = None
        itr = optim_dict['itr']
        to_train = optim_dict['to_train']

        # train model-free on its train frequency
        if 'mf' in to_train:
            logger.log('Training MF agent...')
            mf_loss = self.mf_agent.optimize_agent(samples['mf'], optim_dict)
        # train model-based on its train frequency
        if 'mb' in to_train:
            logger.log('Training MB agent...')
            mb_loss = self.mb_agent.optimize_agent(samples['mb'], optim_dict)
        # train filter on its frequency
        if 'filter' in to_train:
            logger.log('Training Filter...')
            # ac_mf = self.mf_agent.act(experience.obs)
            # dyn_bd = self.mb_agent.dynamics.predict(obs=experience.obs, ac=ac_mf, stats=self.mb_agent.stats, split_return=True)
            filter_loss = self.safety_filter.optimize_agent(samples['filter'], optim_dict)
        return {"MFـLoss": mf_loss, "MB_Loss": mb_loss, "Filter_Loss": filter_loss}

    def pre_train_filter(self, samples, pre_train_dict=None):
        if self.params.safety_filter_is_on:
            logger.log('Pretraining started...')
            self.safety_filter.pre_train(samples, pre_train_dict)
        else:
            pass

    def after_optimize(self):
        for agent in self.agents:
            agent.after_optimize()

    def sample_mode(self, device='cpu', sample_dict=None):
        # Set model-free policy on eval mode
        if hasattr(self.mf_agent, 'policy'):
            self.mf_agent.policy.eval()
            self.mf_agent.policy = to_device(self.mf_agent.policy, device)

        # Set dynamics on eval mode
        self.mb_agent.dynamics.eval_mode(device=device)

        # Set cbf filter on eval mode
        self.safety_filter.filter_net.eval()
        self.safety_filter.filter_net = to_device(self.safety_filter.filter_net, device)

    def on_episode_reset(self, episode):
        for agent in self.agents:
            agent.on_episode_reset(episode)


    def is_safety_criteria_violated(self, obs):
        return self.safety_filter.is_safety_criteria_violated(obs)
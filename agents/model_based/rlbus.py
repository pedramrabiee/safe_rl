from agents.base_agent import BaseAgent
from agents.model_based.bus import BUS

from utils.misc import *
from logger import logger
# RL Backup Shield Agent
class RLBUS(BUS):
    def initialize(self, params, init_dict=None):
        super().initialize(params, init_dict)

        # list models and optimizers
        self.models = [*self.shield.rl_backup.models]
        self.optimizers = [*self.shield.rl_backup.optimizers]
        self.models_dict = dict(shield=self.shield.rl_backup.models_dict)
        self.optimizers_dict = dict(shield=self.shield.rl_backup.optimizers_dict)

        self.agents = [self.shield.rl_backup]

        if self.params.use_mf_desired_policy:
            self.models = [*self.models, *self.desired_policy.models]
            self.optimizers = [*self.optimizers, *self.desired_policy.optimizers]
            self.models_dict['desired_policy'] = self.desired_policy.models_dict
            self.optimizers_dict['desired_policy'] = self.desired_policy.optimizers_dict
            self.agents.append(self.desired_policy)

    def step(self, obs, explore=False, init_phase=False):
        # TODO: CHECK SCALING, CHECK NUMPY
        # obs = self.obs_proc.proc(obs, proc_key='shield').squeeze()
        # obs.squeeze()
        if self.params.use_mf_desired_policy:
            u_des, _ = self.desired_policy.act(obs)
        else:
            u_des = self.desired_policy.act(obs)
        return self.shield.shield(obs, u_des), None
    def optimize_agent(self, samples, optim_dict=None):
        rl_backup_loss = None
        desired_policy_loss = None
        to_train = optim_dict['to_train']

        # train model-free desired policy on its train frequency
        if 'desired_policy' in to_train:
            logger.log('Training Desired Policy...')
            desired_policy_loss = self.desired_policy.optimize_agent(samples['desired_policy'], optim_dict)

        # train rl-backup on its train frequency
        if 'rl_backup' in to_train:
            logger.log('Training RL Backup Policy...')
            rl_backup_loss = self.shield.optimize_agent(samples['rl_backup'], optim_dict)

        return {"Desired_Policy_Loss": desired_policy_loss,
                "RL_Backup_Loss": rl_backup_loss}


    def sample_mode(self, device='cpu', sample_dict=None):
        # Set desired policy on eval mode
        if hasattr(self.desired_policy, 'policy'):
            self.desired_policy.policy.eval()
            self.desired_policy.policy = to_device(self.desired_policy.policy, device)

        # set rl backup on eval mode
        self.shield.rl_backup.policy.eval()
        self.shield.rl_backup.policy = to_device(self.shield.rl_backup.policy, device)

    def after_optimize(self):
        for agent in self.agents:
            agent.after_optimize()


    def on_episode_reset(self, episode):
        for agent in self.agents:
            agent.on_episode_reset(episode)





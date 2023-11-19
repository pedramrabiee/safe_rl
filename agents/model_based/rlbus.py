from agents.model_based.bus import BUS
from utils.misc import *
from logger import logger
from utils.scale import action2newbounds, action2oldbounds


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
        # The 'shield' method expects unnormalized actions (i.e., old action bounds), while the 'step' method is
        # designed to return normalized actions (i.e., new action bounds).
        if self.params.use_mf_desired_policy:
            ac_des, _ = self.desired_policy.act(obs)
            # it is assumed that model free policy outputs the actions in [-1.0, 1.0]
            # (or in the new action bounds (wrapped). Thus, we have to scale it
            ac_des = action2oldbounds(ac_des)
        else:
            # it is assumed that the designed desired policy outputs in the old action bounds (unwrapped)
            ac_des = self.desired_policy.act(obs)

        if self.params.to_shield:
            ac_shield = self.shield.shield(obs, ac_des)
            # scale back the shielded action to new
            ac_shield = action2newbounds(ac_shield)
            return ac_shield, None

        self.custom_plotter.filter_push_action((ac_des, ac_des))

        return action2newbounds(ac_des), None

    def optimize_agent(self, samples, optim_dict=None):
        rl_backup_loss = None
        desired_policy_loss = None
        to_train = optim_dict['to_train']

        # train model-free desired policy on its train frequency
        if 'desired_policy' in to_train:
            logger.log('Training Desired Policy...')
            desired_policy_loss = self.desired_policy.optimize_agent(samples['desired_policy'], optim_dict)
            optim_info_desired = {}
            for k, v in desired_policy_loss.items():
                 optim_info_desired[f'{k}/DesiredPolicy'] = v
            logger.add_tabular(optim_info_desired, cat_key='des_policy_iteration')
            logger.dump_tabular(cat_key='des_policy_iteration', wandb_log=True)

        # train rl-backup on its train frequency
        if 'rl_backup' in to_train:
            logger.log('Training RL Backup Policy...')
            rl_backup_loss = self.shield.optimize_agent(samples['rl_backup'], optim_dict)

            optim_info_rl_backup = {}
            for k, v in rl_backup_loss.items():
                optim_info_rl_backup[f'{k}/RLBackup'] = v
            logger.add_tabular(optim_info_rl_backup, cat_key='rl_backup_iteration')
            logger.dump_tabular(cat_key='rl_backup_iteration', wandb_log=True)


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






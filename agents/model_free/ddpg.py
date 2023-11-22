import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from agents.base_agent import BaseAgent
from distributions.dist_functional import one_hot_from_logits
from networks.mlp import MLPNetwork
from networks.multi_input_mlp import MultiInputMLP
from utils.misc import *
from logger import logger
from utils import scale
from utils.seed import rng


class DDPGAgent(BaseAgent):
    def initialize(self, params, init_dict=None):
        # get the observation dim from observation process class
        self._obs_dim = self.obs_proc.obs_dim(proc_key='mf')

        # instantiate policy and critic
        self.policy = MLPNetwork(in_dim=self._obs_dim, out_dim=self._ac_dim,
                                 **params.pi_net_kwargs)

        if params.multi_in_critic:
            self.critic = MultiInputMLP(in1_dim=self._obs_dim, in2_dim=self._ac_dim, out_dim=1,
                                        **params.q_net_kwargs, **params.multi_in_critic_kwargs)
        else:
            self.critic = MLPNetwork(in_dim=self._obs_dim + self._ac_dim, out_dim=1,
                                     **params.q_net_kwargs)

        # make target nets
        self.policy_target = hard_copy(self.policy)
        self.critic_target = hard_copy(self.critic)

        # instantiate optimizers
        self.policy_optimizer = params.pi_optim_cls(self.policy.parameters(), **params.pi_optim_kwargs)
        self.critic_optimizer = params.q_optim_cls(self.critic.parameters(), **params.q_optim_kwargs)

        # instantiate action exploration
        if not self._discrete_action:
            self.exploration = params.exp_strategy_cls(self._ac_dim, **params.exp_kwargs)

        # list models
        self.models = [self.policy, self.critic, self.policy_target, self.critic_target]
        self.models_dict = dict(policy=self.policy,
                                critic=self.critic,
                                policy_target=self.policy_target,
                                critic_target=self.critic_target)

        self.optimizers = [self.policy_optimizer, self.critic_optimizer]
        self.optimizers_dict = dict(policy_optimizer=self.policy_optimizer,
                                    critic_optimizer=self.critic_optimizer)

        self.params = params

        # initialize last episode noise is reset and rescaled as -1
        self._reset_noise_ep = -1

    def step(self, obs, explore=False, init_phase=False):
        # process observation to match the models' input requirement
        obs = self.obs_proc.proc(obs, proc_key='mf')
        obs = torch.as_tensor(obs, dtype=torch.float64)
        if init_phase:
            return self.init_phase_step(obs, explore), None

        action = self.policy(obs)
        if self._discrete_action:   # discrete action
            if explore:
                action = F.gumbel_softmax(action, hard=True)
            else:
                action = one_hot_from_logits(action)
        else:   # continuous action
            if explore:
                action += torchify(self.exploration.noise())
            action = action.clamp(self._ac_lim['low'], self._ac_lim['high'])
        return action, None

    def init_phase_step(self, obs, explore):
        return torchify(rng.uniform(low=scale.ac_new_bounds[0], high=scale.ac_new_bounds[1], size=self._ac_dim),
                        device=obs.device) * self.params.init_phase_coef

    def sample_mode(self, device='cpu', sample_dict=None):
        super(DDPGAgent, self).sample_mode(device=device)
        if not self._discrete_action:
            episode = sample_dict['episode']

            # set noise scale, and reset noise state at the start of each episode
            self._reset_rescale_noise(episode) # FIXME: this only works when the episode does not finish during the sampling stage

    def optimize_agent(self, samples, optim_dict=None):
        # run one gradient descent step for Q
        self.critic_optimizer.zero_grad()
        loss_critic = self._compute_critic_loss(samples)
        loss_critic.backward()
        # clip grad
        if self.params.use_clip_grad_norm:
            clip_grad_norm_(self.critic.parameters(), self.params.clip_max_norm)
        self.critic_optimizer.step()

        # freeze q-network to save computational effort
        freeze_net(self.critic)

        # run one gradient descent for policy
        self.policy_optimizer.zero_grad()
        loss_policy = self._compute_policy_loss(samples)
        loss_policy.backward()
        # clip grad
        if self.params.use_clip_grad_norm:
            clip_grad_norm_(self.policy.parameters(), self.params.clip_max_norm)

        self.policy_optimizer.step()

        # unfreeze q-network
        unfreeze_net(self.critic)
        optim_info = {"Loss/Policy": loss_policy.cpu().data.numpy(),
                      "Loss/Critic": loss_critic.cpu().data.numpy()}

        # add log to logger
        logger.add_tabular(optim_info, cat_key='iteration')

        return optim_info

    def after_optimize(self):
        # update target networks
        polyak_update(target=self.policy_target,
                      source=self.policy,
                      tau=self.params.tau)
        polyak_update(target=self.critic_target,
                      source=self.critic,
                      tau=self.params.tau)

    def _compute_critic_loss(self, sample, critic_loss_dict=None):
        q = self.critic(torch.cat((sample.obs, sample.ac), dim=-1))

        # Bellman backup for Q fucntion
        with torch.no_grad():
            if self._discrete_action:
                target_ac = one_hot_from_logits(self.policy_target(sample.next_obs))
            else:
                target_ac = self.policy_target(sample.next_obs)

            q_pi_target = self.critic_target(torch.cat((sample.next_obs, target_ac), dim=-1))

            backup = sample.rew + self.params.gamma * (1 - sample.done) * q_pi_target

        # MSE loss against Bellman backup
        loss_critic = F.mse_loss(q, backup)
        return loss_critic

    def _compute_policy_loss(self, sample, policy_loss_dict=None):
        if self._discrete_action:
            q_pi = self.critic(torch.cat((sample.obs, F.gumbel_softmax(self.policy(sample.obs), hard=True)), dim=-1))
        else:
            q_pi = self.critic(torch.cat((sample.obs, self.policy(sample.obs)), dim=-1))
        return -q_pi.mean()


    def _reset_rescale_noise(self, episode):
        if not self._reset_noise_ep == episode:
            # rescale noise
            explr_pct_remaining = max(0, self.params.n_exploration_episode - episode) / self.params.n_exploration_episode
            scale = self.params.final_noise_scale + (self.params.init_noise_scale - self.params.final_noise_scale) * explr_pct_remaining
            self.exploration.scale = scale
            # reset noise state
            self.exploration.reset()
            # store the last episode the noise is reset and rescaled
            self._reset_noise_ep = episode

    def on_episode_reset(self, episode):
        self._reset_rescale_noise(episode)



from agents.base_agent import BaseAgent
import torch
from utils.scale import action2newbounds


class BUS(BaseAgent):
    def initialize(self, params, init_dict=None):
        self.params = params
        self.shield = init_dict.shield
        self.desired_policy = init_dict.desired_policy
        self._ac_bounds = init_dict['ac_bounds']

    def step(self, obs, explore=False, init_phase=False):
        # The 'shield' method expects unnormalized actions (i.e., old action bounds), while the 'step' method is
        # designed to return normalized actions (i.e., new action bounds).
        # TODO: CHECK NUMPY
        # TODO: scale back action
        ac_des = self.desired_policy.act(obs)
        if self.params.to_shield:
            ac_shield = self.shield.shield(obs, ac_des)
            ac_shield = action2newbounds(ac_shield)
            return ac_shield, None
        return action2newbounds(ac_des), None


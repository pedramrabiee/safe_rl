from agents.base_agent import BaseAgent
import torch


class BUS(BaseAgent):
    def initialize(self, params, init_dict=None):
        self.params = params
        self.shield = init_dict.shield
        self.desired_policy = init_dict.desired_policy

    def step(self, obs, explore=False, init_phase=False):
        # TODO: CHECK SCALING, CHECK NUMPY
        # obs = self.obs_proc.proc(obs, proc_key='shield').squeeze()
        # obs.squeeze()
        if self.params.to_shield:
            return self.shield.shield(obs, self.desired_policy.act(obs)), None
        return self.desired_policy.act(obs), None

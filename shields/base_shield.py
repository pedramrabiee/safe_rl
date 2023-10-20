from agents.base_agent import BaseAgent
import torch

class BaseSheild(BaseAgent):
    def initialize(self, params, init_dict=None):
        raise NotImplementedError

    @torch.no_grad()
    def act(self, obs, explore=True):
        raise NotImplementedError

    @torch.no_grad()
    def filter(self, obs, ac, filter_dict=None):
        raise NotImplementedError

    def pre_train(self, samples, pre_train_dict=None):
        pass


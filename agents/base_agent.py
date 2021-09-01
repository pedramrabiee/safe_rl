from utils.misc import *
from utils.torch_utils import itermap_on_dict, load_state_dict_from_dict

class BaseAgent:
    def __init__(self, agent_type, ac_dim, ac_lim, timestep, replay_buffer, obs_proc=None, discrete_action=False):
        self._agent_type = agent_type
        self._ac_dim = ac_dim
        self._ac_lim = ac_lim
        self._timestep = timestep
        self._buffer = replay_buffer
        self._discrete_action = discrete_action
        self._curr_buf_id = 0
        self.obs_proc = obs_proc
        self.models_dict = None
        self.optimizers_dict = None

    def initialize(self, params, init_dict=None):
        raise NotImplementedError

    def reset(self):
        """implement actions to be taken when this function is called such as restarting noise,
         or stats at the beginning of each episode"""
        pass

    @torch.no_grad()
    def act(self, obs, explore=True):
        ac, info = self.step(obs, explore=explore)
        if torch.is_tensor(ac):
            ac = ac.numpy()
        return ac.squeeze(axis=0) if (ac.ndim > 1 and ac.shape[0] == 1) else ac, info

    def step(self, obs, explore=False):
        raise NotImplementedError

    def push_to_buffer(self, experience, push_to_all=False):
        if push_to_all:     # push experience to all buffers
            for buffer in self._buffer:
                buffer.push(experience)
        else:               # push experience to current buffer (the one with curr_buf_id)
            self.buffer.push(experience)

    def get_samples(self, inds, device='cpu'):
        return self.buffer.sample_by_indices(inds, device)

    def get_random_indices(self, batch_size):
        return self.buffer.get_random_indices(batch_size)

    def _calculate_bootstrap_value(self, info):
        raise NotImplementedError

    def sample_mode(self, device='cpu', sample_dict=None):
        if hasattr(self, 'policy'):
            self.policy.eval()
            self.policy = to_device(self.policy, device)
        if hasattr(self, 'dynamics'):
            self.dynamics.eval_mode(device=device)

    def eval_mode(self, device='cpu'):
        """Switch neural net model to evaluation mode"""
        if hasattr(self, 'models'):
            for model in self.models:
                model.eval()
                model = to_device(model, device)
        if hasattr(self, 'dynamics'):
            self.dynamics.eval_mode(device=device)

    def train_mode(self, device='cpu'):
        """Switch neural net model to training mode"""
        if hasattr(self, 'models'):
            for model in self.models:
                model.train()
                model = to_device(model, device)
        if hasattr(self, 'dynamics'):
            self.dynamics.train_mode(device=device)

    def optimize_agent(self, samples, optim_dict=None):
        raise NotImplementedError

    def after_optimize(self):
        """implement what needs to be updated for all agents simultaneously here"""
        pass

    @property
    def algo(self):
        return self._agent_type

    def get_params(self):
        params = {}
        # params['models'] = {k: model.state_dict() for k, model in enumerate(self.models)}
        # params['optimizers'] = {k: optim.state_dict() for k, optim in enumerate(self.optimizers)}
        params['models'] = itermap_on_dict(x=self.models_dict, func=lambda x: x.state_dict())
        params['optimizers'] = itermap_on_dict(x=self.optimizers_dict, func=lambda x: x.state_dict())

        if hasattr(self, 'extra_params_dict'):
            # params['extra_params'] = itermap_on_dict(x=self.extra_params_dict, func=lambda x: x.detach())
            params['extra_params'] = self.extra_params_dict
        return params

    def load_params(self, checkpoint, custom_load_list=None):
        if custom_load_list is None:
            load_state_dict_from_dict(state_dict=checkpoint['models'],
                                      load_to=self.models_dict)
            load_state_dict_from_dict(state_dict=checkpoint['optimizers'],
                                      load_to=self.optimizers_dict)
            # for k, model in enumerate(self.models):
            #     model.load_state_dict(checkpoint['models'][k])
            # for k, optim in enumerate(self.optimizers):
            #     optim.load_state_dict(checkpoint['optimizers'][k])
            if hasattr(self, 'extra_params_dict'):
                self.extra_params_dict = checkpoint['extra_params']

        else:
            for key in custom_load_list:
                load_state_dict_from_dict(state_dict=checkpoint['models'][key],
                                          load_to=self.models_dict[key])
                load_state_dict_from_dict(state_dict=checkpoint['optimizers'][key],
                                          load_to=self.optimizers_dict[key])
                if hasattr(self, 'extra_params_dict'):
                    if key in self.extra_params_dict:
                        self.extra_params_dict[key] = checkpoint['extra_params'][key]


    def get_buffer(self, to_tensor=False, device="cpu"):
        return self.buffer.get_buffer(to_tensor, device)

    def init_buffer(self, data):
        self.buffer.init_buffer(data)

    @property
    def buffer_size(self):
        return self.buffer.buffer_size

    @property
    def buffer(self):
        if isinstance(self._buffer, list):
            return self._buffer[self.curr_buf_id]
        else:
            return self._buffer

    @property
    def curr_buf_id(self):
        return self._curr_buf_id

    @curr_buf_id.setter
    def curr_buf_id(self, buf_id):
        self._curr_buf_id = buf_id



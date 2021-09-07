from utils.misc import get_ac_space_info
from attrdict import AttrDict

class AgentFactory:
    def __init__(self, env, config):
        self._env = env
        self._config = config

    def __call__(self, agent_type):
        if agent_type == 'mb':
            return self.instantiate_mb_agent()
        elif agent_type == 'ddpg':
            return self.instantiate_ddpg_agent()
        elif agent_type == 'sf':
            return self.instantiate_sf_agent()
        elif agent_type == 'cbf':
            return self.instantiate_cbf_filter()

    def instantiate_mb_agent(self):
        from agents.model_based.mb import MBAgent
        from dynamics.nominal_dynamics import get_nominal_dyn_cls

        agent_info = self._agent_info_from_env()
        params = self._config.get_agent_params('mb')
        train_env = self._config.setup['train_env']

        nom_dyn_cls, nom_dyn_params = get_nominal_dyn_cls(train_env, self._env)
        nominal_dyn_dict = dict(cls=nom_dyn_cls,
                                params=nom_dyn_params)
        params['dynamics_params']['nominal_dyn_dict'] = nominal_dyn_dict

        agent = MBAgent(agent_type='MB',
                        ac_dim=agent_info['ac_dim'],
                        ac_lim=dict(low=agent_info['ac_lim_low'],
                                    high=agent_info['ac_lim_high']),
                        timestep=agent_info['timestep'],
                        # replay_buffer=[buffer(self._config.buffer_size) for buffer in params.replay_buffer_cls],
                        replay_buffer=params.replay_buffer_cls(self._config.buffer_size),
                        discrete_action=agent_info['discrete_action'],
                        obs_proc=self._config.setup['obs_proc'],
                        custom_plotter=self._config.setup['custom_plotter']
                        )

        agent.initialize(params=params, init_dict=dict(bounds=agent_info['bounds'],
                                                       reward_gen=self._config.setup['reward_gen']))
        return agent

    def instantiate_ddpg_agent(self):
        from agents.model_free.ddpg import DDPGAgent
        from buffers.replay_buffer import ReplayBuffer

        agent_info = self._agent_info_from_env()

        agent = DDPGAgent(agent_type='DDPG',
                          ac_dim=agent_info['ac_dim'],
                          ac_lim=dict(low=agent_info['ac_lim_low'],
                                      high=agent_info['ac_lim_high']),
                          timestep=agent_info['timestep'],
                          replay_buffer=ReplayBuffer(self._config.buffer_size),
                          discrete_action=agent_info['discrete_action'],
                          obs_proc=self._config.setup['obs_proc'],
                          custom_plotter=self._config.setup['custom_plotter']
                          )

        params = self._config.get_agent_params('ddpg')
        agent.initialize(params)
        return agent

    def instantiate_sf_agent(self):
        from agents.model_based.sf import SFAgent
        from buffers.replay_buffer import ReplayBuffer
        from buffers.buffer_queue import BufferQueue

        agent_info = self._agent_info_from_env()

        agent = SFAgent(agent_type='SF',
                        ac_dim=agent_info['ac_dim'],
                        ac_lim=dict(low=agent_info['ac_lim_low'],
                                    high=agent_info['ac_lim_high']),
                        timestep=agent_info['timestep'],
                        replay_buffer=[ReplayBuffer(self._config.buffer_size),
                                       BufferQueue(self._config.buffer_size)],
                        discrete_action=agent_info['discrete_action'],
                        obs_proc=self._config.setup['obs_proc'],
                        custom_plotter=self._config.setup['custom_plotter']
                        )

        params = self._config.get_agent_params('sf')

        agent.initialize(params, init_dict=AttrDict(mf_agent=self(params.mf),
                                                    mb_agent=self(params.mb),
                                                    safety_filter=self(params.filter)))
        return agent


    def instantiate_cbf_filter(self):
        from filters.cbf_filter import CBFFilter
        from buffers.replay_buffer import ReplayBuffer

        agent_info = self._agent_info_from_env()

        agent = CBFFilter(
            agent_type='CBF',
            ac_dim=agent_info['ac_dim'],
            ac_lim=dict(low=agent_info['ac_lim_low'],
                        high=agent_info['ac_lim_high']),
            timestep=agent_info['timestep'],
            replay_buffer=ReplayBuffer(self._config.buffer_size),
            discrete_action=agent_info['discrete_action'],
            obs_proc=self._config.setup['obs_proc'],
            custom_plotter=self._config.setup['custom_plotter']
        )

        params = self._config.get_agent_params('cbf')
        agent.initialize(params)

        return agent


    # Helper functions
    def _agent_info_from_env(self):
        ac_dim, discrete_action = get_ac_space_info(self._env.action_space)
        if not discrete_action:
            ac_lim_high = self._env.action_space.high[0]  # FIXME: assumes bounds are the same for all dimensions
            ac_lim_low = self._env.action_space.low[0]  # FIXME: assumes bounds are the same for all dimensions
        else:
            ac_lim_high = ac_lim_low = None
        if hasattr(self._env, 'dt'):
            timestep = self._env.dt
        elif hasattr(self._env, 'robot'):   # used in safety_gym environments
            timestep = self._env.robot.sim.model.opt.timestep
        elif hasattr(self._env.unwrapped, 'robot'):    # used in safety_gym environments
            timestep = self._env.unwrapped.robot.sim.model.opt.timestep
        else:
            raise NotImplementedError


        return dict(
            ac_dim=ac_dim,
            discrete_action=discrete_action,
            ac_lim_high=ac_lim_high,
            ac_lim_low=ac_lim_low,
            timestep=timestep,
            bounds=self._env.action_bounds,
        )


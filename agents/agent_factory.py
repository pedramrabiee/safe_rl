from utils.misc import get_ac_space_info
from attrdict import AttrDict

class AgentFactory:
    def __init__(self, env, config):
        self._env = env
        self._config = config

    def __call__(self, agent_type):
        get_agent_instantiator_func_name = f"instantiate_{agent_type}_agent"
        # params_attr_name = f"{key}_params"

        if hasattr(self, get_agent_instantiator_func_name):
            get_params_func = getattr(self, get_agent_instantiator_func_name)
            return get_params_func()
        else:
            # Handle cases where the agent_type is not found
            raise ValueError(f"Agent instantiation method is not implemented: {agent_type}")


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
                                                    safety_filter=self(params.shield)))
        return agent

    def instantiate_cbf_filter(self):
        from shields.cbf_shield import CBFSheild
        from buffers.replay_buffer import ReplayBuffer

        agent_info = self._agent_info_from_env()

        agent = CBFSheild(
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

    def instantiate_cbf_test_agent(self):
        from agents.model_based.cbf_test import CBFTESTAgent
        from buffers.replay_buffer import ReplayBuffer

        agent_info = self._agent_info_from_env()

        agent = CBFTESTAgent(
            agent_type='CBFTest',
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

    def instantiate_backup_shield_agent(self):
        from shields.backup_shield import BackupShield, get_backup_shield_info_from_env
        agent_info = self._agent_info_from_env()

        agent = BackupShield(
            agent_type='BackupShield',
            ac_dim=agent_info['ac_dim'],
            ac_lim=dict(low=agent_info['ac_lim_low'],
                        high=agent_info['ac_lim_high']),
            timestep=agent_info['timestep'],
            replay_buffer=None,
            discrete_action=agent_info['discrete_action'],
            obs_proc=self._config.setup['obs_proc'],
            custom_plotter=self._config.setup['custom_plotter']
        )

        params = self._config.get_agent_params('backup_shield')
        agent.initialize(params, init_dict=AttrDict(
            **get_backup_shield_info_from_env(
                env=self._env,
                env_info=self._config.setup.train_env,
                obs_proc=self._config.setup.obs_proc)))
        return agent


    def instantiate_bus_agent(self):
        from shields.backup_shield import get_desired_policy
        from agents.model_based.bus import BUS
        from buffers.replay_buffer import ReplayBuffer
        agent_info = self._agent_info_from_env()
        agent = BUS(
            agent_type='BUS',
            ac_dim=agent_info['ac_dim'],
            ac_lim=dict(low=agent_info['ac_lim_low'],
                        high=agent_info['ac_lim_high']),
            timestep=agent_info['timestep'],
            replay_buffer=ReplayBuffer(self._config.buffer_size),
            discrete_action=agent_info['discrete_action'],
            obs_proc=self._config.setup['obs_proc'],
            custom_plotter=self._config.setup['custom_plotter']
        )

        params = self._config.get_agent_params('bus')
        agent.initialize(params, init_dict=AttrDict(shield=self('backup_shield'),
                                                    desired_policy=get_desired_policy(self._config.setup.train_env)))

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
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

        agent.initialize(params=params, init_dict=dict(ac_bounds=agent_info['ac_bounds'],
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

    def instantiate_td3_agent(self):
        from agents.model_free.td3 import TD3Agent
        from buffers.replay_buffer import ReplayBuffer

        agent_info = self._agent_info_from_env()

        agent = TD3Agent(agent_type='TD3',
                         ac_dim=agent_info['ac_dim'],
                         ac_lim=dict(low=agent_info['ac_lim_low'],
                                     high=agent_info['ac_lim_high']),
                         timestep=agent_info['timestep'],
                         replay_buffer=ReplayBuffer(self._config.buffer_size),
                         discrete_action=agent_info['discrete_action'],
                         obs_proc=self._config.setup['obs_proc'],
                         custom_plotter=self._config.setup['custom_plotter']
                         )

        params = self._config.get_agent_params('td3')
        agent.initialize(params)
        return agent

    def instantiate_sac_agent(self):
        from agents.model_free.sac import SACAgent
        from buffers.replay_buffer import ReplayBuffer

        agent_info = self._agent_info_from_env()

        agent = SACAgent(agent_type='SAC',
                         ac_dim=agent_info['ac_dim'],
                         ac_lim=dict(low=agent_info['ac_lim_low'],
                                     high=agent_info['ac_lim_high']),
                         timestep=agent_info['timestep'],
                         replay_buffer=ReplayBuffer(self._config.buffer_size),
                         discrete_action=agent_info['discrete_action'],
                         obs_proc=self._config.setup['obs_proc'],
                         custom_plotter=self._config.setup['custom_plotter']
                         )

        params = self._config.get_agent_params('sac')
        agent.initialize(params)
        return agent

    def instantiate_maddpg_agent(self):
        from agents.model_free.maddpg import MADDPGAgent
        from buffers.replay_buffer import ReplayBuffer

        agent_info = self._agent_info_from_env()

        agent = MADDPGAgent(agent_type='MADDPG',
                            ac_dim=agent_info['ac_dim'],
                            ac_lim=dict(low=agent_info['ac_lim_low'],
                                        high=agent_info['ac_lim_high']),
                            timestep=agent_info['timestep'],
                            replay_buffer=ReplayBuffer(self._config.buffer_size),
                            discrete_action=agent_info['discrete_action'],
                            obs_proc=self._config.setup['obs_proc'],
                            custom_plotter=self._config.setup['custom_plotter']
                            )
        params = self._config.get_agent_params('maddpg')
        # sum up action and observation spaces dimension of all agents in the environment for maddpg critic
        # TODO: Update this
        ac_spaces = self._env.action_space
        obs_spaces = self._env.observation_space
        critic_ac_dim = critic_obs_dim = 0
        for ac_space in ac_spaces:
            ac_dim, _ = get_ac_space_info(ac_space)
            critic_ac_dim += ac_dim
        for obs_space in obs_spaces:
            critic_obs_dim += obs_space.shape[0]   # FIXME: Make a get_obs_space_shape function and incorporate visual obs space shape
        init_dict = dict(critic_obs_dim=critic_obs_dim,
                         critic_ac_dim=critic_ac_dim)
        agent.initialize(params, init_dict)
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
        agent.initialize(params,
                         init_dict=AttrDict(
                             **get_backup_shield_info_from_env(
                                 env=self._env,
                                 env_info=self._config.setup.train_env,
                                 obs_proc=self._config.setup.obs_proc),
                             ac_bounds=agent_info['ac_bounds']
                         ))
        return agent


    def instantiate_rl_backup_shield_agent(self):
        from shields.rl_backup_shield import RLBackupShield
        from shields.backup_shield import get_backup_shield_info_from_env
        from buffers.replay_buffer import ReplayBuffer
        agent_info = self._agent_info_from_env()

        agent = RLBackupShield(
            agent_type='RLBackupShield',
            ac_dim=agent_info['ac_dim'],
            ac_lim=dict(low=agent_info['ac_lim_low'],
                        high=agent_info['ac_lim_high']),
            timestep=agent_info['timestep'],
            replay_buffer=ReplayBuffer(self._config.buffer_size),
            discrete_action=agent_info['discrete_action'],
            obs_proc=self._config.setup['obs_proc'],
            custom_plotter=self._config.setup['custom_plotter']
        )

        params = self._config.get_agent_params('rl_backup_shield')
        agent.initialize(params,
                         init_dict=AttrDict(
                             **get_backup_shield_info_from_env(
                                 env=self._env,
                                 env_info=self._config.setup.train_env,
                                 obs_proc=self._config.setup.obs_proc),
                             rl_backup=self(params.rl_backup_agent),
                             ac_bounds=agent_info['ac_bounds']
                         ))
        return agent

    def instantiate_rl_backup_shield_explorer_agent(self):
        from shields.rl_backup_shield_explorer import RLBackupShieldExplorer
        from shields.backup_shield import get_backup_shield_info_from_env
        from buffers.replay_buffer import ReplayBuffer
        agent_info = self._agent_info_from_env()

        agent = RLBackupShieldExplorer(
            agent_type='RLBackupShieldExplorer',
            ac_dim=agent_info['ac_dim'],
            ac_lim=dict(low=agent_info['ac_lim_low'],
                        high=agent_info['ac_lim_high']),
            timestep=agent_info['timestep'],
            replay_buffer=ReplayBuffer(self._config.buffer_size),
            discrete_action=agent_info['discrete_action'],
            obs_proc=self._config.setup['obs_proc'],
            custom_plotter=self._config.setup['custom_plotter']
        )

        params = self._config.get_agent_params('rl_backup_shield_explorer')
        agent.initialize(params,
                         init_dict=AttrDict(
                             **get_backup_shield_info_from_env(
                                 env=self._env,
                                 env_info=self._config.setup.train_env,
                                 obs_proc=self._config.setup.obs_proc),
                             rl_backup=self(params.rl_backup_agent),
                             ac_bounds=agent_info['ac_bounds']
                         ))
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
        desired_policy = get_desired_policy(self._config.setup.train_env)()
        agent.initialize(params, init_dict=AttrDict(shield=self('backup_shield'),
                                                    desired_policy=desired_policy,
                                                    ac_bounds=agent_info['ac_bounds']))

        return agent


    def instantiate_rlbus_agent(self):
        from shields.backup_shield import get_desired_policy
        from agents.model_based.rlbus import RLBUS
        from buffers.replay_buffer import ReplayBuffer

        agent_info = self._agent_info_from_env()
        agent = RLBUS(
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

        params = self._config.get_agent_params('rlbus')
        if params.use_mf_desired_policy:
            desired_policy = self(params.desired_policy_agent)
        else:
            desired_policy = get_desired_policy(self._config.setup.train_env)()

        agent.initialize(params, init_dict=AttrDict(shield=self(params.shield_agent),
                                                    desired_policy=desired_policy,
                                                    ac_bounds=agent_info['ac_bounds']))
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
            ac_bounds=self._env.action_bounds,
        )

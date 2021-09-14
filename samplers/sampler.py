import numpy as np
from tqdm import tqdm
from utils.misc import add_noise
from logger import logger
from utils import scale
from gym.spaces import Dict

class Sampler:
    def __init__(self, config):
        self._max_episode_len = int(config.max_episode_len)
        self._episode_steps_per_itr = int(config.episode_steps_per_itr)
        self._config = config

    def initialize(self, env, env_eval, agent, obs_proc, custom_plotter, safe_set=None, safe_set_eval=None):
        self.env = env
        self.agent = agent
        self.env_eval = env_eval
        self.obs_proc = obs_proc
        self.safe_set = safe_set
        self.safe_set_eval = safe_set_eval
        self.custom_plotter = custom_plotter
        logger.log('Sampler initialized...')
        self._init_sample_round = True
        self._episode_counter = 0
        self._min_buffer_size = int(self._config.n_episode_initial_data_collection * self._config.max_episode_len)
        self._reset_buffer_queue()

    def collect_data(self, train_itr):
        sample_dict = dict(episode=self.episode_completed)
        # set networks on eval mode
        self.agent.sample_mode(device=self._config.sampler_device, sample_dict=sample_dict)

        if self._init_sample_round:
            obs = self._env_reset()
            self._episode_time = 0
            self._episode_return = 0
            self._safety_violation_counter = 0
            self._safety_violations = []
            self._returns = []
            pbar = tqdm(total=self._min_buffer_size, desc='Initial Sampling Progress')
        else:
            obs = self._last_obs
            pbar = tqdm(total=self._episode_steps_per_itr, desc='Sampling on iteration %d' % train_itr)


        episode_steps = 0
        while True:
            init_phase = True if self.episode_completed < self._config.n_episode_init_phase else False
            ac, act_info = self.agent.act(obs, explore=True, init_phase=init_phase)      # Each agent takes care of observation processing in its step method
            next_obs, rew, done, info = self.env.step(ac)

            if act_info is not None:
                info = {**info, **act_info}
            # process observations store transition data in the temporary buffer
            obs_proc = self.obs_proc.proc(obs, proc_key='buffer')
            next_obs_proc = self.obs_proc.proc(next_obs, proc_key='buffer')
            obs_proc = obs_proc if isinstance(obs_proc, np.ndarray) else obs_proc
            next_obs_proc = next_obs_proc if isinstance(next_obs_proc, np.ndarray) else obs_proc

            self._buffer_queue(obs_proc,
                               ac.reshape(1, -1),
                               rew.reshape(1, -1),
                               next_obs_proc,
                               done.reshape(1, -1),
                               info)

            # sampler_push_obs append observation to the current timestep row.
            # You need to call sampler_push_action somewhere (either here or inside the agent) before calling sampler_push_obs
            self.custom_plotter.sampler_push_obs(obs)
            self._safety_violation_counter += int(not self.safe_set.is_geo_safe(self.obs_proc.proc(next_obs,
                                                                                                   proc_key='filter')))
            self._episode_return += rew
            self._episode_time += 1
            if done or self._episode_time >= self._max_episode_len:
                self._episode_counter += 1
                self._episode_time = 0
                # reset environment
                obs = self._env_reset()
                # reset agent
                self.agent.reset()
                self._returns.append(self._episode_return)
                self._safety_violations.append(self._safety_violation_counter)
                if len(self._returns) % self._config.episodes_per_return_log == 0:        # log returns every n episodes completed
                    if self._config.episodes_per_return_log == 1:
                        logger.add_tabular({'Sampling/Return': self._returns[0],
                                            'Sampling/SafetyViolations': self._safety_violations[0]}, stats=False, cat_key='episode')
                    else:
                        logger.add_tabular({'Sampling/Return': self._returns,
                                            'Sampling/SafetyViolations': self._safety_violations}, stats=True, cat_key='episode')
                    logger.dump_tabular(cat_key='episode', log=False, wandb_log=True, csv_log=False)
                    self._returns = []
                    self._safety_violations = []
                self._episode_return = 0
                self._safety_violation_counter = 0

                # dump plots
                self.custom_plotter.dump_sampler_plots(self._episode_counter)
                if 'cbf_value' in logger._plot_queue.keys():
                    logger.dump_plot_with_key(plt_key="cbf_value",
                                              filename='cbf_value_episode_%d' % self._episode_counter,
                                              plt_info=dict(
                                                  xlabel=r'Timestep',
                                                  ylabel=r'$h$'))
            else:
                obs = next_obs

            episode_steps += 1
            pbar.update(1)
            if episode_steps >= self._episode_steps_per_itr and self.buffer_size() > self._min_buffer_size:
                # add noise to data to be stored if required
                if self._config.add_noise_when_buffering.is_true:
                    self._add_noise()
                self.agent.push_to_buffer((self._obs_buf, self._ac_buf, self._rew_buf,
                                           self._next_obs_buf, self._done_buf, self._info_buf),
                                          push_to_all=True)
                self._reset_buffer_queue()
                self._init_sample_round = False
                pbar.close()
                self._last_obs = obs
                break

    def evaluate(self):
        logger.log('Evaluating...')
        # Move policy to sampler device
        self.agent.eval_mode(device=self._config.evaluation_device)

        max_episode_len_eval = int(self._config.max_episode_len_eval)

        obs = self._env_reset(eval=True)
        episode_time = eval_episode = 0
        episode_return = 0
        episode_safety_violation = 0
        returns = []
        safety_violations = []
        pbar = tqdm(total=self._config.n_episodes_evaluation * max_episode_len_eval, desc='Evaluating')
        while True:
            ac, _ = self.agent.act(obs, explore=False, init_phase=False)
            next_obs, rew, done, info = self.env_eval.step(ac)

            episode_return += rew
            episode_time += 1
            episode_safety_violation += int(not self.safe_set_eval.is_geo_safe(self.obs_proc.proc(next_obs,
                                                                                                  proc_key='filter')))

            if done or episode_time >= max_episode_len_eval:
                eval_episode += 1
                episode_time = 0
                obs = self._env_reset(eval=True)
                returns.append(episode_return)
                safety_violations.append(episode_safety_violation)
                episode_return = 0
                episode_safety_violation = 0
            else:
                obs = next_obs
            pbar.update(1)
            if eval_episode >= self._config.n_episodes_evaluation:
                logger.add_tabular({'Evaluation/Return': returns,
                                    'Evaluation/SafetyViolations': safety_violations}, stats=True, cat_key='evaluation_episode')
                pbar.close()
                break

    def sample(self, device='cpu', ow_batch_size=None):     # ow_batch_size: overwrite batch size
        if ow_batch_size is None:
            return self.sample(device=device, ow_batch_size=self._config.sampling_batch_size)
        batch_size = self.buffer_size() if (ow_batch_size == 'all') else ow_batch_size
        random_indices = self.agent.get_random_indices(batch_size)
        return self.agent.get_samples(random_indices, device=device)

    def _buffer_queue(self, obs, ac, rew, next_obs, done, info):
        """temporary replay buffer to minimize accessing to agents' buffer at each step of data collection"""
        if isinstance(self._obs_buf, list):     # for Dict observation-space from safety-gym (see _reset_buffer_queue below)
            self._obs_buf.append(obs)
            self._next_obs_buf.append(next_obs)
        else:
            if self._obs_buf is None:
                self._obs_buf = obs
                self._next_obs_buf = next_obs
            else:
                self._obs_buf = np.concatenate((self._obs_buf, obs), axis=0)
                self._next_obs_buf = np.concatenate((self._next_obs_buf, next_obs), axis=0)

        if self._ac_buf is None:
            self._ac_buf = ac
            self._rew_buf = rew
            self._done_buf = done
        else:
            self._ac_buf = np.concatenate((self._ac_buf, ac), axis=0)
            self._rew_buf = np.concatenate((self._rew_buf, rew), axis=0)
            self._done_buf = np.concatenate((self._done_buf, done), axis=0)
        self._info_buf.append(info)

    def _reset_buffer_queue(self):
        self._obs_buf = [] if isinstance(self.env.observation_space, Dict) else None
        self._next_obs_buf = [] if isinstance(self.env.observation_space, Dict) else None
        self._ac_buf, self._rew_buf, self._done_buf, self._info_buf =\
             None, None, None, []

    @property
    def episode_completed(self):
        return self._episode_counter

    def buffer_size(self):
        return self._rew_buf.shape[0] if self._init_sample_round else self.agent.buffer_size + \
                                                                      (0 if self._rew_buf is None
                                                                       else self._rew_buf.shape[0])

    def _add_noise(self):
        data_to_be_noised = self._config.add_noise_when_buffering.data
        noise_to_signal = self._config.add_noise_when_buffering.noise_to_signal
        if 'obs' in data_to_be_noised:
            self._obs_buf = add_noise(data_inp=self._obs_buf, noise_to_signal=noise_to_signal)
        if 'next_obs' in data_to_be_noised:
            self._next_obs_buf = add_noise(data_inp=self._next_obs_buf, noise_to_signal=noise_to_signal)
        if 'ac' in data_to_be_noised:
            self._ac_buf = add_noise(data_inp=self._ac_buf, noise_to_signal=noise_to_signal)
        if 'rew' in data_to_be_noised:
            self._rew_buf = add_noise(data_inp=self._rew_buf, noise_to_signal=noise_to_signal)


    def _env_reset(self, eval=False):
        if not eval:
            return self.safe_set.safe_reset() if self._config.env_spec_config.safe_reset else self.env.reset()
        return self.safe_set_eval.safe_reset() if self._config.env_spec_config.safe_reset else self.env_eval.reset()


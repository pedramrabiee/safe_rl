import numpy as np
from tqdm import tqdm
from utils.misc import add_noise
from logger import logger
from gym.spaces import Dict

class Sampler:
    """Class to sample data from environment for RL training"""
    def __init__(self, config):
        """
        Initialize the sampler

        Args:
            config (AttrDict): Configuration for sampling
        """
        self._max_episode_len = int(config.max_episode_len)
        self._episode_steps_per_itr = int(config.episode_steps_per_itr)
        self._config = config

    def initialize(self, env, env_eval, agent, obs_proc, custom_plotter, safe_set=None, safe_set_eval=None):
        """
        Initialize the sampler with environment, agent etc.

        Args:
            env (gym.Env): Training environment
            env_eval (gym.Env): Evaluation environment
            agent (RLAgent): RL agent
            obs_proc (ObservationProcessor): For processing observations
            custom_plotter (Plotter): For plotting data
            safe_set (SafeSet): For safety constraints
            safe_set_eval (SafeSet): For safety constraints during evaluation
        """
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
        self._min_buffer_size = int(self._config.n_episode_initial_data_collection * self._config.max_episode_len /
                                    self._config.step_save_freq)
        self._reset_buffer_queue()

    def collect_data(self, train_itr):
        """
        Collect a batch of data from environment

        Args:
            train_itr (int): Current training iteration
        """

        # Set networks on eval mode
        # Optional dictionary to be passed to agent for sampling
        sample_dict = dict(episode=self.episode_completed)
        self.agent.sample_mode(device=self._config.sampler_device, sample_dict=sample_dict)

        # Initialize variables for new episode
        if self._init_sample_round:
            # This is only executed the first time sampler collects data (initial data collection).
            obs = self._env_reset()
            self.agent.on_episode_reset(self.episode_completed)
            self._episode_time = 0
            self._episode_return = 0
            self._safety_violation_counter = 0
            # self._safety_criteria_violation_counter = 0
            self._safety_violations = []
            self._safety_criteria_violations = []
            self._returns = []
            # Progress bar for initial data collection
            pbar = tqdm(total=self._min_buffer_size * self._config.step_save_freq, desc='Initial Sampling Progress')
        else:
            obs = self._last_obs
            # Progress bar for regular sampling
            pbar = tqdm(total=self._episode_steps_per_itr * self._config.step_save_freq, desc='Sampling on iteration %d' % train_itr)


        episode_steps = 0

        # save_skip is used to subsample transitions before saving to the buffer
        # by saving transitions to buffer after every self._config.step_save_freq steps.
        # Saving to buffer is skipped when save_skip is not zero.
        # It is incremented each step and reset to 0 when it hits the threshold or the episode ends.
        save_skip = 0

        # Collect episode
        while True:
            # Check if still in initial episodes
            init_phase = True if self.episode_completed < self._config.n_episode_init_phase else False

            # Get action from agent
            # Each agent handles observation processing in its step method
            ac, act_info = self.agent.act(obs, explore=True, init_phase=init_phase)

            # Step environment
            next_obs, rew, done, info = self.env.step(ac)

            # Add extra info from agent's act_info dic
            if act_info is not None:
                info = {**info, **act_info}

            # Reset skip counter if episode ended
            if done: save_skip = 0

            # Process observations store transition data in the temporary buffer
            if not save_skip:
                # Process observations
                obs_proc = self.obs_proc.proc(obs, proc_key='buffer')
                next_obs_proc = self.obs_proc.proc(next_obs, proc_key='buffer')

                # Make sure observations are numpy arrays
                obs_proc = obs_proc if isinstance(obs_proc, np.ndarray) else obs_proc
                next_obs_proc = next_obs_proc if isinstance(next_obs_proc, np.ndarray) else obs_proc

                # Add transition
                # TODO: Changed ob
                self._buffer_queue(obs_proc.reshape(1, -1),
                                   ac.reshape(1, -1),
                                   rew.reshape(1, -1),
                                   next_obs_proc.reshape(1, -1),
                                   done.reshape(1, -1),
                                   info)

            # Increment skip counter
            save_skip += 1
            save_skip = save_skip if save_skip < self._config.step_save_freq else 0


            # sampler_push_obs append observation to the current timestep row.
            # TODO: # You need to call sampler_push_action somewhere (either here or inside the agent) before calling sampler_push_obs

            # Log observation
            self.custom_plotter.push(dict(obs=obs, ac=ac))

            # TODO: Fix this after you fixed safety class
            # Check safety violation
            self._safety_violation_counter += int(not self.safe_set.is_des_safe(self.obs_proc.proc(next_obs,
                                                                                                   proc_key='safe_set')))
            # TODO: Fix this after you fixed safety class
            # Check if agent has custom safety criteria
            # if hasattr(self.agent, 'is_safety_criteria_violated'):
            #     self._safety_criteria_violation_counter += int(self.agent.is_safety_criteria_violated(
            #         self.obs_proc.proc(next_obs,
            #                            proc_key='filter')))

            # Update episode return
            self._episode_return += rew
            self._episode_time += 1

            # Check if episode is done
            if done or self._episode_time >= self._max_episode_len:
                # Increment episode counter
                self._episode_counter += 1

                # Reset episode variables
                self._episode_time = 0
                save_skip = 0

                # Reset environment
                obs = self._env_reset()
                self.agent.on_episode_reset(self.episode_completed)

                # Store episode results
                self._returns.append(self._episode_return)

                # TODO: Fix this after you fixed safety class
                self._safety_violations.append(self._safety_violation_counter)
                # self._safety_criteria_violations.append(self._safety_criteria_violation_counter)

                # Occasionally log returns
                if len(self._returns) % self._config.episodes_per_return_log == 0:        # log returns every n episodes completed
                    # TODO: Fix this after you fixed safety class
                    if self._config.episodes_per_return_log == 1:
                        logger.add_tabular({'Sampling/Return': self._returns[0],
                                            'Sampling/SafetyViolations': self._safety_violations[0],
                                            # 'Sampling/SafetyCriteriaViolations': self._safety_criteria_violations[0]
                                            },
                                           stats=False, cat_key='episode')
                    else:
                        logger.add_tabular({'Sampling/Return': self._returns,
                                            'Sampling/SafetyViolations': self._safety_violations,
                                            # 'Sampling/SafetyCriteriaViolations': self._safety_criteria_violations
                                            },
                                           stats=True, cat_key='episode')
                    logger.dump_tabular(cat_key='episode', log=False, wandb_log=True, csv_log=False)
                    self._returns = []
                    self._safety_violations = []
                    # self._safety_criteria_violations = []

                # Reset accumulators
                self._episode_return = 0
                self._safety_violation_counter = 0
                # self._safety_criteria_violation_counter = 0

                # Plotting
                self.custom_plotter.dump(episode=self._episode_counter)

            else:
                # Continue episode
                obs = next_obs

            # Increment steps
            episode_steps += 1
            pbar.update(1)

            # Check if collected enough data
            if int(episode_steps / self._config.step_save_freq) >= self._episode_steps_per_itr and self.buffer_size() >= self._min_buffer_size:

                # Add noise if needed
                if self._config.add_noise_when_buffering.is_true:
                    self._add_noise()

                # Save buffer data
                self.agent.push_to_buffer((self._obs_buf, self._ac_buf, self._rew_buf,
                                           self._next_obs_buf, self._done_buf, self._info_buf),
                                          push_to_all=True)
                # Reset buffers
                self._reset_buffer_queue()
                self._init_sample_round = False

                # Close progress bar
                pbar.close()

                # Save last observation to continue from there on the next data collection round
                self._last_obs = obs
                break

    def evaluate(self):
        """
        Evaluate agent's performance
        """
        logger.log('Evaluating...')
        # Move policy to sampler device
        self.agent.eval_mode(device=self._config.evaluation_device)

        max_episode_len_eval = int(self._config.max_episode_len_eval)

        obs = self._env_reset(eval=True)
        self.agent.on_episode_reset(self.episode_completed)

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
            # TODO:
            episode_safety_violation += int(not self.safe_set_eval.is_des_safe(self.obs_proc.proc(next_obs,
                                                                                                  proc_key='filter')))

            if done or episode_time >= max_episode_len_eval:
                eval_episode += 1
                episode_time = 0
                obs = self._env_reset(eval=True)
                self.agent.on_episode_reset(self.episode_completed)

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
        """
        Sample a batch of data from buffer

        Args:
            device (str): Device to load data
            ow_batch_size (int): Batch size to override config

        Returns:
            (obs_batch, ac_batch, rew_batch, next_obs_batch, done_batch): Sampled batch
        """
        if ow_batch_size is None:
            return self.sample(device=device, ow_batch_size=self._config.sampling_batch_size)
        batch_size = self.buffer_size() if (ow_batch_size == 'all') else ow_batch_size
        random_indices = self.agent.get_random_indices(batch_size)
        return self.agent.get_samples(random_indices, device=device)

    def _buffer_queue(self, obs, ac, rew, next_obs, done, info):
        """
        Add transition to temporary buffer
        Temporary replay buffer to minimize accessing to agents' buffer at each step of data collection
        Args:
            obs (np.ndarray): Observation
            ac (np.ndarray): Action
            rew (float): Reward
            next_obs (np.ndarray): Next observation
            done (bool): Done
            info (dict): Additional info
        """

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
        """Reset the temporary buffers"""
        self._obs_buf = [] if isinstance(self.env.observation_space, Dict) else None
        self._next_obs_buf = [] if isinstance(self.env.observation_space, Dict) else None
        self._ac_buf, self._rew_buf, self._done_buf, self._info_buf =\
             None, None, None, []

    @property
    def episode_completed(self):
        """Get number of episodes completed"""
        return self._episode_counter

    def buffer_size(self):
        """Get size of buffer"""
        return self._rew_buf.shape[0] if self._init_sample_round else self.agent.buffer_size + \
                                                                      (0 if self._rew_buf is None
                                                                       else self._rew_buf.shape[0])

    def _add_noise(self):
        """Add noise to data before adding to buffer"""
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
        """
        Reset environment

        Args:
            eval (bool): Whether to reset evaluation env
        """
        if not eval:
            return self.safe_set.safe_reset() if self._config.env_spec_config.safe_reset else self.env.reset()
        return self.safe_set_eval.safe_reset() if self._config.env_spec_config.safe_reset else self.env_eval.reset()

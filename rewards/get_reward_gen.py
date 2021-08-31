def get_reward_gen(train_env):
    if train_env['env_id'] == 'Pendulum-v0':
        from rewards.inverted_pendulum_reward import inverted_pendulum_reward_gen
        return inverted_pendulum_reward_gen
    if train_env['env_id'] == 'HalfCheetah-v1' or 'HalfCheetah-v2' or 'HalfCheetah-v3': #TODO: check to see if all the versions have the same reward function
        from rewards.halfcheetah_reward import halfcheetah_reward_gen
        return halfcheetah_reward_gen
    else:
        print('Reward generator is not implemented for this environment.')

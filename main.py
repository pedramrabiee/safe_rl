import os
from trainers.trainer_factory import trainer_factory
from attrdict import AttrDict
from utils.seed import set_seed
from utils.process_observation import get_obsproc_cls
from utils.custom_plotter import get_custom_plotter_cls
from utils.get_reward_gen import get_reward_gen
from envs_utils.get_env_info import get_env_info
from logger import logger
import torch

torch.set_default_dtype(torch.float64)


def make_setup(env_nickname, agent, load_config_path=None):
    env_info = get_env_info(env_nickname)
    setup = AttrDict(agent=agent,
                     train_env=dict(env_id=env_info['env_id'],
                                    env_collection=env_info['env_collection'],
                                    env_nickname=env_nickname),
                     eval_env=dict(env_id=env_info['env_id'],
                                   env_collection=env_info['env_collection'],
                                   env_nickname=env_nickname),
                     load_config_path=load_config_path,
                     # enter the env_config.pickle file path (don't include env_config.pickle itself)
                     # if you wish to load the env_config file. otherwise, set it None
                     )

    setup['custom_plotter_cls'] = get_custom_plotter_cls(setup['train_env'], agent=agent)
    setup['reward_gen'] = get_reward_gen(setup['train_env'])
    setup['obs_proc_cls'] = get_obsproc_cls(setup['train_env'], agent=agent)

    return setup


if __name__ == "__main__":
    # Set random.seed, generate default_rng, torch.manual_seed, torch.cuda.manual_seed_all
    seed = set_seed()

    # See envs_utils.get_env_info for a list of acceptable environment nicknames
    setup = make_setup(env_nickname='pendulum',
                       agent='rlbus')

    root_dir = os.path.dirname(os.path.realpath(__file__))

    trainer_cls = trainer_factory(setup['agent'])
    trainer = trainer_cls(setup, root_dir)

    try:
        if trainer.config.evaluation_mode:
            # evaluate
            trainer.evaluate()
        else:
            # train
            trainer.train()
    except Exception as e:
        # email failure if not in debugging mode
        # logger.notify_failure(e)
        raise e
        # email success if not in debugging mode
        # logger.notify_completion()

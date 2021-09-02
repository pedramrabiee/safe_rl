import os
from rewards.get_reward_gen import get_reward_gen
from trainers.trainer_factory import trainer_factory
from attrdict import AttrDict
from utils.process_observation import get_obsproc_cls
from utils.custom_plotter import get_custom_plotter_cls
from logger import logger

if __name__ == "__main__":
    # accepted env_ids to date:
    # 'Pendulum-v0' from gym
    # 'Point' from safety_gym

    setup = AttrDict(agent='sf',
                     train_env=dict(env_id='Pendulum-v0',
                                    env_collection='gym'),
                     eval_env=dict(env_id='Pendulum-v0',
                                   env_collection='gym'),
                     load_config_path=None,  # enter the config.pickle file path (don't include config.pickle itself)
                     # if you wish to load the config file. otherwise, set it None
                     )

    setup['reward_gen'] = get_reward_gen(setup['train_env'])
    setup['obs_proc_cls'] = get_obsproc_cls(setup['train_env'])
    setup['custom_plotter_cls'] = get_custom_plotter_cls(setup['train_env'])

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
        logger.notify_failure(e)
        raise e
    else:
        # email success if not in debugging mode
        logger.notify_completion()


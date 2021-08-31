def trainer_factory(agent_type):
    if agent_type == 'mb':
        from trainers.mb_trainer import MBTrainer
        return MBTrainer
    elif agent_type == 'ddpg':
        from trainers.ddpg_trainer import DDPGTrainer
        return DDPGTrainer
    elif agent_type == 'sf':
        from trainers.sf_trainer import SFTrainer
        return SFTrainer
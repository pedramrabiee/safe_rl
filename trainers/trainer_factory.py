import importlib


def trainer_factory(agent_type):
    if agent_type in shared_trainer:
        agent_type = shared_trainer[agent_type]
    try:
        trainer_module = importlib.import_module(f'trainers.{agent_type}_trainer')
        trainer_class_name = f'{agent_type.upper()}Trainer'
        trainer_class = getattr(trainer_module, trainer_class_name)
        return trainer_class
    except ImportError:
        # Handle cases where the module or class is not found
        raise ImportError(f"Trainer for agent_type '{agent_type}' not found")
    except AttributeError:
        # Handle cases where the class is not found in the module
        raise AttributeError(f"Trainer class '{trainer_class_name}' not found")



# To use another trainer for a method
shared_trainer = dict(td3='mf',
                      ddpg='mf',
                      sac='mf')

_nicknames = {
        'pendulum': {'env_id': 'Pendulum-v0', 'env_collection': 'gym'},
        'point': {'env_id': 'Point', 'env_collection': 'safety_gym'},
        'cbf_test': {'env_id': 'cbf_test', 'env_collection': 'misc_env'},
        'multi_dashpot': {'env_id': 'multi_dashpot', 'env_collection': 'misc_env'}
    }


def get_env_info(nickname):
    return _nicknames[nickname]

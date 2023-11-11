from envs_utils.gym.pendulum.pendulum_configs import env_config as pendulum_config
from envs_utils.safety_gym.point.point_configs import env_config as point_config

class AffineInControlTorchDyn:
    def dynamics(self, state, u):
        return self.f(state) + self.g(state) * u



class PendulumTorchDyn(AffineInControlTorchDyn):
    def f(self, state):
        return pendulum_config.f_torch(state)

    def g(self, state):
        return pendulum_config.g_torch(state)


# TODO: make automatic torch dyn instantiator
def get_torch_dyn(env):
    dyn_mapping = {'Pendulum-v0': PendulumTorchDyn}
    try:
        return dyn_mapping[env.spec.id]
    except KeyError:
        raise ValueError('Invalid input value')
import numpy as np

def euler_integrate_time_inv_from_func(fun, obs, ac, ts, trig_idx=None):
    """has_trig used when you have cosine and sin of angle theta in your states. In this case, using integration can
     result in invalid cosine and sine for the next timestep (i.e. cos^2(next_theta) + sin^2(next_theta) \neq.
     In this case pass the indices for [cosine, sine, theta_dot]
    """
    next_obs = obs + fun(obs, ac) * ts
    if trig_idx is not None:
        theta_dot = next_obs[:, trig_idx[2]]
        theta = np.arctan2(obs[:, trig_idx[1]], obs[:, trig_idx[0]])  # TODO: check this
        theta_next = euler_integrate_time_inv_from_deriv(theta_dot, theta, ts)
        next_obs[:, trig_idx[0]] = np.cos(theta_next)
        next_obs[:, trig_idx[1]] = np.sin(theta_next)

    return next_obs

def euler_integrate_time_inv_from_deriv(deriv_value, obs, ts, trig_idx=None):
    """has_trig used when you have cosine and sin of angle theta in your states. In this case, using integration can
     result in invalid cosine and sine for the next timestep (i.e. cos^2(next_theta) + sin^2(next_theta) \neq.
     In this case pass the indices for [cosine, sine, theta_dot]
    """
    next_obs = obs + deriv_value * ts
    if trig_idx is not None:
        theta_dot = next_obs[:, trig_idx[2]]
        theta = np.arctan2(obs[:, trig_idx[1]], obs[:, trig_idx[0]])  # TODO: check this
        theta_next = euler_integrate_time_inv_from_deriv(theta_dot, theta, ts)
        next_obs[:, trig_idx[0]] = np.cos(theta_next)
        next_obs[:, trig_idx[1]] = np.sin(theta_next)

    return next_obs

def rk4_integrate(fun, t, y, ts, has_trig=None, integrate_with_substitution=False):
    pass
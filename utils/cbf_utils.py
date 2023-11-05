import numpy as np
from utils.optim import solve_qp, make_box_constraints_from_bounds
from utils.misc import scalar_to_vector


def cbf_qp(h, Lfh, Lgh, alpha_func, Q, c, A_u=None, b_u=None):
    A = np.vstack((-Lgh, A_u)) if A_u is not None else -Lgh
    b = np.hstack((Lfh + alpha_func(h), b_u)) if b_u is not None else Lfh + alpha_func(h)
    return solve_qp(Q, c, A, b)


def cbf_qp_box_constrained(h, Lfh, Lgh, alpha_func, Q, c, u_bound):
    A_u, b_u = make_box_constraints_from_bounds(u_bound)
    return cbf_qp(h=h, Lfh=Lfh, Lgh=Lgh, alpha_func=alpha_func, Q=Q, c=c, A_u=A_u, b_u=b_u)




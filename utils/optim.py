from cvxopt import matrix, solvers
import numpy as np
from scipy.linalg import block_diag



def qp_from_np(P, q, G=None, h=None, A=None, b=None, return_sol_dict=False):
    solvers.options['show_progress'] = False
    sol = solvers.qp(*np_to_matrix([P, q, G, h, A, b]))
    if return_sol_dict:
        return sol
    return np.array(sol['x'])

def slacked_qp_approx_feasibility_check(sol, eps_start_index, eps_threshold=1e-8):
    eps = sol[eps_start_index:]
    return np.abs(eps).max() < eps_threshold

def np_to_matrix(np_matrix_list):
    a = [matrix(x) if x is not None else None for x in np_matrix_list]
    return a

def add_slack_var_to_qp(P, q, G, A, K_eps = 1e48, index_to_start_adding_slack_var=None):
    if index_to_start_adding_slack_var is None:
        num_slack_vars = G.shape[0]
    else:
        num_slack_vars = G.shape[0] - index_to_start_adding_slack_var

    new_P = block_diag(P, K_eps * np.eye(num_slack_vars))
    new_q = np.vstack([q, np.zeros((num_slack_vars, 1))])
    new_A = np.hstack([A, np.zeros((A.shape[0], num_slack_vars))])
    new_G = np.hstack([G, np.vstack([np.zeros((G.shape[0] - num_slack_vars, num_slack_vars)),
                                     np.eye(num_slack_vars)])])

    return new_P, new_q, new_G, new_A

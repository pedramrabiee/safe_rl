import cvxopt
from cvxopt import matrix, solvers, spmatrix, sparse
import numpy as np


def qp_from_np(P, q, G=None, h=None, A=None, b=None):
    solvers.options['show_progress'] = False
    sol = solvers.qp(*np_to_matrix([P, q, G, h, A, b]))
    return np.array(sol['x'])

def np_to_matrix(np_matrix_list):
    a = [matrix(x) if x is not None else None for x in np_matrix_list]
    return a
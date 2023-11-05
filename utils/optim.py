from cvxopt import matrix, solvers
import cvxpy as cp
import numpy as np
from scipy.linalg import block_diag


# Inequality constrained QP
def solve_qp(Q, c, A, b):
    """
        Solve a Quadratic Programming (QP) problem.

        Parameters:
            Q (numpy.ndarray): Quadratic cost matrix of shape (n, n).
            c (numpy.ndarray): Linear cost vector of shape (n,).
            A (numpy.ndarray): Constraint matrix of shape (p, n).
            b (numpy.ndarray): Constraint vector of shape (p,).

        Returns:
            numpy.ndarray: Optimal solution of the QP problem, a vector of shape (n,).
        """

    # Define the optimization variables
    n = Q.shape[0]
    x = cp.Variable(n)

    # Define the objective function
    objective = cp.Minimize(0.5 * cp.quad_form(x, Q) + c @ x)

    # Define the constraints
    constraints = [A @ x <= b]

    # Formulate the QP problem
    problem = cp.Problem(objective, constraints)

    # Solve the QP problem
    problem.solve()

    # Extract the optimal solution
    optimal_x = x.value


    # Return the optimal solution and the optimal objective value
    return optimal_x, problem.value


def solve_lp(c, A, b):
    """
    Solve a Linear Programming (LP) problem.

    Parameters:
        c (numpy.ndarray): Linear cost vector of shape (n,).
        A (numpy.ndarray): Constraint matrix of shape (m, n).
        b (numpy.ndarray): Constraint vector of shape (m,).

    Returns:
        numpy.ndarray: Optimal solution of the LP problem, a vector of shape (n,).
    """
    # Define the optimization variables
    n = c.shape[0]
    x = cp.Variable(n)

    # Define the objective function
    objective = cp.Minimize(c @ x)

    # Define the constraints
    constraints = [A @ x <= b]

    # Formulate the LP problem
    problem = cp.Problem(objective, constraints)

    # Solve the LP problem
    problem.solve()

    # Extract the optimal solution
    optimal_x = x.value

    # Return the optimal solution
    return optimal_x, problem.value


def make_box_constraints_from_bounds(bounds):
    """
    Create box constraints (inequality constraints) for variables based on given lower and upper bounds.

    Given a set of lower and upper bounds for variables, this function generates box constraints in the form of
    inequality constraints that ensure each variable stays within its specified bounds.

    Parameters:
        bounds (list or numpy.ndarray): A list or numpy array of shape (n, 2) where n is the number of variables.
            Each row represents the lower and upper bounds for a variable.

    Returns:
        numpy.ndarray: A matrix A of shape (2n, n) representing the inequality coefficients for the box constraints.
        numpy.ndarray: A vector b of length 2n representing the right-hand side of the box constraints.

    Examples:
    bounds = [[-1, 1], [0, 2], [2, 4]]
    A, b = make_box_constraints_from_bounds(bounds)
    print(A)
    [[-1.  0.  0.]
     [ 0. -1.  0.]
     [ 0.  0. -1.]
     [ 1.  0.  0.]
     [ 0.  1.  0.]
     [ 0.  0.  1.]]
    print(b)
    [ 1.  0.  2. -1. -0. -4.]
    """
    bounds = np.array(bounds)
    num_variables = bounds.shape[0]

    # Create A
    A = np.vstack((-np.eye(num_variables), np.eye(num_variables)))

    # Create b for x_i >= x_min and x_i <= x_max
    b = np.concatenate((-bounds[:, 0], bounds[:, 1]))

    return A, b



# Deprecated: The following methods are used for cvxopt qp implementation
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
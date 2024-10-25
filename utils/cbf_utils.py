import numpy as np
from utils.optim import solve_qp, make_box_constraints_from_bounds
from torch.autograd import grad
import torch


def cbf_qp(h, Lfh, Lgh, alpha_func, Q, c, A_u=None, b_u=None):
    """
    Solve a Quadratic Program (QP) for Control Barrier Functions (CBF) with optional box constraints.

    This function formulates and solves a QP to enforce Control Barrier Functions (CBF) while optionally
    enforcing box constraints on the control input.

    Parameters:
    - h (numpy.ndarray): The CBF value at the current state.
    - Lfh (numpy.ndarray): The Lie derivative of the CBF with respect to f.
    - Lgh (numpy.ndarray): The Lie derivative of the CBF with respect to g.
    - alpha_func (function): The extended class-K alpha function.
    - Q (numpy.ndarray): The positive definite Q matrix for the QP cost.
    - c (numpy.ndarray): The linear term for the QP cost.
    - A_u (numpy.ndarray, optional): The inequality constraint matrix for control input.
    - b_u (numpy.ndarray, optional): The inequality constraint vector for control input.

    Returns:
    - numpy.ndarray: The optimal control input that satisfies the CBF and optional constraints.
    """
    A = np.vstack((-Lgh, A_u)) if A_u is not None else -Lgh
    b = np.hstack((Lfh + alpha_func(h), b_u)) if b_u is not None else Lfh + alpha_func(h)
    # A, b = preprocess_constraint(A, b)
    return solve_qp(Q, c, A, b)


def cbf_qp_box_constrained(h, Lfh, Lgh, alpha_func, Q, c, u_bound):
    """
        Solve a Quadratic Program (QP) for Control Barrier Functions (CBF) with box constraints on the control input.

        This function formulates and solves a QP to enforce Control Barrier Functions (CBF) while also enforcing box
        constraints on the control input.

        Parameters:
        - h (numpy.ndarray): The CBF value at the current state.
        - Lfh (numpy.ndarray): The Lie derivative of the CBF with respect to f.
        - Lgh (numpy.ndarray): The Lie derivative of the CBF with respect to g.
        - alpha_func (function): The extended class-K alpha function.
        - Q (numpy.ndarray): The positive definite Q matrix for the QP cost.
        - c (numpy.ndarray): The linear term for the QP cost.
        - u_bound (tuple or list): A tuple or list representing the upper and lower bounds for control input.

        Returns:
        - numpy.ndarray: The optimal control input that satisfies the CBF and box constraints.
        """

    A_u, b_u = make_box_constraints_from_bounds(u_bound)
    return cbf_qp(h=h, Lfh=Lfh, Lgh=Lgh, alpha_func=alpha_func, Q=Q, c=c, A_u=A_u, b_u=b_u)


# minimum intervention qp with box constraints
def min_intervention_qp_box_constrained(h, Lfh, Lgh, alpha_func, u_des, u_bound):
    """
        Solve a Quadratic Program (QP) for Minimum Intervention Control Barrier Functions (CBF) with box constraints.

        This function formulates and solves a QP to achieve minimum intervention while enforcing Control Barrier Functions (CBF)
        and box constraints on the control input.

        Parameters:
        - h (numpy.ndarray): The CBF value at the current state.
        - Lfh (numpy.ndarray): The Lie derivative of the CBF with respect to f.
        - Lgh (numpy.ndarray): The Lie derivative of the CBF with respect to g.
        - alpha_func (function): The extended class-K alpha function.
        - u_des (numpy.ndarray): The desired control input.
        - u_bound (tuple or list): A tuple or list representing the upper and lower bounds for control input.

        Returns:
        - numpy.ndarray: The optimal control input that minimizes intervention while satisfying the CBF and box constraints.

        """
    Q = 2 * np.eye(u_des.shape[0])
    c = np.array([-2 * u_des])
    return cbf_qp_box_constrained(h, Lfh, Lgh, alpha_func, Q, c, u_bound)

def preprocess_constraint(A, b):
    """
        Preprocess inequality constraints to improve numerical stability in a Quadratic Program (QP).

        This function normalizes the rows of the constraint matrix A and the constraint vector b to improve
        numerical stability in the QP solving process. If any row in A contains zeros, it will consider
        the second minimum value after 0 to avoid division by zero.

        Parameters:
        - A (numpy.ndarray): The inequality constraint matrix.
        - b (numpy.ndarray): The inequality constraint vector.

        Returns:
        - numpy.ndarray: The normalized inequality constraint matrix.
        - numpy.ndarray: The normalized inequality constraint vector.
        """
    if A.shape[1] > 1:
        # Create a boolean mask for zero elements in A
        zero_mask = (A == 0)
        # Find the second smallest value after 0 for each row
        second_min_after_zero = np.partition(np.abs(A), 1, axis=1)[:, 1]
        # Replace zeros with the second minimum value
        modified_A = A + zero_mask * second_min_after_zero[:, np.newaxis]
        # Calculate norm_vals as the minimum absolute value in each row
        norm_vals = np.min(np.abs(modified_A), axis=1)
        # Avoid division by zero by setting norm_vals to 1 when it's zero
        norm_vals[norm_vals == 0] = 1
        return A / norm_vals[:, np.newaxis], b / norm_vals
    else:
        # Handle the case when A has a single column
        norm_vals = np.abs(A)
        norm_vals[norm_vals == 0] = 1
        # Normalize A and b by dividing only non-zero elements
        return A / norm_vals, b / norm_vals.squeeze()



def lie_deriv(x, func, field):
    x.requires_grad_()
    h_val = func(x)
    func_deriv = grad(h_val, x)[0]
    x.requires_grad_(requires_grad=False)
    field_val = field(x)
    res = torch.mm(func_deriv.T, field_val)
    return res
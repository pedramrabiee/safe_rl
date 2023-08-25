import numpy as np
from scipy.linalg import block_diag
from utils.optim import qp_from_np, add_slack_var_to_qp, slacked_qp_approx_feasibility_check
from utils.misc import e_or, e_not

def polyderon_from_lb_ub(dim, lb, ub):
    A = np.block([
        [np.eye(dim)],
        [-np.eye(dim)]
    ])

    if isinstance(lb, list):
        lb = np.array(lb)
    if isinstance(ub, list):
        ub = np.array(ub)

    if np.isscalar(lb):
        lb = np.full(dim, lb)
    if np.isscalar(ub):
        ub = np.full(dim, ub)
    assert lb.shape[0] == dim, 'lower bound dimension mismatch'
    assert ub.shape[0] == dim, 'upper bound dimension mismatch'

    b = np.hstack((ub, -lb)).reshape(-1, 1)
    A, b = remove_inf_from_polyhedron(A, b)
    H = np.hstack((A, b))
    return A, b, H

def remove_inf_from_polyhedron(A, b):
    not_inf_mask = e_not(e_or(list(b == np.inf), list(b == -np.inf)))
    A = A[not_inf_mask, :]
    b = b[not_inf_mask]
    return A, b


def lqr_cftoc_openloop(x0, Ad, Bd, Q, P, R, N, Ax, bx, Af, bf, Au, bu):
    x0 = x0.reshape(-1, 1)
    n = Ad.shape[0]
    nu = Bd.shape[1]

    P, q, G, h, A, b = get_lqr_cftoc_matrices(Ad, Bd, Q, P, R, N, Ax, bx, Af, bf, Au, bu)
    A = A @ x0

    sol = qp_from_np(P=P, q=q, G=G, h=h, A=A, b=b, return_sol_dict=True)
    is_optimal = sol['status'] == 'optimal'
    ans = np.array(sol['x'])
    x = None
    u = None
    is_feasible = False
    if is_optimal:
        x = ans[:n * N].reshape(N, n)
        u = ans[n * N:].reshape(N, nu)
        is_feasible = True

    return x, u, is_feasible

def get_lqr_cftoc_matrices(Ad, Bd, Q, P, R, N, Ax, bx, Af, bf, Au, bu):
    n = Ad.shape[0]
    barH = block_diag(np.kron(np.eye(N - 1), Q), P, np.kron(np.eye(N), R))
    q = np.zeros((barH.shape[0], 1))

    # equality constraints
    blk1 = np.kron(np.eye(N), np.eye(n))
    blk2 = np.block(
        [
            [np.zeros((n, N * n))],
            [np.hstack([np.kron(np.eye(N - 1), -Ad), np.zeros(((N - 1) * n, n))])]
        ]
    )
    G0eq = np.hstack([blk1 + blk2, np.kron(np.eye(N), -Bd)])
    E0eq = np.vstack([Ad, np.zeros(((N - 1) * n, n))])

    # inequality constraints
    G0in = block_diag(np.kron(np.eye(N - 1), Ax), Af, np.kron(np.eye(N), Au))
    E0in = np.vstack([np.kron(np.ones((N-1, 1)), bx), bf, np.kron(np.ones((N, 1)), bu)])

    return dict(P=barH, q=q, G=G0in, h=E0in, A=G0eq, b=E0eq)


class LQR_CFTOC:
    def __init__(self, Ad, Bd, Q, P, R, N, Ax, bx, Af, bf, Au, bu):
        qp_matrices = self._get_lqr_cftoc_matrices(Ad, Bd, Q, P, R, N, Ax, bx, Af, bf, Au, bu)
        self.P = qp_matrices['P']
        self.q = qp_matrices['q']
        self.G = qp_matrices['G']
        self.h = qp_matrices['h']
        self.A = qp_matrices['A']
        self.b = qp_matrices['b']
        self.N = N
        self.n = Ad.shape[0]
        self.nu = Bd.shape[1]

    def _get_lqr_cftoc_matrices(self, Ad, Bd, Q, P, R, N, Ax, bx, Af, bf, Au, bu):
        n = Ad.shape[0]
        barH = block_diag(np.kron(np.eye(N - 1), Q), P, np.kron(np.eye(N), R))
        q = np.zeros((barH.shape[0], 1))

        # equality constraints
        blk1 = np.kron(np.eye(N), np.eye(n))
        blk2 = np.block(
            [
                [np.zeros((n, N * n))],
                [np.hstack([np.kron(np.eye(N - 1), -Ad), np.zeros(((N - 1) * n, n))])]
            ]
        )
        G0eq = np.hstack([blk1 + blk2, np.kron(np.eye(N), -Bd)])
        E0eq = np.vstack([Ad, np.zeros(((N - 1) * n, n))])

        # inequality constraints
        G0in = block_diag(np.kron(np.eye(N - 1), Ax), Af, np.kron(np.eye(N), Au))
        E0in = np.vstack([np.kron(np.ones((N - 1, 1)), bx), bf, np.kron(np.ones((N, 1)), bu)])

        return dict(P=barH, q=q, G=G0in, h=E0in, A=G0eq, b=E0eq)

    def solve(self, x0):
        x0 = x0.reshape(-1, 1)
        b = self.b @ x0

        sol = qp_from_np(P=self.P, q=self.q, G=self.G, h=self.h, A=self.A, b=b, return_sol_dict=True)
        is_optimal = sol['status'] == 'optimal'
        ans = np.array(sol['x'])
        x = None
        u = None
        is_feasible = False
        if is_optimal:
            x = ans[:self.n * self.N].reshape(self.N, self.n)
            u = ans[self.n * self.N:].reshape(self.N, self.nu)
            is_feasible = True

        return x, u, is_feasible



class LQR_CFTOC_WO_SUBSTITUTION:
    def __init__(self, Ad, Bd, Q, P, R, N, Ax, bx, Af, bf, Au, bu):
        qp_matrices = self._get_lqr_cftoc_matrices(Ad, Bd, Q, P, R, N, Ax, bx, Af, bf, Au, bu)
        self.H = qp_matrices['H']
        self.F = qp_matrices['F']
        self.G0 = qp_matrices['G0']
        self.E0 = qp_matrices['E0']
        self.w0 = qp_matrices['w0']
        self.N = N
        self.n = Ad.shape[0]
        self.nu = Bd.shape[1]

    def _get_lqr_cftoc_matrices(self, Ad, Bd, Q, P, R, N, Ax, bx, Af, bf, Au, bu):
        n = Ad.shape[0]
        Sx = np.vstack([np.linalg.matrix_power(Ad, p + 1) for p in range(N)])
        Su = []
        for c in range(N):
            zeros = np.zeros((c * Bd.shape[0], Bd.shape[1]))
            rest = np.vstack([np.linalg.matrix_power(Ad, p) @ Bd for p in range(N - c)])
            Su.append(np.vstack([zeros, rest]))
        Su = np.hstack(Su)
        print(Su)

        Qbar = block_diag(np.kron(np.eye(N - 1), Q), P)
        Rbar = np.kron(np.eye(N), R)
        H = Su.T @ Qbar @ Su + Rbar
        F = Sx.T @ Qbar @ Su

        G0 = []
        for c in range(N):
            zeros = np.zeros((c * (Ax @ Bd).shape[0], (Ax @ Bd).shape[1]))
            rest = [Ax @ np.linalg.matrix_power(Ad, p) @ Bd for p in range(N - c - 1)]
            rest = np.vstack(rest) if not rest == [] else np.zeros((0, zeros.shape[1]))
            rest = np.vstack([rest, Af @ np.linalg.matrix_power(Ad, N - c - 1) @ Bd])
            G0.append(np.vstack([zeros, rest]))
        G0 = np.hstack(G0)
        G0 = np.vstack([np.kron(np.eye(N), Au), G0])

        E0 = np.vstack([-Ax @ np.linalg.matrix_power(Ad, p + 1) for p in range(N - 1)])
        E0 = np.vstack([E0, -Af @ np.linalg.matrix_power(Ad, N)])
        E0 = np.vstack([np.zeros((N * Au.shape[0], n)), E0])
        w0 = np.vstack([
            np.kron(np.ones((N, 1)), bu),
            np.kron(np.ones((N - 1, 1)), bx),
            bf
        ])
        return dict(H=H, F=F, G0=G0, E0=E0, w0=w0)

    def solve(self, x0):
        x0 = x0.reshape(-1, 1)
        q = self.F.T @ x0
        E0 = self.w0 + self.E0 @ x0

        sol = qp_from_np(P=self.H, q=q, G=self.G0, h=E0, A=None, b=None, return_sol_dict=True)
        is_optimal = sol['status'] == 'optimal'
        ans = np.array(sol['x'])
        u = None
        is_feasible = False
        if is_optimal:
            u = ans.reshape(self.N, self.nu)
            is_feasible = True
        return u, is_feasible
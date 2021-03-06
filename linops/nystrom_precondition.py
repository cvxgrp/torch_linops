from __future__ import annotations
import torch
from dataclasses import dataclass
import linops as lo

@dataclass
class _NystromApproximation:
    U: torch.Tensor
    Lambda_hat: torch.Tensor

@dataclass
class NystromPreconditioner:
    A_hat: _NystromApproximation
    mu: torch.Tensor

    def __call__(self, v):
        U = self.A_hat.U
        Lambda_hat = self.A_hat.Lambda_hat
        mu = self.mu
        return (Lambda_hat[-1] + mu) * (U @ ((U.T @ v) / (Lambda_hat + mu))) + (v - U @ (U.T @ v))

def construct_approximation(
        A: lo.LinearOperator,
        l_0: int,
        l_max: int,
        power_iter_count: int,
        error_tol):
    n = A.shape[0]
    assert A.shape[0] == A.shape[1]

    Y = torch.Tensor((n, 0))
    Omega = torch.Tensor((n, 0))
    E = torch.inf
    m = l_0
    break_early = False
    U = torch.Tensor([])
    Lambda_hat = torch.Tensor([])
    nu = torch.sqrt(n) * torch.finfo().eps

    while E > error_tol:
        Omega_0 = torch.randn((n, m))
        Omega_0, _ = torch.linalg.qr(Omega_0)
        Y_0 = lo.operator_matrix_product(A, Omega_0)
        Omega = torch.hstack([Omega, Omega_0])
        Y = torch.hstack([Y, Y_0])
        # Check the meaning of everything on line 8 of E.2
        Y_nu = Y + nu * Omega
        C = torch.linalg.cholesky(Omega.T @ Y_nu)
        B = torch.linalg.solve(C, Y_nu)  # Verify this is efficient
        U, Sigma, _ = torch.linalg.svd(B)  # Figure out what the second argument to SVD is
        Lambda_hat = torch.relu(Sigma * Sigma - nu)

        if break_early:
            break

        E = randomized_power_err_est(A, U, Lambda_hat, power_iter_count)
        m, l_0 = l_0, 2 * l_0
        if l_0 > l_max:
            l_0 = l_0 - m
            m = l_max - l_0
            break_early = True
    return _NystromApproximation(U, Lambda_hat)


def randomized_power_err_est(A, U, Lambda_hat, q):
    g = torch.randn(A.shape[0])
    v_0 = g / torch.linalg.norm(g)
    E_hat = torch.inf
    for _ in range(q):
        v = A @ v_0 - U @ (Lambda_hat * (U.T @ v_0))
        E_hat = v_0.T @ v
        v = v / torch.linalg.norm(v)
        v_0 = v
    return E_hat

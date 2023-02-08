from __future__ import annotations

import math

import torch
import linops as lo

"""
This module contains various ways to compute divergences.

Reference paper 1: https://arxiv.org/pdf/2010.09649.pdf
Reference paper 2: https://arxiv.org/pdf/2301.07825.pdf
"""

def hutchinson(A: lo.LinearOperator, m: int=400):
    k, ell = A.shape
    assert k == ell
    if k <= m:
        return exact_divergence(A)
    total = 0
    for _ in range(m):
        z = (2 * torch.randint(2, size=(k,), device=A.device) - 1).float()
        total = total + (z * (A @ z)).sum()
    return total / m



def hutchpp(A: lo.LinearOperator, m: int=102):
    """
    Algorithm taken from reference paper above on Hutch++.
    """
    n, ell = A.shape
    assert n == ell
    assert m % 3 == 0
    if n <= m:
        return exact_divergence(A)
    k = m // 3
    S = 2.0 * torch.randint(0, 2, (n, k), device=A.device) - 1.0
    G = 2.0 * torch.randint(0, 2, (n, k), device=A.device) - 1.0

    BS = lo.operator_matrix_product(A, S)
    Q, _ = torch.linalg.qr(BS)
    G_prime = G - Q @ (Q.T @ G)
    total = 0
    for i in range(Q.shape[1]):
        z = Q[:,i]
        total = total + (z * (A @ z)).sum()

    for i in range(k):
        z = G_prime[:,i]
        total = total + (z * (A @ z)).sum() / k

    return total

def exact_divergence(A: lo.LinearOperator):
    m, n = A.shape
    assert m == n
    one = torch.ones(1, device=A.device)
    divergence = 0
    for i in range(m):
        divergence = divergence + A[i, i] @ one
    return divergence

def xtrace(A, m: int=80):
    n, n1 = A.shape
    assert n == n1
    if n <= m:
        return exact_divergence(A)

    m = m // 2
    def normalize_columns(M):
        return M / torch.linalg.vector_norm(M, dim=0)

    def diag_of_AB(A, B):
        return torch.sum(A * B, dim=0)

    Z = torch.randn(n, m)
    Omega = math.sqrt(n) * normalize_columns(Z)
    Y = A @ Omega
    Q, R = torch.linalg.qr(Y)

    W = Q.T @ Omega
    S = normalize_columns(torch.linalg.inv(R).T)
    scale = (n - m + 1) / (n - torch.linalg.vector_norm(W, dim=0)**2 +
                           torch.abs(diag_of_AB(S, W) * torch.linalg.vector_norm(S, dim=0))**2)
    Z = A @ Q
    H = Q.T @ Z
    HW = H @ W
    T = Z.T @ Omega
    dSW = diag_of_AB(S, W)
    dSHS = diag_of_AB(S, H @ S)
    dTW = diag_of_AB(T, W)
    dWHW = diag_of_AB(W, HW)
    dSRmHW = diag_of_AB(S, R - HW)
    dTmHRS = diag_of_AB(T - H.T @ W, S)

    ests = torch.trace(H) - dSHS + (
            dWHW - dTW + dTmHRS * dSW + torch.abs(dSW)**2 * dSHS + dSW * dSRmHW) * scale

    return torch.mean(ests), torch.std(ests) / math.sqrt(m)


def xnystrace(A: lo.LinearOperator, m: int=80):
    """
    Entirely untested. Never run code.
    """
    n, n1 = A.shape
    assert n == n1
    m = m // 2
    def normalize_columns(M):
        return M / torch.linalg.vector_norm(M, dim=0)

    def diag_of_AB(A, B):
        return torch.sum(A * B, dim=0)

    Omega = math.sqrt(n) * normalize_columns(torch.randn(n, m))
    Y = A @ Omega

    nu = torch.finfo(Omega.dtype).eps * torch.linalg.vector_norm(Y) / math.sqrt(n)
    Y = Y + nu * Omega

    
    Q, R = torch.linalg.qr(Y)
    H = Omega.T @ Y
    C = torch.linalg.cholesky((H + H.T) / 2, upper=True)

    B = torch.linalg.solve_triangular(C, R, upper=True, left=False) # Double check this

    QQ, RR = torch.linalg.qr(Omega)
    WW = QQ.T @ Omega
    SS = normalize_columns(torch.linalg.inv(RR).T)

    scale = (n - m + 1) / (n - torch.linalg.vector_norm(WW, dim=0)**2 +
                           torch.abs(diag_of_AB(SS, WW) * torch.linalg.vector_norm(SS, dim=0))**2)
    W = Q.T @ Omega
    S =  torch.linalg.solve_triangular(C.T, B, upper=False, left=False) * torch.diag(torch.linalg.inv(H))**(-0.5) # Double check triangular solve
    dSW = diag_of_AB(S, W)
    ests = torch.linalg.norm(B, 'fro')**2 - torch.linalg.vector_norm(S, dim=0)**2 + torch.abs(dSW)**2 * scale - nu * n
    return torch.mean(ests), torch.std(ests) / math.sqrt(m)

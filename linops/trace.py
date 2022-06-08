from __future__ import annotations

import torch
import linops as lo

"""
This module contains various ways to compute divergences.

Reference paper: https://arxiv.org/pdf/2010.09649.pdf
"""

def hutchinson(A: lo.LinearOperator, m: int=1000):
    k, ell = A.shape
    assert k == ell
    if k <= m:
        return exact_divergence(A)
    total = 0
    for _ in range(m):
        z = (2 * torch.randint(2, size=k, device=A.device) - 1).float()
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

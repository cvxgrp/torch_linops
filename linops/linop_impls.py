import torch
from linops import LinearOperator

class ZeroOperator(LinearOperator):
    supports_operator_matrix = True
    def __init__(self, shape, adjoint=None):
        self._shape = shape
        self._adjoint = ZeroOperator((shape[1], shape[0]), self) \
                if adjoint is None else adjoint

    def _matmul_impl(self, v):
        shape = (self.shape[0], *v.shape[1:])
        return torch.zeros(shape, device=v.device)

class IdentityOperator(LinearOperator):
    supports_operator_matrix = True
    efficient_inverse = True
    def __init__(self, n):
        self._adjoint = self
        self._shape = (n, n)

    def _matmul_impl(self, v):
        return v

    def solve_A_x_eq_b(self, b, x0=None):
        return b


class DiagonalOperator(LinearOperator):
    supports_operator_matrix = True
    efficient_inverse = True
    def __init__(self, diag):
        self._adjoint = self
        self._diag = diag
        m, = diag.shape
        self._shape = (m, m)

    def _matmul_impl(self, v):
        if len(v.shape) == 1:
            return self._diag * v
        else:
            return self._diag[:, None] * v

    def solve_A_x_eq_b(self, b, x0=None):
        if len(b.shape) == 1:
            return b / self._diag
        else:
            return b / self._diag[:, None]

class KKTOperator(LinearOperator):
    def __init__(self, H: LinearOperator, A: LinearOperator):
        k, ell = H.shape
        assert k == ell
        m, n = A.shape
        assert n == k
        self._shape = (m + n, m + n)
        self._adjoint = self
        self._A = A
        self._H = H
        self._m = m
        self._n = n

    def _matmul_impl(self, y):
        n = self._n
        return torch.hstack([
            self._H @ y[:n] - self._A.T @ y[n:],
            -self._A @ y[:n]
        ])

import torch
from linops.linops import LinearOperator, Symmetric, AdjointShield

class ZeroOperator(LinearOperator):
    supports_operator_matrix = True
    def __init__(self, shape, adjoint=None):
        super().__init__()
        self._shape = shape
        self._adjoint = ZeroOperator((shape[1], shape[0]), AdjointShield(self)) \
                if adjoint is None else adjoint

    def _matmul_impl(self, v):
        shape = (self.shape[0], *v.shape[1:])
        return torch.zeros(shape, device=v.device)

class SparseOperator(LinearOperator):
    supports_operator_matrix = True
    def __init__(self, X, XT, adjoint=None):
        super().__init__()
        self._shape = X.shape
        self._X = X
        if adjoint is None:
            self._adjoint = SparseOperator(XT, X, AdjointShield(self))
        else:
            self._adjoint = adjoint

    def _matmul_impl(self, v):
        return self._X @ v


class IdentityOperator(LinearOperator):
    supports_operator_matrix = True
    efficient_inverse = True
    def __init__(self, n):
        super().__init__()
        self._adjoint = Symmetric
        self._shape = (n, n)

    def _matmul_impl(self, v):
        return v

    def solve_A_x_eq_b(self, b, x0=None):
        return b


class DiagonalOperator(LinearOperator):
    supports_operator_matrix = True
    efficient_inverse = True
    def __init__(self, diag):
        super().__init__()
        self._adjoint = Symmetric
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
        super().__init__()
        k, ell = H.shape
        assert k == ell
        m, n = A.shape
        assert n == k
        self._shape = (m + n, m + n)
        self._adjoint = Symmetric
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

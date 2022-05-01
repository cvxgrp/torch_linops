import torch
import operator



def aslinearoperator(A):
    if isinstance(A, LinearOperator):
        return A
    else:
        return MatrixOperator(A)

class LinearOperator:
    _adjoint = None

    def __call__(self, x):
        return self @ x

    def __matmul__(self, b):
        if isinstance(b, LinearOperator):
            return _JoinOperator(self, b)
        else:
            return self._matmul_impl(b)

    def __rmatmul__(self, a):
        if isinstance(a, LinearOperator):
            return _JoinOperator(a, self)
        else:
            return self.T @ a

    def __mul__(self, c):
        return ScaleOperator(c) @ self
    
    def __truediv__(self, c):
        return ScaledownOperator(c) @ self

    def __rmul__(self, c):
        return self @ ScaleOperator(c)

    # __radd__ not needed because we only support sums of linear operators
    def __add__(self, b):
        if isinstance(b, LinearOperator):
            return _BinaryOperator(self, b, operator.add)
        else:
            return NotImplemented

    # __rsub__ not needed because we only support sums of linear operators
    def __sub__(self, b):
        if isinstance(b, LinearOperator):
            return _BinaryOperator(self, b, operator.sub)
        else:
            return NotImplemented

    def __neg__(self):
        return _UrnaryOperator(self, operator.neg)

    def __pos__(self):
        return _UrnaryOperator(self, operator.pos)

    def __pow__(self, n):
        return _PowOperator(self, n)

    @property
    def T(self):
        """
        Can probably be auto-generated with
            D_x(v @ A @ x) = A^T v
        """
        if self._adjoint is None:
            raise NotImplementedError("Implement automatic adjoints")
        else:
            return self._adjoint

    def _matmul_impl(self, v):
        v
        raise NotImplementedError()

    def solve_I_p_lambda_AT_A_x_eq_b(self, lambda_, b):
        self, lambda_, b
        return (IdentityOperator() + lambda_ * self.T @ self).solve_A_x_eq_b(b)

    def solve_A_x_eq_b(self, b):
        b
        raise NotImplementedError()

class IdentityOperator(LinearOperator):
    def __init__(self):
        self._adjoint = self

    def _matmul_impl(self, v):
        return v

class DiagonalOperator(LinearOperator):
    def __init__(self, diag):
        self._adjoint = self
        self._diag = diag

    def _matmul_impl(self, v):
        return self._diag * v

class ScaleOperator(LinearOperator):
    def __init__(self, c):
        self._c = c
        self._adjoint = self
    def _matmul_impl(self, v):
        return self._c * v

class ScaledownOperator(LinearOperator):
    def __init__(self, c):
        self._c = c
        self._adjoint = self
    def _matmul_impl(self, v):
        return v / self._c

    @property
    def T(self):
        return self

class MatrixOperator(LinearOperator):
    def __init__(self, matrix, adjoint=None):
        self._matrix = matrix
        if adjoint is None:
            self._adjoint = MatrixOperator(self._matrix.T, self)
        else:
            self._adjoint = adjoint
        self.__ATA_matrix = None
        self.__lambda_cache = None
        self.__IplATA_matrix = None

    def _matmul_impl(self, v):
        return self._matrix @ v

    def solve_I_p_lambda_AT_A_x_eq_b(self, lambda_, b):
        if self.__ATA_matrix is None:
            A = self._matrix.clone()
            self.__ATA_matrix = A.T @ A
            self.__I = torch.eye(A.shape[0])

        if self.__lambda_cache != lambda_:
            lATA = lambda_ * (self.__ATA_matrix)
            self.__IplATA_matrix = self.__I + lATA
        return torch.linalg.solve(self.__IplATA_matrix, b)

    def solve_A_x_eq_b(self, b):
        return torch.linalg.solve(self._matrix, b)

class _PowOperator(LinearOperator):
    def __init__(self, linop, k:int):
        self._linop = linop
        self._k = k
        assert k >= 0
        self._adjoint = _PowOperator(self._linop.T, self._k)

    def _matmul_impl(self, v):
        k = self._k
        while k > 0:
            v = self._linop @ v

class _BinaryOperator(LinearOperator):
    """transpose must distribute over op"""
    def __init__(self, left, right, op):
        self._left = left
        self._right = right
        self._op = op
        self._adjoint = _BinaryOperator(self._left.T, self._right.T, self._op)

    def _matmul_impl(self, y):
        return self._op(self._left @ y, self._right @ y)
    
class _UrnaryOperator(LinearOperator):
    """ transpose must pass through op """
    def __init__(self, linop, op):
        self._linop = linop
        self._op = op
        self._adjoint = _UrnaryOperator(self._linop.T, self._op) 

    def _matmul_impl(self, y):
        return self._op(self._linop @ y)
    
class _JoinOperator(LinearOperator):
    def __init__(self, left, right, adjoint=None):
        self._left = left
        self._right = right
        if adjoint is None:
            self._adjoint = _JoinOperator(self._right.T, self._left.T, self)
        else:
            self._adjoint = adjoint

    def _matmul_impl(self, y):
        return self._left @ (self._right @ y)

class _AdjointSelectionOperator(LinearOperator):
    def __init__(self, input_shape, idxs, adjoint):
        self._adjoint = adjoint
        self._input_shape = input_shape
        self._idxs = idxs

    def _matmul_impl(self, y):
        z = torch.zeros(self._input_shape, dtype=y.dtype, device=y.device)
        z[self._idxs] = y
        return z.reshape(-1)

class SelectionOperator(LinearOperator):
    def __init__(self, input_shape, idxs):
        self._adjoint = _AdjointSelectionOperator(input_shape, idxs, self)
        self._input_shape = input_shape
        self._idxs = idxs

    def _matmul_impl(self, X):
        if X.shape != self._input_shape:
            X = X.reshape(self._input_shape)
        return X[self._idxs]

    def solve_I_p_lambda_AT_A_x_eq_b(self, lambda_, b):
        LHS = torch.ones_like(b)
        LHS = LHS.reshape(self._input_shape)
        LHS[self._idxs] += lambda_
        LHS = LHS.reshape(-1)
        return b / LHS

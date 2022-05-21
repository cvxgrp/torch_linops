import torch
import operator
from linops.cg import CG
import linops.nystrom_precondition as nystrom 



def operator_matrix_product(A, M):
    assert A.shape[1] == M.shape[0]
    out = torch.empty((A.shape[0], M.shape[1]))
    for i in range(M.shape[1]):
        out[:, i] = A @ M[:, i]
    return out


def aslinearoperator(A):
    if isinstance(A, LinearOperator):
        return A
    else:
        return MatrixOperator(A)

class LinearOperator:
    _adjoint = None
    _shape: tuple[int, int] = None
    _nystrom_sketch = None
    _last_solve_of_x = None

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
        return _ScaleOperator(c, self.shape) @ self
    
    def __truediv__(self, c):
        return _ScaledownOperator(c, self.shape) @ self

    def __rmul__(self, c):
        return self @ _ScaleOperator(c, self.shape)

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
    
    def __getitem__(self, key):
        out = torch.empty(self.shape[0])
        in_ = torch.empty(self.shape[1])
        if not isinstance(key, tuple):
            # Only recieved one index, so we're only applying to the output
            shape = (out[key].shape[0], self.shape[1])
            return SelectionOperatorV2(shape, key) @ self
        shape = (out[key[0]].shape[0], in_[key[1]][0])
        return SelectionOperatorV2(shape, key[0]) @ self @ SelectionOperatorV2(shape, key[1]).T

    @property
    def shape(self) -> tuple[int, int]:
        return self._shape

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
        raise NotImplementedError()

    def solve_I_p_lambda_AT_A_x_eq_b(self, lambda_, b, x0=None, **kwargs):
        precondition = None
        if 'precondition' in kwargs:
            assert kwargs['precondition'] == 'nystrom'
            if self._nystrom_sketch is None:
                self._nystrom_sketch = nystrom.construct_approximation(
                    self,
                    l_0=kwargs.get('l_0', 10),
                    l_max=kwargs.get('l_max', 500),
                    power_iter_count=kwargs.get('power_iter_count', 50),
                    error_tol=kwargs.get('error_tol', 1.0),
                    )
            precondition = nystrom.NystromPreconditioner(
                    self._nystrom_sketch, lambda_)
        self._last_solve_of_x = CG(
                self + lambda_ * IdentityOperator(self.shape[0]),
                precondition)(b, self._last_solve_of_x)
        return self._last_solve_of_x

    def solve_A_x_eq_b(self, b):
        raise NotImplementedError()

class FusedOperator(LinearOperator):
    def __init__(self, shape: tuple[int, int],
                       in_slices: list[slice],
                       out_slices: list[slice],
                       ops: list[LinearOperator],
                       adjoint=None):
        self._shape = shape
        self._in_slices = in_slices
        self._out_slices = out_slices
        self._ops = ops
        if adjoint is None:
            self._adjoint = FusedOperator((shape[1], shape[0]), out_slices,
                    in_slices, ops, self)
        else:
            self._adjoint = adjoint
    
    def _matmul_impl(self, v):
        out = torch.empty(self._shape[0], device=v.device, dtype=v.dtype)
        for in_s, out_s, op in zip(self._in_slices, self._out_slices, self._ops,
                strict=True):
            out[out_s] += op @ v[in_s]
        return out

class IdentityOperator(LinearOperator):
    def __init__(self, n):
        self._adjoint = self
        self._shape = (n, n)

    def _matmul_impl(self, v):
        return v

class DiagonalOperator(LinearOperator):
    def __init__(self, diag):
        self._adjoint = self
        self._diag = diag
        m, = diag.shape
        self._shape = (m, m)

    def _matmul_impl(self, v):
        return self._diag * v

class _ScaleOperator(LinearOperator):
    def __init__(self, c, shape):
        self._c = c
        self._adjoint = self
        self._shape = shape
    def _matmul_impl(self, v):
        return self._c * v

class _ScaledownOperator(LinearOperator):
    def __init__(self, c, shape):
        self._c = c
        self._adjoint = self
        self._shape = shape
    def _matmul_impl(self, v):
        return v / self._c

    @property
    def T(self):
        return self

class MatrixOperator(LinearOperator):
    def __init__(self, matrix, adjoint=None):
        self._matrix = matrix
        assert len(matrix.shape) == 2
        self._shape = matrix.shape
        if adjoint is None:
            self._adjoint = MatrixOperator(self._matrix.T, self)
        else:
            self._adjoint = adjoint
        self.__ATA_matrix = None
        self.__lambda_cache = None
        self.__IplATA_matrix = None

    def _matmul_impl(self, v):
        return self._matrix @ v

    def solve_I_p_lambda_AT_A_x_eq_b(self, lambda_, b, **kwargs):
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
        assert linop.shape[0] == linop.shape[1]
        self._adjoint = _PowOperator(self._linop.T, self._k)
        self._shape = linop.shape

    def _matmul_impl(self, v):
        k = self._k
        while k > 0:
            v = self._linop @ v

class _BinaryOperator(LinearOperator):
    """transpose must distribute over op"""
    def __init__(self, left, right, op, adjoint=None):
        assert left.shape == right.shape
        self._left = left
        self._right = right
        self._op = op
        if adjoint is None:
            self._adjoint = _BinaryOperator(
                    self._left.T, self._right.T, self._op, self)
        self._shape = left.shape

    def _matmul_impl(self, y):
        return self._op(self._left @ y, self._right @ y)
    
class _UrnaryOperator(LinearOperator):
    """ transpose must pass through op """
    def __init__(self, linop, op):
        self._linop = linop
        self._op = op
        self._adjoint = _UrnaryOperator(self._linop.T, self._op) 
        self._shape = linop.shape

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
        self._shape = (left.shape[0], right.shape[1])

    def _matmul_impl(self, y):
        return self._left @ (self._right @ y)

class _AdjointSelectionOperator(LinearOperator):
    def __init__(self, input_shape, idxs, adjoint, shape):
        self._adjoint = adjoint
        self._input_shape = input_shape
        self._idxs = idxs
        self._shape = shape

    def _matmul_impl(self, y):
        z = torch.zeros(self._input_shape, dtype=y.dtype, device=y.device)
        z[self._idxs] = y
        return z.reshape(-1)

class SelectionOperatorV1(LinearOperator):
    def __init__(self, input_shape, idxs):
        in_shape = 1
        for i in input_shape:
            in_shape *= i
        self._shape = (len(idxs[0]), in_shape)
        self._adjoint = _AdjointSelectionOperator(input_shape,
                idxs, self, (self._shape[1], self._shape[0]))
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

class SelectionOperatorV2(LinearOperator):
    def __init__(self, shape, idxs):
        self._shape = shape
        self._adjoint = _AdjointSelectionOperatorV2(idxs,
                (self._shape[1], self._shape[0]), self)
        self._idxs = idxs

    def _matmul_impl(self, X):
        return X[self._idxs]

    def solve_I_p_lambda_AT_A_x_eq_b(self, lambda_, b):
        LHS = torch.ones_like(b)
        LHS[self._idxs] += lambda_
        return b / LHS

class _AdjointSelectionOperatorV2(LinearOperator):
    def __init__(self,  idxs, shape, adjoint):
        self._adjoint = adjoint
        self._idxs = idxs
        self._shape = shape

    def _matmul_impl(self, y):
        z = torch.zeros(self.shape[0], dtype=y.dtype, device=y.device)
        z[self._idxs] = y
        return z.reshape(-1)

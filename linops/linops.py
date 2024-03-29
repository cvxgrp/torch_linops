import torch
import operator
import linops as lo
from linops.cg import CG
from linops.minres import minres
import linops.nystrom_precondition as nystrom 



def operator_matrix_product(A, M):
    assert A.shape[1] == M.shape[0]
    if A.supports_operator_matrix:
        return A @ M
    out = torch.empty((A.shape[0], M.shape[1]), device=M.device, dtype=M.dtype)
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
    device = None
    supports_operator_matrix = False
    efficient_inverse = False

    def __call__(self, x):
        return self @ x

    def __matmul__(self, b):
        if isinstance(b, LinearOperator):
            return _JoinOperator(self, b)
        elif len(b.shape) == 1:
            return self._matmul_impl(b)
        elif len(b.shape) == 2:
            assert self.shape[1] == b.shape[0]
            if self.supports_operator_matrix:
                return self._matmul_impl(b)
            else:
                out = torch.empty((self.shape[0], b.shape[1]), device=b.device, dtype=b.dtype)
                for i in range(b.shape[1]):
                    out[:, i] = self._matmul_impl(b[:, i])
                return out

    def __rmatmul__(self, a):
        if isinstance(a, LinearOperator):
            return _JoinOperator(a, self)
        else:
            return (self.T @ a.T).T

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
        if not isinstance(key, tuple):
            # Only recieved one index, so we're only applying to the output
            S = _generate_selector(self.shape[0], key[0])
            return S @ self
        leftS = _generate_selector(self.shape[0], key[0])
        rightS = _generate_selector(self.shape[1], key[1], flip=True)
        return leftS @ self @ rightS.T

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
            x = torch.ones(self.shape[1], device=self.device, requires_grad=True)
            g = self @ x
            self._adjoint = VectorJacobianOperator(g, x, self)
        return self._adjoint

    def _matmul_impl(self, v):
        raise NotImplementedError()

    def solve_I_p_lambda_AT_A_x_eq_b(self, lambda_, b, x0=None, **kwargs):
        precondition = None
        if 'precondition' in kwargs and kwargs['precondition'] is not None:
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
                lo.IdentityOperator(self.shape[1]) + lambda_ * (self.T @ self),
                precondition)(b)
        return self._last_solve_of_x

    def solve_A_x_eq_b(self, b, x0=None):
        if self is self._adjoint:
            return minres(self, b, x0=x0)
        else:
            raise NotImplementedError()

class _ScaleOperator(LinearOperator):
    supports_operator_matrix = True
    efficient_inverse = True
    def __init__(self, c, shape):
        self._c = c
        self._adjoint = self
        self._shape = shape

    def _matmul_impl(self, v):
        return self._c * v

    def solve_A_x_eq_b(self, b, x0=None):
        return b / self._c

class _ScaledownOperator(LinearOperator):
    supports_operator_matrix = True
    efficient_inverse = True
    def __init__(self, c, shape):
        self._c = c
        self._adjoint = self
        self._shape = shape

    def _matmul_impl(self, v):
        return v / self._c

    def solve_A_x_eq_b(self, b, x0=None):
        return b * self._c

class MatrixOperator(LinearOperator):
    supports_operator_matrix = True
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
        self.__diagonal_ATA = None

    def _matmul_impl(self, v):
        return self._matrix @ v

    def _direct_solve_I_p_lambda_AT_A_x_eq_b(self, lambda_, b, **kwargs):
        if self.__ATA_matrix is None:
            A = self._matrix.clone()
            self.__ATA_matrix = A.T @ A

        if self.__lambda_cache != lambda_:
            lATA = lambda_ * (self.__ATA_matrix)
            torch.diagonal(lATA)[:] += 1
            self.__IplATA_matrix = lATA
        return torch.linalg.solve(self.__IplATA_matrix, b)

    def solve_A_x_eq_b(self, b, x0=None):
        return torch.linalg.solve(self._matrix, b)

    def solve_I_p_lambda_AT_A_x_eq_b(self, lambda_, b, x0=None, **kwargs):
        return self._direct_solve_I_p_lambda_AT_A_x_eq_b(lambda_, b,
                                                         x0=x0, **kwargs)

    def _cg_solve_I_p_lambda_AT_A_x_eq_b(self, lambda_, b, x0=None, **kwargs):
        if self.__diagonal_ATA is None:
            self.__diagonal_ATA = torch.linalg.vector_norm(self._matrix, axis=0)
        M = DiagonalOperator(lambda_ * self.__diagonal_ATA + 1)
        self._last_solve_of_x = CG(
                IdentityOperator(self.shape[1]) + lambda_ * (self.T @ self),
                M)(b)
        return self._last_solve_of_x

class _PowOperator(LinearOperator):
    def __init__(self, linop, k:int):
        self.supports_operator_matrix = linop.supports_operator_matrix
        self._linop = linop
        self._k = k
        assert k >= 0
        assert linop.shape[0] == linop.shape[1]
        self._adjoint = _PowOperator(self._linop.T, self._k)
        self._shape = linop.shape
        self.efficient_inverse = linop.efficient_inverse

    def _matmul_impl(self, v):
        k = self._k
        while k > 0:
            v = self._linop @ v
            k -= 1
        return v

    def solve_A_x_eq_b(self, b, x0=None):
        if self.efficient_inverse:
            k = self._k
            while k > 0:
                b = self._linop.solve_A_x_eq_b(b, x0)
                k -= 1
            return b
        else:
            return super().solve_A_x_eq_b(b, x0=x0)

class _BinaryOperator(LinearOperator):
    """transpose must distribute over op"""
    def __init__(self, left, right, op, adjoint=None):
        assert left.shape == right.shape
        self.supports_operator_matrix = \
                left.supports_operator_matrix and right.supports_operator_matrix
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
    def __init__(self, linop, op, adjoint=None):
        self._linop = linop
        self.supports_operator_matrix = linop.supports_operator_matrix
        self._op = op
        if adjoint is None:
            self._adjoint = _UrnaryOperator(self._linop.T, self._op, self)
        else:
            self._adjoint = adjoint
        self._shape = linop.shape

    def _matmul_impl(self, y):
        return self._op(self._linop @ y)
    
class _JoinOperator(LinearOperator):
    def __init__(self, left, right, adjoint=None):
        self._left = left
        self._right = right
        self.efficient_inverse = \
                left.efficient_inverse and right.efficient_inverse

        self.supports_operator_matrix = \
                left.supports_operator_matrix and right.supports_operator_matrix
        if adjoint is None:
            self._adjoint = _JoinOperator(self._right.T, self._left.T, self)
        else:
            self._adjoint = adjoint
        self._shape = (left.shape[0], right.shape[1])

    def _matmul_impl(self, y):
        return self._left @ (self._right @ y)

    def solve_A_x_eq_b(self, b, x0=None):
        if self.efficient_inverse:
            y = self._left.solve_A_x_eq_b(b, self._right @ x0)
            return self._right.solve_A_x_eq_b(y, x0)
        else:
            return super().solve_A_x_eq_b(b, x0=x0)

class VectorJacobianOperator(LinearOperator):
    def __init__(self, g, x, adjoint=None):
        self._shape = (x.shape[0], g.shape[0])
        self._adjoint = adjoint
        self.device = x.device
        self._g = g
        self._x = x

    def _matmul_impl(self, v):
        self._x.grad = None
        self._g.backward(gradient=v, retain_graph=True)
        out = self._x.grad
        self._x.grad = None
        return out

class SelectionOperator(LinearOperator):
    #supports_operator_matrix = True
    def __init__(self, shape, idxs):
        self._shape = shape
        self._adjoint = _AdjointSelectionOperator(idxs,
                (self._shape[1], self._shape[0]), self)
        self._idxs = idxs

    def _matmul_impl(self, X):
        return X[self._idxs]

    def solve_I_p_lambda_AT_A_x_eq_b(self, lambda_, b):
        LHS = torch.ones_like(b)
        LHS[self._idxs] += lambda_
        return b / LHS

class _AdjointSelectionOperator(LinearOperator):
    #supports_operator_matrix = True
    def __init__(self,  idxs, shape, adjoint):
        self._adjoint = adjoint
        self._idxs = idxs
        self._shape = shape

    def _matmul_impl(self, y):
        z = torch.zeros(self.shape[0], dtype=y.dtype, device=y.device)
        z[self._idxs] = y
        return z.reshape(-1)

def _generate_selector(size_of_vec, key, flip=False):
    shape_of_output = torch.empty(size_of_vec)[key].shape
    if len(shape_of_output) == 0:
        shape_of_output = 1
    else:
        shape_of_output = shape_of_output[0]
    return SelectionOperator(
            (size_of_vec, shape_of_output)
                if not flip else (shape_of_output, size_of_vec)
            , key)

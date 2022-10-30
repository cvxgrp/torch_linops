# An Abstract Linear Operator Library for pyTorch

This library implements a generic structure for abstract linear operators and
enables a number of standard operations on them:
 * Arithmetic: `A + B`, `A - B`, `-A`, `A @ B` all work exactly as expected to
    combine linear operators.
 * Indexing: `A[k:ell,m:n]` works as expected.
 * Solves: `Ax = b` can be solved with `CG` for PSD matrices, `minres` for
 symmetric matrices, `LSQR` (to be implemented), or `LSMR` (to be implemented).
 * Trace estimation: The trace of square matrices, can be estimated via Hutch++
    and Hutchinson's estimator.
 * [Diamond-Boyd stochastic equilibration](https://web.stanford.edu/~boyd/papers/mf_equil.html)
 * [Randomized Nyström Preconditioning](https://arxiv.org/abs/2110.02820)
 * Automatic adjoint operator generation.

# Using `LinearOperator`s

The public API of the `LinearOperator` library is that every `LinearOperator` has the
following properties and methods:
```python

class LinearOperator:

    # Properties
    shape: tuple[int, int]
    T: LinearOperator
    supports_operator_matrix: bool
    device: torch.Device

    # Matrix multiply
    def __matmul__(self, b: torch.Tensor) -> torch.Tensor: ...
    def __rmatmul__(self, b: torch.Tensor) -> torch.Tensor: ...

    def __matmul__(self, b: LinearOperator) -> LinearOperator: ...
    def __rmatmul__(self, b: LinearOperator) -> LinearOperator: ...
    
    # Linear Solve Methods
    def solve_I_p_lambda_AT_A_x_eq_b(self,
        lambda_: float,
        b: torch.Tensor,
        x0: torch.Tensor | None=None,
        *, precondition: None | Literal['nsytrom'], hot=False) -> torch.Tensor: ...

    def solve_A_x_eq_b(self,
        b: torch.Tensor,
        x0: torch.Tensor | None=None) -> torch.Tensor: ...

    # Transformations on LinearOperator
    def __mul__(self, c: float) -> LinearOperator: ...
    def __rmul__(self, c: float) -> LinearOperator: ...

    def __truediv__(self, c: float) -> LinearOperator: ...

    def __pow__(self, k: int) -> LinearOperator: ...

    def __add__(self, c: LinearOperator) -> LinearOperator: ...

    def __sub__(self, c: LinearOperator) -> LinearOperator: ...

    def __neg__(self) -> LinearOperator: ...

    def __pos__(self) -> LinearOperator: ...

    def __getitem__(self, key) -> LinearOperator: ...
```

The following functions are available in the root of the library:
```python
def operator_matrix_product(A: LinearOperator, M: torch.Tensor) -> torch.Tensor: ...
def aslinearoperator(A: torch.Tensor | LinearOperator) -> LinearOperator: ...
def vstack(ops: list[LinearOperator] | tuple[LinearOperator, ...]) -> LinearOperator: ...
def hstack(ops: list[LinearOperator] | tuple[LinearOperator, ...]) -> LinearOperator: ...

# To be implemented:
def bmat(ops: list[list[LinearOperator]]) -> LinearOperator: ... # Optimizes out ZeroOperator
```

The following functions are available in `linops.trace` for trace estimation:
```python
def hutchpp(A: lo.LinearOperator, m: int) -> float: ...
def hutchinson(A: lo.LinearOperator, m: int) -> float: ...
```

`linops.equilibration` contains `equilibrate` and `symmetric_equilibrate`.
Their public API is not finalized, if you wish to use them it is recommend you read the source code.

# Creating Linear Operators

Linear operators can be constructed in the following way:
 * Creating a sub-class of `LinearOperator` 
 * Calling one of the following constructors:
    * `IdentityOperator(n: int)`
    * `DiagonalOperator(diag: torch.Tensor)`: where `diag` is a 1D torch tensor.
    * `MatrixOperator(M: torch.Tensor)`: where `M` is a 2D torch tensor.
    * `SelectionOperator(shape: tuple[int, int], idxs: slice | list[int | slice])`
    * `KKTOperator(H: LinearOperator, A: LinearOperator)`: where `H` is a square `LinearOperator` and `A` is a `LinearOperator`
    * `VectorJacobianOperator(f: torch.Tensor, x: torch.Tensor)`: where `f` is the output of the function being differentiated
        which has a torch autograd value and `x` is the vector on which `ensures_grad` was called.
    * `ZeroOperator(shape: tuple[int, int])`
 * Combining operators via:
    * `A + B`, `A - B`, `A @ B` for `A`, `B` linear operators
    * `hstack`, `vstack`
    * `A`, `c A`, `A / c`, `v * A`, `A / v` for scalar `c` and vector `v`.

# Implementing `LinearOperator`s

To implement a `LinearOperator` the following are mandatory:

 * Set `_shape: tuple[int, int]`  to the shape of the operator.
 * Set `device` appropriately, if the operator requires vectors to be on a particular device.
 *  Implement a method `def _matmul_impl(self, v: torch.Tensor) -> torch.Tensor: ...` that implements your matrix vector product.

The following are recommended to improve performance:

 * If your `_matmul_impl` method handles matrix inputs correctly, set `supports_operator_matrix: bool` to `True`.
 * If it is possible to describe the adjoint operator, set `_adjoint: LinearOperator` to point to the adjoint of your operator. If you do not compute this, then one will be autogenerated by differentiating through your `_matmul_impl`.

It is suggested that, if possible, you replace any other methods with specialized implementations.

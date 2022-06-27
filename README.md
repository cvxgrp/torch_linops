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

Linear operators can be constructed in the following way:
 * Creating a sub-class of `LinearOperator` 
 * Calling one of the following constructors:
    * `IdentityOperator(n)`
    * `DiagonalOperator(diag)`: where `diag` is a 1D torch tensor.
    * `MatrixOperator(M)`: where `M` is a 2D torch tensor.
    * `SelectionOperator(shape, idxs)`
    * `KKTOperator(H, A)`: where `H` is a square `LinearOperator` and `A` is a `LinearOperator`
    * `VectorJacobianOperator(f, x)`: where `f` is the output of the function being differentiated
        which has a torch autograd value and `x` is the vector on which `ensures_grad` was called.
    * `ZeroOperator(shape)`
 * Combining operators via:
    * `A + B`, `A - B`, `A @ B` for `A`, `B` linear operators
    * `hstack`, `vstack`
    * `A`, `c A`, `A / c`, `v * A`, `A / v` for scalar `c` and vector `v`.

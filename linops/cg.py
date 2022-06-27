"""

Copyright 2021 Parth Nobel
Copyright 2019 Shane Barratt


Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""

import torch
import time

def cg_batch_Kis1(A_mm, B, M_mm=None, X0=None, rtol=1e-3, atol=0.,
                  maxiter=None, verbose=False):
    """Solves a batch of PD matrix linear systems using the preconditioned CG algorithm.
    This function solves a batch of matrix linear systems of the form
        A X = B
    where A is a n x n positive definite matrix and B is a n x m matrix,
    and X is the n x m matrix representing the solution for the ith system.
    Args:
        A_mm: A callable that performs a batch matrix multiply of A and a n x m matrix.
        B: A n x m matrix representing the right hand sides.
        M_mm: (optional) A callable that performs a batch matrix multiply of the preconditioning
            matrices M and a n x m matrix. (default=identity matrix)
        X0: (optional) Initial guess for X, defaults to M_bmm(B). (default=None)
        rtol: (optional) Relative tolerance for norm of residual. (default=1e-3)
        atol: (optional) Absolute tolerance for norm of residual. (default=0)
        maxiter: (optional) Maximum number of iterations to perform. (default=5*n)
        verbose: (optional) Whether or not to print status messages. (default=False)
    """
    n, = B.shape

    if M_mm is None:
        M_mm = lambda x: x
    if X0 is None:
        X0 = M_mm(B)
    if maxiter is None:
        maxiter = 5 * n

    assert B.shape == (n,)
    assert X0.shape == (n,)
    assert rtol > 0 or atol > 0
    assert isinstance(maxiter, int)

    X_k = X0
    R_k = B - A_mm(X_k)
    Z_k = M_mm(R_k)

    P_k = torch.zeros_like(Z_k)

    P_k1 = P_k
    R_k1 = R_k
    R_k2 = R_k
    X_k1 = X0
    Z_k1 = Z_k
    Z_k2 = Z_k

    B_norm = torch.norm(B)
    epsilon = torch.max(rtol*B_norm, atol*torch.ones_like(B_norm))

    if verbose:
        print("%03s | %010s %06s" % ("it", "dist", "it/s"))

    optimal = False
    start = time.perf_counter()
    for k in range(1, maxiter + 1):
        start_iter = time.perf_counter()
        Z_k = M_mm(R_k)

        if k == 1:
            P_k = Z_k
            R_k1 = R_k
            X_k1 = X_k
            Z_k1 = Z_k
        else:
            R_k2 = R_k1
            Z_k2 = Z_k1
            P_k1 = P_k
            R_k1 = R_k
            Z_k1 = Z_k
            X_k1 = X_k
            denominator = (R_k2 * Z_k2).sum()
            denominator[denominator == 0] = 1e-8
            beta = (R_k1 * Z_k1).sum() / denominator
            P_k = Z_k1 + beta * P_k1

        denominator = (P_k * A_mm(P_k)).sum()
        denominator[denominator == 0] = 1e-8
        alpha = (R_k1 * Z_k1).sum() / denominator
        X_k = X_k1 + alpha * P_k
        R_k = R_k1 - alpha * A_mm(P_k)
        end_iter = time.perf_counter()

        residual_norm = torch.norm(A_mm(X_k) - B)

        if verbose:
            print("%03d | %8.4e %8.4e %4.2f" %
                  (k, residual_norm, epsilon,
                    1. / (end_iter - start_iter)))

        if (residual_norm <= epsilon).all():
            optimal = True
            break

    end = time.perf_counter()

    if verbose:
        if optimal:
            print("Terminated in %d steps (optimal). Took %.3f ms." %
                  (k, (end - start) * 1000))
        else:
            print("Terminated in %d steps (reached maxiter). Took %.3f ms." %
                  (k, (end - start) * 1000))


    if verbose:
        info = {
            "niter": k,
            "optimal": optimal,
            "residual": residual_norm
        }
        print(info)

    return X_k

def CG(A_mm, M_mm=None, rtol=1e-3, atol=0., maxiter=None, X0=None,
            verbose=False):
    bX0 = None
    class CG(torch.autograd.Function):
        
        @staticmethod
        def forward(_ctx, B):
            X = cg_batch_Kis1(A_mm, B, M_mm=M_mm, X0=X0, rtol=rtol,
                        atol=atol, maxiter=maxiter, verbose=verbose)
            return X

        @staticmethod
        def backward(_ctx, dX):
            nonlocal bX0
            dB = cg_batch_Kis1(A_mm, dX, M_mm=M_mm, X0=bX0, rtol=rtol,
                        atol=atol, maxiter=maxiter, verbose=verbose)
            bX0 = dB
            return dB
    return CG.apply

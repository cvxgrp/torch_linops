import torch
import linops as lo

def minres(A, b, M=None, x0=None, tol=1e-5, maxiters=None, verbose=True):
    """
    Code based on scipy.sparse.linalg.minres

    Copyright 2022 Parth Nobel
    Copyright 2001-2002 Enthought, Inc. 2003-2022, SciPy Developers.

    All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions
    are met:

    1. Redistributions of source code must retain the above copyright
        notice, this list of conditions and the following disclaimer.

    2. Redistributions in binary form must reproduce the above
        copyright notice, this list of conditions and the following
        disclaimer in the documentation and/or other materials provided
        with the distribution.

    3. Neither the name of the copyright holder nor the names of its
        contributors may be used to endorse or promote products derived
        from this software without specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
    "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
    LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
    A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
    OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
    SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
    LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
    DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
    THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
    """
    m, n = A.shape
    assert m == n
    if M is None:
        M = lo.IdentityOperator(m)
    if maxiters is None:
        maxiters = 5 * n
    if x0 is None:
        x = torch.zeros_like(b)
    else:
        x = x0
    iters = 0
    # Stopping criterion only
    Anorm = 0 # scalar

    # Stopping criterion only
    Acond = 0 # scalar
    eps = torch.tensor(torch.finfo(b.dtype).eps, device=x.device) # scalar

    r1 = b - A @ x # Supports block
    y = M @ r1 # Supports block

    #beta1 = r1 @ y # Needs modification
    beta1 = inner(r1, y)
    assert beta1.min() >= 0, "M must be PD"
    if (beta1 == 0).all():
        return x
    bnorm = torch.linalg.vector_norm(b)
    if bnorm == 0:
        return b

    beta1 = torch.sqrt(beta1) # Supports block

    oldb = torch.zeros_like(beta1) # Supports block
    beta = beta1 # Supports block
    dbar = torch.zeros_like(beta1) # Supports block
    epsilon = torch.zeros_like(beta1) # Supports block
    phibar = beta1 # Supports block
    rhs1 = beta1 # Supports block
    rhs2 = torch.zeros_like(beta1) # Supports block

    # Stopping criterion only
    tnorm2 = torch.zeros_like(beta1)
    # Only for illconditioning detection
    gmax = torch.tensor(0.0, device=x.device)
    # Only for illconditioning detection
    gmin = torch.tensor(torch.finfo(x.dtype).max, device=x.device)

    cs = -torch.ones_like(beta1)
    sn = torch.zeros_like(beta1) # Supports block
    w = torch.zeros_like(b) # Supports block
    w2 = torch.zeros_like(b) # Supports block
    r2 = r1 # Supports block
    
    while iters < maxiters:
        iters += 1
        s = 1.0 / beta # Supports block
        v = s * y # Supports block
        y = A @ v # Supports block
        if iters >= 2:
            y = y - (beta / oldb) * r1

        alpha = inner(v, y)
        y = y - (alpha / beta) * r2
        r1 = r2
        r2 = y
        y = M @ r2
        oldb = beta
        beta = inner(r2, y)
        assert (beta >= 0).all()
        beta = torch.sqrt(beta)
        tnorm2 += alpha**2 + oldb**2 + beta**2
        if iters == 1:
            if (beta / beta1).min() <= 10 * eps:
                #assert False, "I think this occurs when A = c * I"
                pass

        oldeps = epsilon
        delta = cs * dbar + sn * alpha
        gbar = sn * dbar - cs * alpha
        epsilon = sn * beta
        dbar = -cs * beta
        root = L2norm2entry(gbar, dbar)
        #Arnorm = phibar * root

        gamma = L2norm2entry(gbar, beta)
        gamma = torch.maximum(gamma, eps)
        cs = gbar / gamma
        sn = beta / gamma
        phi = cs * phibar
        phibar = sn * phibar


        denom = 1.0 / gamma
        w1 = w2
        w2 = w
        w = (v - oldeps * w1 - delta * w2) * denom
        x = x + phi * w

        gmax = torch.max(gmax, gamma).max()
        gmin = torch.min(gmin, gamma).min()
        z = rhs1 / gamma
        rhs1 = rhs2 - delta * z
        rhs2 = - epsilon * z

        Anorm = torch.sqrt(tnorm2.max())
        ynorm = torch.linalg.norm(x)
        epsa = Anorm * eps
        epsx = Anorm * ynorm * eps
        #epsr = Anorm * ynorm * tol
        diag = gbar
        if (diag == 0).any():
            diag = epsa
        qrnorm = phibar
        rnorm = qrnorm
        if ynorm == 0 or Anorm == 0:
            test1 = torch.inf
        else:
            test1 = rnorm / (Anorm * ynorm)

        if Anorm == 0:
            test2 = torch.inf
        else:
            test2 = root / Anorm
        Acond = gmax / gmin
        t1 = 1 + test1.max()
        t2 = 1 + test2.max()
        if t2 <= 1:
            break
        if t1 <= 1:
            break

        if Acond >= 0.1 / eps:
            assert False, "System is ill-conditioned."
        if (epsx >= beta1).any():
            assert False
        if test2.max() <= tol:
            break
        if test1.max() <= tol:
            break
    return x

def L2norm2entry(x, y):
    return torch.sqrt(x**2 + y**2)

def inner(x, y):
    return (x * y).sum(axis=0) # Supports block

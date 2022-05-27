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
    Anorm = 0
    Acond = 0
    eps = torch.tensor(torch.finfo(b.dtype).eps, device=x.device)

    r1 = b - A @ x
    y = M @ r1

    beta1 = r1 @ y
    assert beta1 >= 0
    if beta1 == 0:
        return x
    bnorm = torch.linalg.vector_norm(b)
    if bnorm == 0:
        return b

    beta1 = torch.sqrt(beta1)

    oldb = 0
    beta = beta1
    dbar = 0
    epsilon = 0
    phibar = beta1
    rhs1 = beta1
    rhs2 = 0
    tnorm2 = torch.tensor(0.0, device=x.device)
    gmax = torch.tensor(0.0, device=x.device)
    gmin = torch.tensor(torch.finfo(x.dtype).max, device=x.device)
    cs = -1
    sn = 0
    w = torch.zeros_like(b)
    w2 = torch.zeros_like(b)
    r2 = r1
    
    while iters < maxiters:
        iters += 1
        s = 1.0 / beta
        v = s * y
        y = A @ v
        if iters >= 2:
            y = y - (beta / oldb) * r1

        alpha = v @ y
        y = y - (alpha / beta) * r2
        r1 = r2
        r2 = y
        y = M @ r2
        oldb = beta
        beta = r2 @ y
        assert beta >= 0
        beta = torch.sqrt(beta)
        tnorm2 += alpha**2 + oldb**2 + beta**2
        if iters == 1:
            if beta / beta1 <= 10 * eps:
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

        gmax = torch.max(gmax, gamma)
        gmin = torch.min(gmin, gamma)
        z = rhs1 / gamma
        rhs1 = rhs2 - delta * z
        rhs2 = - epsilon * z

        Anorm = torch.sqrt(tnorm2)
        ynorm = torch.linalg.norm(x)
        epsa = Anorm * eps
        epsx = Anorm * ynorm * eps
        #epsr = Anorm * ynorm * tol
        diag = gbar
        if diag == 0:
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
        t1 = 1 + test1
        t2 = 1 + test2
        if t2 <= 1:
            break
        if t1 <= 1:
            break

        if Acond >= 0.1 / eps:
            assert False, "System is ill-conditioned."
        if epsx >= beta1:
            assert False
        if test2 <= tol:
            break
        if test1 <= tol:
            break
    return x

def L2norm2entry(x, y):
    return torch.sqrt(x**2 + y**2)

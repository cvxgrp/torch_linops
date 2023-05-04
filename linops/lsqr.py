""" 
Code based on scipy.sparse.linalg.lsqr

The original Fortran code was written by C. C. Paige and M. A. Saunders as
described in
C. C. Paige and M. A. Saunders, LSQR: An algorithm for sparse linear
equations and sparse least squares, TOMS 8(1), 43--71 (1982).
C. C. Paige and M. A. Saunders, Algorithm 583; LSQR: Sparse linear
equations and least-squares problems, TOMS 8(2), 195--209 (1982).
It is licensed under the following BSD license:
Copyright (c) 2006, Systems Optimization Laboratory

All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:
1.  Redistributions of source code must retain the above copyright
    notice, this list of conditions and the following disclaimer.

2.  Redistributions in binary form must reproduce the above
    copyright notice, this list of conditions and the following
    disclaimer in the documentation and/or other materials provided
    with the distribution.

3   Neither the name of Stanford University nor the names of its
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
The Fortran code was translated to Python for use in CVXOPT by Jeffery
Kline with contributions by Mridul Aanjaneya and Bob Myhill.
Adapted for SciPy by Stefan van der Walt.
"""

import torch

eps = torch.finfo(torch.float64).eps

def _sym_ortho(a, b):
    if b == 0:
        return torch.sign(a), 0, abs(a)
    elif a == 0:
        return 0, torch.sign(b), abs(b)
    elif abs(b) > abs(a):
        tau = a / b
        s = torch.sign(b) / torch.sqrt(1 + tau * tau)
        c = s * tau
        r = b / s
    else:
        tau = b / a 
        c = torch.sign(a) / torch.sqrt(1+tau*tau)
        s = c * tau
        r = a / c
    return c, s, r

def lsqr(A, b, damp=0.0, atol=1e-6, btol=1e-6, conlim=1e8,
         iter_lim=None, calc_var=False, x0=None):

        m, n = A.shape
        if iter_lim is None:
            iter_lim = 2 * n
        var = torch.zeros(n, device=b.device, dtype=b.dtype)

        itn = 0
        istop = 0
        ctol = 0
        if conlim > 0:
            ctol = 1/conlim
        anorm = 0
        acond = 0
        dampsq = damp**2
        ddnorm = 0
        res2 = 0
        xnorm = 0
        xxnorm = 0
        z = 0
        cs2 = -1
        sn2 = 0
        u = b
        bnorm = torch.linalg.norm(b)

        if x0 is None:
            x = torch.zeros(n, device=b.device, dtype=b.dtype)
            beta = bnorm.clone()
        else:
            x = x0
            u = u - A@x
            beta = torch.linalg.norm(u)

        if beta > 0:
            u = (1/beta) * u
            v = A.T@u
            alfa = torch.linalg.norm(v)
        else:
            v = x.clone()
            alfa = 0
        
        if alfa > 0:
            v = (1/alfa) * v
        w = v.clone()

        rhobar = alfa
        phibar = beta
        rnorm = beta
        r1norm = rnorm
        r2norm = rnorm

        arnorm = alfa * beta
        if arnorm == 0:
            return x

        # Main iteration loop
        while itn < iter_lim:
            itn = itn + 1
            u = A@v - alfa * u
            beta = torch.linalg.norm(u)

            if beta > 0:
                u = (1/beta)*u
                anorm = torch.sqrt(anorm**2 + alfa**2 + beta**2 + dampsq)
                v = A.T@u - beta * v
                alfa = torch.linalg.norm(v)
                if alfa > 0:
                    v = (1 / alfa * v)

            if damp > 0:
                rhobar1 = torch.sqrt(rhobar**2 + dampsq)
                cs1 = rhobar / rhobar1
                sn1 = damp / rhobar1
                psi = sn1 * phibar
                phibar = cs1 * phibar
            else:
                rhobar1 = rhobar
                psi = 0
            cs, sn, rho = _sym_ortho(rhobar1, beta)

            theta = sn * alfa
            rhobar = -cs * alfa
            phi = cs * phibar
            phibar = sn * phibar
            tau = sn * phi

            t1 = phi / rho
            t2 = -theta / rho
            dk = (1/rho) * w

            x = x + t1 * w
            w = v + t2 * w
            ddnorm = ddnorm + torch.linalg.norm(dk)**2

            if calc_var:
                var = var + dk**2

            delta = sn2 * rho
            gambar = -cs2 * rho
            rhs = phi - delta * z
            zbar = rhs / gambar
            xnorm = torch.sqrt(xxnorm + zbar**2)
            gamma = torch.sqrt(gambar**2 +theta**2)
            cs2 = gambar / gamma
            sn2 = theta / gamma
            z = rhs / gamma
            xxnorm = xxnorm + z**2

            acond = anorm * torch.sqrt(ddnorm)
            res1 = phibar**2
            res2 = res2 + psi**2
            rnorm = torch.sqrt(res1 + res2)
            arnorm = alfa * torch.abs(tau)

            if damp > 0:
                r1sq = rnorm**2 - dampsq * xxnorm
                r1norm = torch.sqrt(torch.abs(r1sq))
                if r1sq < 0:
                    r1norm = -r1norm
            else:
                r1norm = rnorm
            r2norm = rnorm

            test1 = rnorm / bnorm
            test2 = arnorm / (anorm * rnorm + eps)
            test3 = 1 / (acond + eps)
            t1 = test1 / (1 + anorm * xnorm/bnorm)
            rtol = btol + atol * anorm / bnorm

            if itn >= iter_lim:
                istop = 7
            if 1 + test3 <= 1:
                istop = 6
            if 1 + test2 <= 1:
                istop = 5
            if 1 + t1 <= 1:
                istop = 4

            if test3 <= ctol:
                istop = 3
            if test2 <= atol:
                istop = 2
            if test1 <= rtol:
                istop = 1

            if istop != 0:
                break
    
        return x

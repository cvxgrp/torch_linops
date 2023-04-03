import torch

import linops as lo
import linops.trace as lot

diag_big = 1.0 / (torch.arange(3_600_000)+ 1)
A_big = lo.DiagonalOperator(diag_big)
eps_big = 1e-2

diag_small = 1.0 / (torch.arange(360)+ 1)
A_small = lo.DiagonalOperator(diag_small)
eps_small = 1e-2

def run_test_on(f, big=True):
    if big:
        diag = diag_big
        A = A_big
        eps = eps_big
    else:
        diag = diag_small
        A = A_small
        eps = eps_small
    trace, err = f(A)
    true_trace = diag.sum()
    assert torch.abs(trace - true_trace) < eps * true_trace
    assert torch.abs(trace - true_trace) < 2.5 * (err + 1e-6 * true_trace)

def test_hutchinson():
    run_test_on(lot.hutchinson)

def test_hutchpp():
    run_test_on(lot.hutchpp)

def test_xtrace():
    run_test_on(lot.xtrace)

def test_xnystrace():
    run_test_on(lot.xnystrace)

def test_exact():
    run_test_on(lot.exact_trace, False)

if __name__ == "__main__":
    test_exact()

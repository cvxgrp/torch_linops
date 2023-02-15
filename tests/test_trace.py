import torch

import linops as lo
import linops.trace as lot

diag = 1.0 / (torch.arange(3_600_000)+ 1)
A = lo.DiagonalOperator(diag)
eps = 1e-2

def run_test_on(f):
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

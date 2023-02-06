import torch

import linops as lo
import linops.trace as lot

import pytest
diag = 1.0 / (torch.arange(3_600_000)+ 1)
A = lo.DiagonalOperator(diag)
eps = 1e-2

def test_hutchinson():

    assert torch.abs(lot.hutchinson(A) - diag.sum()) < eps * diag.sum()

def test_hutchpp():
    assert torch.abs(lot.hutchpp(A) - diag.sum()) < eps * diag.sum()

def test_xtrace():
    assert torch.abs(lot.xtrace(A)[0] - diag.sum()) < eps * diag.sum()

def test_xnystrace():
    assert torch.abs(lot.xnystrace(A)[0] - diag.sum()) < eps * diag.sum()

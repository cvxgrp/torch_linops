import torch

import linops as lo
import linops.diag as lod

import pytest
diag = 1.0 / (torch.arange(3_600_000)+ 1)
A = lo.DiagonalOperator(diag)
eps = 1e-2

def test_xtrace():
    assert torch.linalg.vector_norm(lod.xdiag(A, 300) - diag) < eps * torch.linalg.vector_norm(diag)

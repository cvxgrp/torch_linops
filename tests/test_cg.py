import torch

import linops as lo
from linops.cg import CG

import pytest

def test_rename_one():
    A = lo.IdentityOperator(3)

    assert (torch.linalg.matrix_rank(lo.operator_matrix_product(A, torch.eye(3)))) == 3

    cg = CG(A)
    b = torch.tensor([1., 4. , 6.])
    x = cg(b)

    assert torch.linalg.norm(b - A @ x) <= 1e-3



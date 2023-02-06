import torch
import linops as lo
import linops.minres
import scipy.sparse as sp
import scipy.sparse.linalg

def to_implement_test():
    A_tensor = torch.Tensor(
                [[3., 1.],
                [1., 3.]]
            )
    A = lo.MatrixOperator(
            A_tensor
    )
    b = torch.Tensor([5., 7.])
    y_lo = lo.minres.minres(A, b)
    
    A = A_tensor.numpy()
    b = b.numpy()
    y_sp = sp.linalg.minres(A, b)
    print(y_lo)
    print(y_sp)

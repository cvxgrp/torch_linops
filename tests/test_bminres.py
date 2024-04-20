import torch
import linops as lo
import linops.block_minres
import scipy.sparse as sp
import scipy.sparse.linalg

def to_implement_test():
    A_tensor = torch.Tensor(
                [[3., 0., 0],
                [0., 80., 0],
                [0., 0, -7.]]
            )
    A = lo.MatrixOperator(
            A_tensor
    )
    b = torch.Tensor([[5.], [7.], [2.]])
    y_lo = lo.block_minres.bminres(A, b)

    A = A_tensor.numpy()
    b = b.numpy()
    y_sp = sp.linalg.minres(A, b, show=True, maxiter=10)
    print(y_lo)
    print(y_sp)

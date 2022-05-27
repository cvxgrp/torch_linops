import torch
import linops as lo
import linops.minres
import scipy.sparse as sp
import scipy.sparse.linalg
A_tensor = torch.Tensor(
            [[3., 1.],
             [1., 3.]]
        )
A = lo.MatrixOperator(
        A_tensor
)
b = torch.Tensor([5., 7.])
breakpoint()
y_lo = lo.minres.minres(A, b)

A = A_tensor.numpy()
b = b.numpy()
breakpoint()
y_sp = sp.linalg.minres(A, b)
print(y_lo)
print(y_sp)

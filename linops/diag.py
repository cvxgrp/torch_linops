import torch
import linops

def xdiag(A, m: int):
    n, n1 = A.shape
    assert n == n1
    m = m // 2
    def normalize_columns(M):
        return M / torch.linalg.vector_norm(M, dim=0)

    def diag_of_AB(A, B):
        return torch.sum(A * B, dim=0)

    Omega = (2 * torch.randint(2, size=(n, m), device=A.device) - 1).float()
    Y = linops.operator_matrix_product(A, Omega)
    Q, R = torch.linalg.qr(Y)

    Z = linops.operator_matrix_product(A.T, Q)
    T = Z.T @ Omega
    S = normalize_columns(torch.linalg.inv(R).T)
    dQZ = diag_of_AB(Q.T, Z.T)
    dQSSZ = diag_of_AB((Q @ S).T, (Z @ S).T)
    dOmegaQT = diag_of_AB(Omega.T, (Q @ T).T)
    dOmegaY = diag_of_AB(Omega.T, Y.T)
    dOmegaQSST = diag_of_AB(Omega.T, 
                            (Q @ S @ torch.diag(diag_of_AB(S, T))).T)

    return dQZ + (-dQSSZ + dOmegaY -dOmegaQT +dOmegaQSST) / m

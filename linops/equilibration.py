import torch

def equilibrate(
        A,
        alpha_squared=1,
        beta_squared=1,
        gamma=1.0,
        M=10,
        iterations=20,
):
    """
        Algorithm taken from:

        S. Diamond, S. Boyd. Stochastic Matrix-Free Equilibration,
            *Journal of Optimization Theory and Applications*,
            172(2), 436-454, 2017.

        All constants are random.
    """
    n, m = A.shape
    u_k = torch.zeros(n)
    v_k = torch.zeros(m)
    u_bar = torch.zeros(n)
    v_bar = torch.zeros(m)

    for t in range(1, iterations):
        D = torch.exp(u_k)
        E = torch.exp(v_k)
        s = torch.randint(-1, 2, size=n)
        w = torch.randint(-1, 2, size=m)

        u: torch.Tensor = u_k - 2 * (
                        torch.abs(D * (A @ (E * s)))**2 - alpha_squared + gamma * u_k
                    ) / (gamma * (t + 1))
        u_k = torch.clip(u, -M, M)

        # Note A = A.T
        v: torch.Tensor = v_k - 2 * (
                        torch.abs(E * (A @ (D * w)))**2 - beta_squared + gamma * v_k
                    ) / (gamma * (t + 1))
        v_k = torch.clip(v, -M, M)
        u_bar = (2 / (t + 2)) * u_k + (t / (t+2)) * u_bar
        v_bar = (2 / (t + 2)) * v_k + (t / (t+2)) * v_bar
            
    D = torch.exp(u_bar)
    E = torch.exp(v_bar)
    return D, E

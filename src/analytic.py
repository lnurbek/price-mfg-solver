import torch

def compute_analytic_solution(Q, x0, T, c0, r1, r2, y1, y2):
    """
    Compute the analytic price and trajectories

    Args:
        Q: tensor of shape (N,)
        x0: tensor of shape (M,)
        T: time horizon
        c0, r1, r2, y1, y2: scalars
    Returns:
        omega_star: tensor of shape (N,)
        B: scalar
        z: (M, N+1) tensor of analytic trajectories
    """

    N = Q.shape[0]
    dt = T / N
    x_bar = x0.mean()

    # Compute the analytic price omega*

    omega_star = torch.zeros_like(Q, dtype=Q.dtype)

    for i in range(N):
        term1 = r2 * (y2 - x_bar)
        term2 = r1 * (T - i * dt) * (y1 - x_bar)
        term3 = -c0 * Q[i]

        # Sum over j
        sum_term = 0.0
        for j in range(N):
            max_t = max(j * dt, i * dt)
            coeff = - (r2 + r1 * T) + r1 * max_t
            sum_term += coeff * Q[j]
        sum_term = sum_term / N

        omega_star[i] = term1 + term2 + term3 + sum_term


    # Compute the auxiliary constant B

    k = (r1 / c0) ** 0.5

    time_grid = torch.arange(N, dtype=omega_star.dtype) * dt  # [0, dt, 2dt, ..., (N-1)dt]
    weights = (r2 / c0) * torch.cosh(k * (T - time_grid)) + k * torch.sinh(k * (T - time_grid))
    B = r2 * (y2 - y1) + (1 / N) * torch.sum(omega_star * weights)
    B = B.item()  # Convert to scalar
    
    # Compute the trajectories z

    M = x0.shape[0]

    z = torch.zeros(M, N + 1, dtype=x0.dtype)
    z[:, 0] = x0  # initial condition

    # Case 1: r1 == 0
    if r1 == 0.0:
        for i in range(1, N + 1):
            t_i = i * dt
            # scalar coefficient
            drift = (B - r2 * (x0 - y1)) / (c0 + r2 * T) * t_i  # shape (M,)
            control_sum = torch.cumsum(omega_star, dim=0)
            z[:, i] = x0 + drift - (1 / (c0 * N)) * control_sum[i - 1]  # scalar value

    # Case 2: r1 ≠ 0
    else:
        kT_tensor = torch.tensor(k * T, dtype=x0.dtype)
        cosh_kT = torch.cosh(kT_tensor)
        sinh_kT = torch.sinh(kT_tensor)
        denom = c0 * k * cosh_kT + r2 * sinh_kT

        for i in range(1, N + 1):
            t_i = i * dt
            cosh_kti = torch.cosh(torch.tensor(k * t_i, dtype=x0.dtype))
            sinh_kti = torch.sinh(torch.tensor(k * t_i, dtype=x0.dtype))


            drift = (
                y1 +
                (x0 - y1) * cosh_kti +
                (B - (x0 - y1) * (c0 * k * sinh_kT + r2 * cosh_kT)) / denom * sinh_kti
            )  # shape (M,)

            # control sum: sum_{j=0}^{i-1} omega_j * cosh(k (i-j) dt)
            indices = torch.arange(i, dtype=x0.dtype)  # 0 to i-1
            delta_t = (i - indices) * dt  # shape (i,)
            weights = torch.cosh(k * delta_t)  # shape (i,)
            control_sum = torch.matmul(omega_star[:i], weights)  # scalar
            z[:, i] = drift - (1 / (c0 * N)) * control_sum

    return omega_star, z
import torch

def compute_trajectories(alpha: torch.Tensor, x0: torch.Tensor, dt: float) -> torch.Tensor:
    """
    Compute trajectories z_x from controls alpha_x using forward Euler (vectorized).

    Args:
        alpha: Tensor of shape (M, N), control values
        x0: Tensor of shape (M,), initial positions
        dt: float, time step size

    Returns:
        z: Tensor of shape (M, N+1), trajectories
    """
    z_increments = dt * alpha
    z_relative = torch.cumsum(z_increments, dim=1)
    z_full = torch.cat([x0.unsqueeze(1), x0.unsqueeze(1) + z_relative], dim=1)
    return z_full

def compute_objective(
    alpha: torch.Tensor,
    omega: torch.Tensor,
    x0: torch.Tensor,
    dt: float,
    sigma: float,
    Q: torch.Tensor,
    L: callable,
    g: callable
) -> torch.Tensor:
    """
    Compute the full discretized objective function.

    Args:
        alpha: (M, N) control for each sample
        omega: (N,) price function
        x0: (M,) initial sample positions
        dt: timestep size
        sigma: regularization weight
        Q: (N,) supply function
        L: function L(z, a) → shape (M, N)
        g: function g(z_T) → shape (M,)

    Returns:
        Scalar objective (torch.Tensor)
    """
    # First term: regularization and supply
    obj1 = dt * torch.sum(0.5 * sigma * omega**2 + omega * Q)

    # Second term: agent trajectories + expected cost
    z = compute_trajectories(alpha, x0, dt)          # (M, N+1)
    L_term = L(z[:, :-1], alpha)                     # (M, N)
    coupling = alpha * omega                         # (M, N), broadcasts
    integrand = L_term + coupling
    integral = dt * torch.sum(integrand, dim=1)      # (M,)
    terminal = g(z[:, -1])                           # (M,)
    obj2 = torch.mean(integral + terminal)

    return obj1 - obj2

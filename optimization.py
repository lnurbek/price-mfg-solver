import torch
from objective import compute_objective

def run_optimization(
    alpha_init: torch.Tensor,
    omega_init: torch.Tensor,
    x0: torch.Tensor,
    dt: float,
    sigma: float,
    Q: torch.Tensor,
    L: callable,
    g: callable,
    tau_alpha: float,
    tau_omega: float,
    num_iters: int
):
    M, _ = alpha_init.shape

    # Detach and prepare starting points
    alpha_prev = alpha_init.detach().clone()
    omega = omega_init.detach().clone()

    for _ in range(num_iters):
        # Step 1: Create a fresh alpha tensor with gradient tracking
        alpha = alpha_prev.clone().detach().requires_grad_(True)

        # Step 2: Compute objective and gradient
        loss = compute_objective(alpha, omega, x0, dt, sigma, Q, L, g)
        loss.backward()
        grad_alpha = alpha.grad.detach()

        # Step 3: Primal update
        alpha_new = alpha.detach() + (tau_alpha * M / dt) * grad_alpha

        # Step 4: Extrapolation
        alpha_bar = 2 * alpha_new - alpha_prev

        # Step 5: Dual update
        omega = (1 / (tau_omega * sigma + 1)) * omega + \
                (tau_omega / (tau_omega * sigma + 1)) * (alpha_bar.mean(dim=0) - Q)

        # Step 6: Prepare for next iteration
        alpha_prev = alpha_new

    return alpha_prev, omega

import torch
from objective import compute_objective

# def run_optimization(
#     alpha_init: torch.Tensor,
#     omega_init: torch.Tensor,
#     x0: torch.Tensor,
#     dt: float,
#     sigma: float,
#     Q: torch.Tensor,
#     L: callable,
#     g: callable,
#     tau_alpha: float,
#     tau_omega: float,
#     num_iters: int
# ):
#     """
#     Run primal-dual optimization on the MFG objective.

#     Args:
#         alpha_init: (M, N) initial controls
#         omega_init: (N,) initial price trajectory
#         x0: (M,) initial positions of samples
#         dt: time step size
#         sigma: regularization parameter in the objective
#         Q: (N,) supply vector
#         L: function L(z, a) → (M, N)
#         g: function g(z_T) → (M,)
#         tau_alpha: primal step size
#         tau_omega: dual step size
#         num_iters: number of iterations

#     Returns:
#         final_alpha, final_omega, alpha_history, omega_history, objective_history
#     """

#     # Step 1.1: Clone initial values to avoid modifying inputs
#     M, N = alpha_init.shape

#     alpha = alpha_init.clone().detach().requires_grad_(True)  # primal variable
#     omega = omega_init.clone().detach()                       # dual variable

#     for _ in range(num_iters):
#         # Step 2.1: Compute objective (creates computation graph)
#         loss = compute_objective(alpha, omega, x0, dt, sigma, Q, L, g)

#         # Step 2.2: Compute gradient w.r.t. alpha
#         # loss.backward()
#         loss.backward(retain_graph=True)
#         grad_alpha = alpha.grad.detach()

#         # Step 2.3: Gradient ascent step for alpha
#         alpha_new = alpha + (tau_alpha * M / dt) * grad_alpha  # shape (M, N)

#         # Step 2.4: Extrapolation step
#         alpha_bar = 2 * alpha_new - alpha

#         # Step 2.5: Omega update (dual variable update)
#         omega = (1 / (tau_omega * sigma + 1)) * omega + \
#                 (tau_omega / (tau_omega * sigma + 1)) * (alpha_bar.mean(dim=0) - Q)

#         # Step 2.6: Reinitialize alpha for next iteration
#         alpha = alpha_new.detach().clone().requires_grad_(True)

#     return alpha.detach(), omega

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
    M, N = alpha_init.shape

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

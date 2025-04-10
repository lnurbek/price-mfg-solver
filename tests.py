import torch
from objective import compute_objective, compute_trajectories

# Set float64 for accuracy
dtype = torch.float64
torch.set_default_dtype(dtype)

# Dummy test cost functions
def L(z, a):
    return 0.5 * a**2 + 0.1 * z**2

def g(z_T):
    return 0.5 * z_T**2

# Parameters
M, N = 5, 10
T = 1.0
dt = T / N
sigma = 0.01

# Random inputs
torch.manual_seed(0)
alpha = torch.randn(M, N, dtype=dtype, requires_grad=True)
omega = torch.sin(torch.linspace(0, T, N))
Q = torch.cos(torch.linspace(0, T, N))
x0 = torch.linspace(-1, 1, M)

# Compute objective
obj = compute_objective(alpha, omega, x0, dt, sigma, Q, L, g)
print("Objective value:", obj.item())

# Compute gradient w.r.t. alpha
obj.backward()
print("Gradient shape:", alpha.grad.shape)
print("Gradient (alpha):\n", alpha.grad)

def finite_difference_check_alpha(
    obj_fn, alpha, omega, x0, dt, sigma, Q, L, g, eps=1e-6
):
    alpha = alpha.detach().clone().requires_grad_(True)
    obj = obj_fn(alpha, omega, x0, dt, sigma, Q, L, g)
    obj.backward()
    grad_autograd = alpha.grad.detach().clone()

    M, N = alpha.shape
    grad_fd = torch.zeros_like(alpha)

    for i in range(M):
        for j in range(N):
            alpha_plus = alpha.detach().clone()
            alpha_plus[i, j] += eps
            obj_plus = obj_fn(alpha_plus, omega, x0, dt, sigma, Q, L, g)

            alpha_minus = alpha.detach().clone()
            alpha_minus[i, j] -= eps
            obj_minus = obj_fn(alpha_minus, omega, x0, dt, sigma, Q, L, g)

            grad_fd[i, j] = (obj_plus - obj_minus) / (2 * eps)

    max_diff = torch.max(torch.abs(grad_autograd - grad_fd)).item()
    print(f"Max difference (autograd vs finite diff): {max_diff:.2e}")
    return grad_autograd, grad_fd

if __name__ == "__main__":
    ...

    grad_autograd, grad_fd = finite_difference_check_alpha(
        compute_objective, alpha, omega, x0, dt, sigma, Q, L, g
    )

    print("Autograd gradient:\n", grad_autograd)
    print("Finite difference gradient:\n", grad_fd)
    print("Absolute diff:\n", torch.abs(grad_autograd - grad_fd))

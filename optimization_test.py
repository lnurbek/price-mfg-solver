import torch
from optimization import run_optimization
from objective import compute_objective, compute_trajectories


# Use float64 for accuracy in testing
torch.set_default_dtype(torch.float64)
torch.manual_seed(0)

# Cost function parameters
c0 = 1.0    # control cost weight
r1 = 0.0    # running potential strength
r2 = 100.0    # terminal cost strength
y1 = 0    # running potential center
y2 = 0    # terminal cost target


# Define test cost functions
def L(z, a):
    return 0.5 * c0 * a**2 + 0.5 * r1 * (z - y1)**2

def g(z_T):
    return 0.5 * r2 * (z_T - y2)**2


# Problem setup
M, N = 10, 50
T = 1.0
dt = T / N
sigma = 0.0
tau_alpha = 1e-2
tau_omega = 1e-1
num_iters = 10000

# Initial data
x0 = torch.linspace(-1, 1, M)
Q = torch.sin(10*torch.linspace(0, T, N))

alpha0 = torch.randn(M, N)
omega0 = torch.zeros(N)

import time
import psutil
import os

def print_memory_usage():
    mem_MB = psutil.Process(os.getpid()).memory_info().rss / 1024**2
    print(f"Memory usage: {mem_MB:.2f} MB")

print_memory_usage()
start_time = time.time()

# Run optimization
alpha_final, omega_final = run_optimization(
    alpha_init=alpha0,
    omega_init=omega0,
    x0=x0,
    dt=dt,
    sigma=sigma,
    Q=Q,
    L=L,
    g=g,
    tau_alpha=tau_alpha,
    tau_omega=tau_omega,
    num_iters=num_iters
)

# Evaluate final objective and gradient w.r.t. alpha
alpha_final = alpha_final.detach().clone().requires_grad_(True)
# Evaluate final objective and gradient w.r.t. omega
omega_final = omega_final.detach().clone().requires_grad_(True)

final_obj = compute_objective(alpha_final, omega_final, x0, dt, sigma, Q, L, g)
final_obj.backward()
grad_alpha = alpha_final.grad
grad_omega = omega_final.grad

# Print results
print(f"Final objective value: {final_obj.item():.6f}")
print(f"Gradient norm (alpha): {torch.norm(grad_alpha):.6e}")
print(f"Gradient norm (omega): {torch.norm(grad_omega):.6e}")

end_time = time.time()
print_memory_usage()
print(f"Optimization took {end_time - start_time:.3f} seconds")

def compute_analytic_omega(Q, x0, T, c0, r1, r2, y1, y2):
    """
    Compute the analytic solution omega* based on the formula.

    Args:
        Q: tensor of shape (N,)
        x0: tensor of shape (M,)
        T: time horizon
        c0, r1, r2, y1, y2: scalars

    Returns:
        omega_star: tensor of shape (N,)
    """
    N = Q.shape[0]
    dt = T / N
    x_bar = x0.mean()

    omega_star = torch.zeros(N, dtype=Q.dtype)

    for i in range(N):
        t_i = i * dt

        term1 = r2 * (y2 - x_bar)
        term2 = r1 * (T - t_i) * (y1 - x_bar)
        term3 = -c0 * Q[i]

        # Sum over j
        sum_term = 0.0
        for j in range(N):
            t_j = j * dt
            max_t = max(t_j, t_i)
            coeff = - (r2 + r1 * T) + r1 * max_t
            sum_term += coeff * Q[j]
        sum_term = sum_term / N

        omega_star[i] = term1 + term2 + term3 + sum_term

    return omega_star

omega_analytic = compute_analytic_omega(Q, x0, T, c0, r1, r2, y1, y2)

print("‖omega_numeric - omega_analytic‖_inf:",
      torch.max(torch.abs(omega_final - omega_analytic)).item())

import matplotlib.pyplot as plt

# Time discretization points (left endpoints)
t_grid = torch.linspace(0, T * (N - 1) / N, N)

plt.figure(figsize=(8, 5))
plt.plot(t_grid, omega_final.detach(), label='Numerical ω', linewidth=2)
plt.plot(t_grid, omega_analytic, label='Analytic ω*', linestyle='--', linewidth=2)
plt.xlabel("Time")
plt.ylabel("Price ω")
plt.title("Comparison of Numerical vs Analytic ω")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

def compute_B(omega_star, T, c0, r1, r2, y1, y2):
    """
    Compute the constant B used in the analytic trajectory formula.

    Args:
        omega_star: tensor of shape (N,) – analytic omega
        T: scalar time horizon
        c0, r1, r2, y1, y2: scalars

    Returns:
        B: scalar
    """
    N = omega_star.shape[0]
    dt = T / N

    k = (r1 / c0) ** 0.5

    time_grid = torch.arange(N, dtype=omega_star.dtype) * dt  # [0, dt, 2dt, ..., (N-1)dt]
    weights = (r2 / c0) * torch.cosh(k * (T - time_grid)) + k * torch.sinh(k * (T - time_grid))
    B = r2 * (y2 - y1) + (1 / N) * torch.sum(omega_star * weights)

    return B.item()

def compute_analytic_trajectories(x0, omega_star, B, T, c0, r1, r2, y1):
    """
    Compute analytic trajectories z_{x_i}(t) based on the known omega* and constants.

    Args:
        x0: (M,) tensor of initial positions
        omega_star: (N,) tensor of analytic prices
        B: scalar (precomputed from compute_B)
        T: time horizon
        c0, r1, r2, y1: scalar constants

    Returns:
        z: (M, N+1) tensor of analytic trajectories
    """
    M = x0.shape[0]
    N = omega_star.shape[0]
    dt = T / N

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
        k = (r1 / c0) ** 0.5
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

    return z

z_numeric = compute_trajectories(alpha_final, x0, dt)  # shape (M, N+1)

B = compute_B(omega_analytic, T, c0, r1, r2, y1, y2)
z_analytic = compute_analytic_trajectories(x0, omega_analytic, B, T, c0, r1, r2, y1)  # shape (M, N+1)

error = torch.max(torch.abs(z_numeric - z_analytic)).item()
print(f"‖z_numeric - z_analytic‖_inf: {error:.6e}")

# import matplotlib.pyplot as plt

# Time grid (including t = T)
t_grid = torch.linspace(0, T, N + 1)

# Plot first 3 trajectories
plt.figure(figsize=(10, 6))
for m in range(min(10, M)):
    plt.plot(t_grid.detach().numpy(), z_numeric[m].detach().numpy(), label=f"Numerical z[{m}]", linewidth=2)
    plt.plot(t_grid.detach().numpy(), z_analytic[m].detach().numpy(), '--', label=f"Analytic z*[{m}]", linewidth=2)


plt.xlabel("Time")
plt.ylabel("Trajectory z(t)")
plt.title("Comparison of Numerical and Analytic Trajectories")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()



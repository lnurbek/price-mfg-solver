import torch
from optimization import run_optimization
from objective import compute_objective, compute_trajectories
from analytic import compute_analytic_omega, compute_analytic_trajectories
import time
import psutil
import os

# Use float64 for accuracy in testing
torch.set_default_dtype(torch.float64)
torch.manual_seed(0)

# Cost function parameters
c0 = 1.0    # control cost weight
r1 = 10.0    # running potential strength
r2 = 10.0    # terminal cost strength
y1 = 0    # running potential center
y2 = 0    # terminal cost target


# Define test cost functions
def L(z, a):
    return 0.5 * c0 * a**2 + 0.5 * r1 * (z - y1)**2

def g(z_T):
    return 0.5 * r2 * (z_T - y2)**2


# Problem setup
M, N = 100, 100
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

z_numeric = compute_trajectories(alpha_final, x0, dt)  # shape (M, N+1)
z_analytic = compute_analytic_trajectories(x0, omega_analytic, T, c0, r1, r2, y1)  # shape (M, N+1)

error = torch.max(torch.abs(z_numeric - z_analytic)).item()
print(f"‖z_numeric - z_analytic‖_inf: {error:.6e}")

# import matplotlib.pyplot as plt

# Time grid (including t = T)
t_grid = torch.linspace(0, T, N + 1)

# Plot a selection of trajectories

# indices = torch.randperm(M)[:10]  # pick 10 random unique indices
indices = range(0, M, 10)  # every 10th trajectory up to M

plt.figure(figsize=(10, 6))
for m in indices:
    plt.plot(t_grid.detach().numpy(), z_numeric[m].detach().numpy(), label=f"Numerical z[{m}]", linewidth=2)
    plt.plot(t_grid.detach().numpy(), z_analytic[m].detach().numpy(), '--', label=f"Analytic z*[{m}]", linewidth=2)


plt.xlabel("Time")
plt.ylabel("Trajectory z(t)")
plt.title("Comparison of Numerical and Analytic Trajectories")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()



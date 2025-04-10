import torch
from optimization import run_optimization
from objective import compute_objective

# Use float64 for accuracy in testing
torch.set_default_dtype(torch.float64)
torch.manual_seed(0)

# Cost function parameters
c0 = 1.0    # control cost weight
r1 = 1.0    # running potential strength
r2 = 1.0    # terminal cost strength
y1 = 0.0    # running potential center
y2 = 0.0    # terminal cost target


# Define test cost functions
def L(z, a):
    return 0.5 * c0 * a**2 + 0.5 * r1 * (z - y1)**2

def g(z_T):
    return 0.5 * r2 * (z_T - y2)**2


# Problem setup
M, N = 100, 100
T = 1.0
dt = T / N
sigma = 0.01
tau_alpha = 1e-2
tau_omega = 1e-1
num_iters = 1000

# Initial data
x0 = torch.linspace(-1, 1, M)
Q = torch.sin(torch.linspace(0, T, N))

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
final_obj = compute_objective(alpha_final, omega_final, x0, dt, sigma, Q, L, g)
final_obj.backward()
grad_alpha = alpha_final.grad

# Print results
print(f"Final objective value: {final_obj.item():.6f}")
print(f"Gradient norm (alpha): {torch.norm(grad_alpha):.6e}")

end_time = time.time()
print_memory_usage()
print(f"Optimization took {end_time - start_time:.3f} seconds")


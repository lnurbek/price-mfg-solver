
import torch
import time
import psutil
import os
import matplotlib.pyplot as plt
from optimization import run_optimization
from objective import compute_objective, compute_trajectories
from analytic import compute_analytic_omega, compute_analytic_trajectories
import os

torch.set_default_dtype(torch.float64)

def print_memory_usage():
    mem_MB = psutil.Process(os.getpid()).memory_info().rss / 1024**2
    print(f"Memory usage: {mem_MB:.2f} MB")

def run_experiment(cfg):
    torch.manual_seed(0)

    c0 = cfg['c0']
    r1, r2 = cfg.get('r1', 0.0), cfg.get('r2', 0.0)
    r3, r4 = cfg.get('r3', 0.0), cfg.get('r4', 0.0)
    y1, y2 = cfg.get('y1', 0.0), cfg.get('y2', 0.0)
    y3, y4 = cfg.get('y3', 0.0), cfg.get('y4', 0.0)
    y5, y6 = cfg.get('y5', 0.0), cfg.get('y6', 0.0)
    Q_func = cfg['Q_func']
    analytic = cfg.get('analytic', False)

    def L(z, a):
        return 0.5 * c0 * a**2 + 0.5 * r1 * (z - y1)**2 + 0.5 * r3 * (z - y3)**2 * (z - y4)**2

    def g(z_T):
        return 0.5 * r2 * (z_T - y2)**2 + 0.5 * r4 * (z_T - y5)**2 * (z_T - y6)**2

    M, N, T = 100, 1000, 1.0
    dt, sigma = T/N, 0.0
    tau_alpha, tau_omega, num_iters = 1e-2, 1e-1, 10000
    x0 = torch.linspace(0, 1, M)
    Q = Q_func(N, T)
    alpha0 = torch.randn(M, N)
    omega0 = torch.zeros(N)

    print(f"Running {cfg['name']}")
    print_memory_usage()
    start_time = time.time()

    alpha_final, omega_final = run_optimization(alpha0, omega0, x0, dt, sigma, Q, L, g,
        tau_alpha, tau_omega, num_iters)

    alpha_final = alpha_final.detach().clone().requires_grad_(True)
    omega_final = omega_final.detach().clone().requires_grad_(True)
    final_obj = compute_objective(alpha_final, omega_final, x0, dt, sigma, Q, L, g)
    final_obj.backward()
    print(f"Final objective value: {final_obj.item():.6f}")
    print(f"Gradient norm (alpha): {torch.norm(alpha_final.grad):.6e}")
    print(f"Gradient norm (omega): {torch.norm(omega_final.grad):.6e}")
    print(f"Optimization took {time.time() - start_time:.3f} seconds")
    print_memory_usage()

    # Plot ω(t)
    os.makedirs("plots", exist_ok=True)
    t_grid = torch.linspace(0, T * (N - 1) / N, N)
    plt.figure(figsize=(8, 5))
    plt.plot(t_grid, omega_final.detach(), label='Numerical ω', linewidth=2)
    plt.plot(t_grid, Q, label='Supply Q', linewidth=1.25)
    if analytic:
        omega_analytic = compute_analytic_omega(Q, x0, T, c0, r1, r2, y1, y2)
        plt.plot(t_grid, omega_analytic, label='Analytic ω*', linestyle='--', linewidth=1)
        print("‖omega_numeric - omega_analytic‖_inf:",
              torch.max(torch.abs(omega_final - omega_analytic)).item())
    plt.xlabel("Time")
    # plt.title(f"{cfg['name']} - ω(t)")
    # plt.title(f"Numerical ω(t)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"plots/{cfg['name']}_omega.png")
    plt.show()

    # Plot z(t)
    z_numeric = compute_trajectories(alpha_final, x0, dt)
    if analytic:
        z_analytic = compute_analytic_trajectories(x0, omega_analytic, T, c0, r1, r2, y1)
    t_grid = torch.linspace(0, T, N + 1)
    plt.figure(figsize=(10, 6))
    for m in range(0, M, 10):
        plt.plot(t_grid.detach().numpy(), z_numeric[m].detach().numpy(), label=f"Numerical z[{m}]", linewidth=2)
        if analytic:
            plt.plot(t_grid.detach().numpy(), z_analytic[m].detach().numpy(), label=f"Analytic z[{m}]", linestyle='--', linewidth=1)
    if analytic:
        print("‖z_numeric - z_analytic‖_inf:", torch.max(torch.abs(z_numeric - z_analytic)).item())
    plt.xlabel("Time")
    plt.ylabel("State")
    # plt.title(f"{cfg['name']} - z(t)")
    # plt.title(f"Numerical z(t)")
    plt.legend(loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"plots/{cfg['name']}_z.png")
    plt.show()

configs = [
    {
        'name': 'case1',
        'c0': 1.0, 'r1': 0.0, 'r2': 10.0,
        'y1': 0.0, 'y2': 0.0,
        'Q_func': lambda N, T: torch.sin(10 * torch.linspace(0, T, N)),
        'analytic': True
    },
    {
        'name': 'case2',
        'c0': 1.0, 'r1': 10.0, 'r2': 0.0,
        'y1': 0.0, 'y2': 0.0,
        'Q_func': lambda N, T: (1 / N ** 0.5) * torch.randn(N).cumsum(0),
        'analytic': True
    },
    {
        'name': 'case3',
        'c0': 1.0, 'r3': 0.0, 'r4': 50.0,
        'y3': 0.25, 'y4': 0.75, 'y5': 0.25, 'y6': 0.75,
        'Q_func': lambda N, T: torch.sin(10 * torch.linspace(0, T, N)),
    },
    {
        'name': 'case4-1',
        'c0': 1.0, 'r3': 50.0, 'r4': 0.0,
        'y3': 0.25, 'y4': 0.75, 'y5': 0.25, 'y6': 0.75,
        'Q_func': lambda N, T: torch.sin(10 * torch.linspace(0, T, N)),
    },
    {
        'name': 'case4-2',
        'c0': 1.0, 'r3': 50.0, 'r4': 0.0,
        'y3': 0.25, 'y4': 0.75, 'y5': 0.25, 'y6': 0.75,
        'Q_func': lambda N, T: (1 / N ** 0.5) * torch.randn(N).cumsum(0),
    },
]

def main():
    run_experiment(configs[0]) # run for case2
if __name__ == '__main__':
    main()


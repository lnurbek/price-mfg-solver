import torch
import numpy as np
import random
import time
import matplotlib.pyplot as plt
from optimization import run_optimization
from objective import compute_objective, compute_trajectories
from analytic import compute_analytic_solution
import os
import sys
import json

torch.set_default_dtype(torch.float64) # Use double precision for accuracy in testing

SEED = 0 # Make everything reproducible

def run_experiments(cfg, seed=None):

    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(True)
        # torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

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
    dt = T/N
    tau_alpha, tau_omega, num_iters = 1e-2, 1e-1, 10000
    x0 = torch.linspace(0, 1, M)
    Q = Q_func(N, T)
    alpha0 = torch.randn(M, N)
    omega0 = torch.zeros(N)

    # --- Save results ---

    caller_dir = os.path.dirname(os.path.abspath(sys.modules['__main__'].__file__))

    # Define results and plots folders relative to caller
    results_path = os.path.join(caller_dir, "results")
    plots_path = os.path.join(caller_dir, "plots")

    # Create them
    os.makedirs(results_path, exist_ok=True)
    os.makedirs(plots_path, exist_ok=True)

    results = {}

    # Save a cleaned version of config (exclude the Q_func but keep the description)
    results['config'] = {k: v for k, v in cfg.items() if k != 'Q_func'}
    

    print(f"Running {cfg['name']}")
    start_time = time.time()

    alpha_final, omega_final = run_optimization(alpha0, omega0, x0, dt, Q, L, g,
        tau_alpha, tau_omega, num_iters)

    alpha_final = alpha_final.detach().clone().requires_grad_(True)
    omega_final = omega_final.detach().clone().requires_grad_(True)
    final_obj = compute_objective(alpha_final, omega_final, x0, dt, Q, L, g)
    final_obj.backward()
    grad_alpha = alpha_final.grad
    grad_omega = omega_final.grad


    print(f"Final objective value: {final_obj.item():.6f}")
    print(f"Gradient norm (alpha): {torch.norm(grad_alpha):.6e}")
    print(f"Gradient norm (omega): {torch.norm(grad_omega):.6e}")
    print(f"Optimization took {time.time() - start_time:.3f} seconds")
    
    # Save results
    results['final_objective'] = final_obj.item()
    results['grad_norm_alpha'] = torch.norm(grad_alpha).item()
    results['grad_norm_omega'] = torch.norm(grad_omega).item()
    results['optimization_time_sec'] = time.time() - start_time
    results['random_seed'] = seed

    # Retrieve analytic results if requested
    if analytic:
        omega_analytic, z_analytic = compute_analytic_solution(Q, x0, T, c0, r1, r2, y1, y2)
    
    # Plot ω(t)
    t_grid = 0+dt*torch.arange(N)
    plt.figure(figsize=(8, 5))
    plt.plot(t_grid, omega_final.detach(), label='Numerical ω', linewidth=2)
    plt.plot(t_grid, Q, label='Supply Q', linewidth=1.25)
    if analytic:
        plt.plot(t_grid, omega_analytic, label='Analytic ω*', linestyle='--', linewidth=1)
        omega_error_analytic = torch.max(torch.abs(omega_final - omega_analytic)).item()
        print("L^infinity error between numerical and analytic ω:", omega_error_analytic)
        results['omega_diff_inf'] = omega_error_analytic
    plt.xlabel("Time")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_path, f"{cfg['name']}_omega.png"))
    plt.show()

    # Plot z(t)
    z_final = compute_trajectories(alpha_final, x0, dt)
    t_grid_full = torch.cat((t_grid,torch.tensor([T])))
    plt.figure(figsize=(8, 5))
    for m in range(0, M, 10):
        plt.plot(t_grid_full.detach().numpy(), z_final[m].detach().numpy(), label=f"Numerical z[{m}]", linewidth=2)
        if analytic:
            plt.plot(t_grid_full.detach().numpy(), z_analytic[m].detach().numpy(), label=f"Analytic z[{m}]", linestyle='--', linewidth=1)
    if analytic:
        z_error_analytic = torch.max(torch.abs(z_final - z_analytic)).item()
        print("L^infinity error between numerical and analytic trajectories:", z_error_analytic)
        results['z_diff_inf'] = z_error_analytic
    plt.xlabel("Time")
    plt.ylabel("State")
    plt.legend(loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_path, f"{cfg['name']}_z.png"))
    plt.show()

    # Save final omega
    results['omega_final'] = omega_final.detach().cpu().numpy().tolist()
    if analytic:
        # Save analytic omega
        results['omega_analytic'] = omega_analytic.detach().cpu().numpy().tolist()
    # Save results to JSON  
    with open(os.path.join(results_path, f"{cfg['name']}_results.json"), "w") as f:
        json.dump(results, f, indent=2)

configs = [
    {
        'name': 'case1',
        'c0': 1.0, 'r1': 0.0, 'r2': 10.0, 'r3': 0.0, 'r4': 0.0,
        'y1': 0.0, 'y2': 0.0, 'y3': 0.25, 'y4': 0.75, 'y5': 0.25, 'y6': 0.75,
        'Q_func': lambda N, T: torch.sin(10 * torch.linspace(0, T, N)),
        'Q_func_description': "Q(t) = sin(10 * t)",
        'analytic': True
    },
    {
        'name': 'case2',
        'c0': 1.0, 'r1': 10.0, 'r2': 0.0, 'r3': 0.0, 'r4': 0.0,
        'y1': 0.0, 'y2': 0.0, 'y3': 0.25, 'y4': 0.75, 'y5': 0.25, 'y6': 0.75,
        'Q_func': lambda N, T: (T / N )** 0.5 * torch.randn(N).cumsum(0),
        'Q_func_description': "A realization of a Wiener process",
        'analytic': True
    },
    {
        'name': 'case3',
        'c0': 1.0, 'r1': 0.0, 'r2': 0.0,'r3': 0.0, 'r4': 50.0,
        'y1': 0.0, 'y2': 0.0, 'y3': 0.25, 'y4': 0.75, 'y5': 0.25, 'y6': 0.75,
        'Q_func': lambda N, T: torch.sin(10 * torch.linspace(0, T, N)),
        'Q_func_description': "Q(t) = sin(10 * t)",
    },
    {
        'name': 'case4-1',
        'c0': 1.0, 'r1': 0.0, 'r2': 0.0,'r3': 50.0, 'r4': 0.0,
        'y1': 0.0, 'y2': 0.0, 'y3': 0.25, 'y4': 0.75, 'y5': 0.25, 'y6': 0.75,
        'Q_func': lambda N, T: torch.sin(10 * torch.linspace(0, T, N)),
        'Q_func_description': "Q(t) = sin(10 * t)",
    },
    {
        'name': 'case4-2',
        'c0': 1.0, 'r1': 0.0, 'r2': 0.0, 'r3': 50.0, 'r4': 0.0,
        'y1': 0.0, 'y2': 0.0, 'y3': 0.25, 'y4': 0.75, 'y5': 0.25, 'y6': 0.75,
        'Q_func': lambda N, T: (T / N )** 0.5 * torch.randn(N).cumsum(0),
        'Q_func_description': "A realization of a Wiener process",
    },
]

def main():
    # run_experiments(configs[1], SEED) # Select the configuration you want to run
    for cfg in configs: # Uncomment to run all configurations
        run_experiments(cfg, SEED)

if __name__ == '__main__':
    main()
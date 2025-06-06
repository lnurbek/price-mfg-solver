# Introduction

This repository contains the implementation of the price-optimization algorithm developed in the paper [https://arxiv.org/abs/2506.04169](https://arxiv.org/abs/2506.04169) for a mean-field games price formation model.

## Repository Structure

- **`src/`** — Source code containing the core modules:
  - **`analytic.py`**
    - Provides analytic solutions when the running and terminal costs are quadratic: $$L(z,\alpha) = c_0 \frac{\alpha^2}{2} + \frac{r_1}{2}(z - y_1)^2, \quad g(z) = \frac{r_2}{2}(z - y_2)^2.$$
    - **`compute_analytic_solution`**: Computes the analytic price and trajectories based on the supply $Q$, initial agent positions $x_0$, terminal time $T$, and cost parameters.

  - **`objective.py`**
    - Defines functions to generate trajectories and evaluate the saddle-point objective function: $$\inf_{\omega} \sup_{\alpha} \mathcal{L}(\omega, \alpha).$$
    - **`compute_trajectories`**: Uses a simple forward Euler integrator to compute trajectories for given controls $\alpha$.
    - **`compute_objective`**: Computes the objective function using a simple left-endpoint quadrature rule.

  - **`optimization.py`**
    - **`run_optimization`**: Runs the main optimization loop of **Algorithm 1** from [https://arxiv.org/abs/2506.04169](https://arxiv.org/abs/2506.04169).


- **`demo/`** — Scripts and folders for reproducing numerical experiments:
  - **`demo.py`**
    - Provides a function to reproduce and save the numerical experiments described in [https://arxiv.org/abs/2506.04169](https://arxiv.org/abs/2506.04169).
    - **`run_experiments`**: Runs a set of experiments for running and terminal costs of the form:
      $$L(z,\alpha) = c_0 \frac{\alpha^2}{2} + \frac{r_1}{2}(z - y_1)^2 + \frac{r_3}{2}(z - y_3)^2 (z - y_4)^2,\quad g(z) = \frac{r_2}{2}(z - y_2)^2 + \frac{r_4}{2}(z - y_5)^2 (z - y_6)^2,$$

      where the supply is either sinusoidal or a realization of a Wiener process.

  - **`plots/`**
    - Directory for storing generated figures of computed prices and state trajectories.

  - **`results/`**
    - Directory for storing saved experiment results in JSON format.


- **`setup.py`** — A script to setup the **`src/`** as package

## Reproducing the Results

To reproduce the numerical experiments presented in [https://arxiv.org/abs/2506.04169](https://arxiv.org/abs/2506.04169), follow these steps:

1. **Clone the repository**:

    ```bash
    git clone https://github.com/lnurbek/price-mfg-solver
    cd price-mfg-solver
    ```

2. **Install the package** and required dependencies using `setup.py`:

    ```bash
    pip install -e .
    ```

    Alternatively, for an exact reproduction environment, install from the pinned `requirements.txt`:

    ```bash
    pip install -r requirements.txt
    ```

3. **Run the demo script**:

    ```bash
    python demo/demo.py
    ```

This will automatically generate the plots and results:
- Figures will be saved in `demo/plots/`.
- Results will be saved in `demo/results/`.

# Extending the Code to New Problems

The code is designed to be modular and easily extensible for different classes of running and terminal costs, dynamics, and objective function evaluations.

### Adapting to Different Running and Terminal Costs

- The `optimization` and `objective` modules accept the running cost $L$ and terminal cost $g$ as **callable function inputs**.
- To solve a different mean-field game with new $L$ and $g$, modify the **wrapper** function `run_experiments` in `demo/demo.py` to construct a suitable class of cost functions based on your model's configuration.
- **No need to modify** the optimization or objective code directly — only the wrapper needs adjustment.

### Adapting to a Different Dynamics or Integrator

- If your problem involves more complex dynamics than the default $\dot{z} = \alpha$ or you prefer a more sophisticated time integrator (e.g., Runge-Kutta methods instead of forward Euler), modify the function **`compute_trajectories`** in `objective.py`.
- Other parts of the code will remain unchanged.

### Using a Different Quadrature Rule

- The objective function is currently evaluated using a simple left-endpoint quadrature rule.
- To implement a more accurate quadrature (e.g., trapezoidal or Simpson’s rule), modify only the **`compute_objective`** function in `objective.py`.


In summary:
- **To change the costs** → adjust the demo wrapper.
- **To change the dynamics or the integrator** → adjust `compute_trajectories`.
- **To change the quadrature** → adjust `compute_objective`.

This modular design ensures that adapting the solver to new problems requires minimal changes.

# Citing This Work

### 📚 Paper

**Xu Wang**, **Samy Wu Fung**, and **Levon Nurbekyan**.  
*“A primal-dual price-optimization method for computing equilibrium prices in mean-field games models.”*  
arXiv preprint, 2025. [arXiv:2506.04169](https://arxiv.org/abs/2506.04169).

BibTeX:
```bibtex
@article{WangFungNurbekyan2025mfg-price-paper,
      title={A primal-dual price-optimization method for computing equilibrium prices in mean-field games models}, 
      author={Xu Wang and Samy Wu Fung and Levon Nurbekyan},
      year={2025},
      eprint={2506.04169},
      archivePrefix={arXiv},
      primaryClass={math.OC},
      url={https://arxiv.org/abs/2506.04169}, 
}
```

### 💻 Code

BibTeX:
```bibtex
@misc{WangFungNurbekyan2025mfg-price-code,
      author = {Xu Wang and Samy Wu Fung and Levon Nurbekyan},
      title = {price-mfg-solver: A primal-dual price-optimization solver for mean-field games},
      year = {2025},
      howpublished = {\url{https://github.com/lnurbek/price-mfg-solver}},
      note = {GitHub repository}
}
```

# License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

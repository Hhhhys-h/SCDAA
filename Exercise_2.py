import subprocess
import sys

def install_requirements():
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "Requirements.txt"])
        print("Success")
    except subprocess.CalledProcessError as e:
        print(f"Failed: {e}")
        sys.exit(1)

install_requirements()

import torch
import numpy as np
import scipy
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

"""
    Exercise 2.1
"""

class LQRSolver:
    """
    Linear Quadratic Regulator Solver: for Exercise 1.1
    """
    def __init__(self, H, M, sigma, C, D, R, T, time_grid=None):
        """
        Initialize the LQR solver

        Args:
            H, M, sigma, C, D, R: LQR matrices
            T: terminal time
            time_grid: time discretization (numpy array or torch tensor)
        """
        self.H = torch.tensor(H, dtype=torch.float32) if not isinstance(H, torch.Tensor) else H
        self.M = torch.tensor(M, dtype=torch.float32) if not isinstance(M, torch.Tensor) else M
        self.sigma = torch.tensor(sigma, dtype=torch.float32) if not isinstance(sigma, torch.Tensor) else sigma
        self.C = torch.tensor(C, dtype=torch.float32) if not isinstance(C, torch.Tensor) else C
        self.D = torch.tensor(D, dtype=torch.float32) if not isinstance(D, torch.Tensor) else D
        self.R = torch.tensor(R, dtype=torch.float32) if not isinstance(R, torch.Tensor) else R
        self.T = T
        self.D_inv = torch.inverse(self.D)

        if time_grid is None:
            self.time_grid = torch.linspace(0, T, 100)
        else:
            self.time_grid = torch.tensor(time_grid, dtype=torch.float32) if not isinstance(time_grid, torch.Tensor) else time_grid

        self.S_values = self.solve_riccati_ode()

    def riccati_ode(self, t, S_flat):
        """Define the Riccati ODE"""
        S = S_flat.reshape(2, 2)
        S_np = S
        H_np = self.H.numpy()
        M_np = self.M.numpy()
        D_inv_np = np.linalg.inv(self.D.numpy())
        C_np = self.C.numpy()
        term1 = S_np @ M_np @ D_inv_np @ M_np.T @ S_np
        term2 = H_np.T @ S_np
        term3 = S_np @ H_np
        dSdt = term1 - term2 - term3 - C_np
        return dSdt.flatten()

    def solve_riccati_ode(self):
        """Solve the Riccati ODE"""
        t_grid = self.time_grid.numpy()
        S_T = self.R.numpy().flatten()
        sol = solve_ivp(
            self.riccati_ode,
            [self.T, 0],
            S_T,
            t_eval=np.flip(t_grid),
            method='RK45',
            rtol=1e-13,
            atol=1e-15
        )
        S_values = []
        for i in range(sol.y.shape[1]):
            S = sol.y[:, i].reshape(2, 2)
            S_values.append(torch.tensor(S, dtype=torch.float32))
        return S_values[::-1]

    def find_nearest_time_index(self, t):
        """Find the nearest time index to t"""
        if t >= self.T:
            return len(self.time_grid) - 1
        if t <= 0:
            return 0
        time_array = self.time_grid.numpy()
        idx = np.searchsorted(time_array, t, side='right') - 1
        return idx

    def value_function(self, t, x):
        """
        Evaluate value function at time t and state x

        Args:
            t: 1D tensor of time values
            x: 2D tensor of states (batch_size x 2)

        Returns:
            1D tensor of value function estimates
        """
        batch_size = x.shape[0]
        values = torch.zeros(batch_size)
        for i in range(batch_size):
            t_i = t[i].item()
            idx = self.find_nearest_time_index(t_i)
            S_t = self.S_values[idx]
            x_i = x[i]
            values[i] = x_i @ S_t @ x_i
            if idx < len(self.time_grid) - 1:
                integral_term = 0.0
                sigma_sigma_T = self.sigma @ self.sigma.T
                for j in range(idx, len(self.time_grid) - 1):
                    dt = self.time_grid[j+1] - self.time_grid[j]
                    S_j = self.S_values[j]
                    S_j_plus_1 = self.S_values[j+1]
                    trace_j = torch.trace(sigma_sigma_T @ S_j)
                    trace_j_plus_1 = torch.trace(sigma_sigma_T @ S_j_plus_1)
                    integral_term += 0.5 * (trace_j + trace_j_plus_1) * dt
                values[i] += integral_term
        return values

    def optimal_control(self, t, x):
        """
        Compute optimal control

        Args:
            t: 1D tensor of time values
            x: 2D tensor of states (batch_size x 2)

        Returns:
            2D tensor of optimal controls (batch_size x 2)
        """
        batch_size = x.shape[0]
        controls = torch.zeros((batch_size, 2))
        for i in range(batch_size):
            t_i = t[i].item()
            idx = self.find_nearest_time_index(t_i)
            S_t = self.S_values[idx]
            controls[i] = -self.D_inv @ self.M.T @ S_t @ x[i]
        return controls

class SoftLQRSolver(LQRSolver):
    """
    Soft Linear Quadratic Regulator Solver: for Exercise 2.1
    """
    def compute_value_constant(self):
        """
        Compute value constant C_{D,τ,γ} in Soft LQR, according to equation (22):
        C = -τ [ (m/2)·log(τ) - m·log(γ) + 0.5·log(det(Σ)) ]
        """
        d = self.D.shape[0]
        Sigma = torch.inverse(self.D + (self.tau / (2 * self.gamma ** 2)) * torch.eye(d))
        det_sigma = torch.det(Sigma)
        C = -self.tau * (0.5 * d * torch.log(torch.tensor(self.tau)) - d * torch.log(torch.tensor(self.gamma)) + 0.5 * torch.log(det_sigma))
        return C.item()

    def __init__(self, H, M, sigma, C, D, R, T, time_grid=None, tau=0.1, gamma=10.0):
        """
        Initialize the Soft LQR solver
        
        Args:
            H, M, sigma, C, D, R: LQR matrices
            T: terminal time
            time_grid: time discretization (optional)
            tau: entropy regularization coefficient
            gamma: prior control distribution variance
        """
        self.H = torch.tensor(H, dtype=torch.float32) if not isinstance(H, torch.Tensor) else H
        self.M = torch.tensor(M, dtype=torch.float32) if not isinstance(M, torch.Tensor) else M
        self.sigma = torch.tensor(sigma, dtype=torch.float32) if not isinstance(sigma, torch.Tensor) else sigma
        self.C = torch.tensor(C, dtype=torch.float32) if not isinstance(C, torch.Tensor) else C
        self.D = torch.tensor(D, dtype=torch.float32) if not isinstance(D, torch.Tensor) else D
        self.R = torch.tensor(R, dtype=torch.float32) if not isinstance(R, torch.Tensor) else R
        self.T = T
        self.D_inv = torch.inverse(self.D)

        self.tau = tau
        self.gamma = gamma

        self.time_grid = torch.linspace(0, T, 100) if time_grid is None else torch.tensor(time_grid, dtype=torch.float32)

        self.S_values = self.solve_riccati_ode()

    def riccati_ode(self, t, S_flat):
        """Riccati ODE with entropy regularization (Soft LQR)"""
        S = S_flat.reshape(2, 2)
        S_np = S
        H_np = self.H.numpy()
        M_np = self.M.numpy()
        C_np = self.C.numpy()
        Sigma = np.linalg.inv(self.D.numpy() + (self.tau / (2 * self.gamma**2)) * np.eye(2))

        term1 = S_np.T @ M_np @ Sigma @ M_np.T @ S
        term2 = H_np.T @ S
        term3 = S @ H_np
        dSdt = term1 - term2 - term3 - C_np
        return dSdt.flatten()

    def control_distribution(self, t, x):
        """
        Compute parameters of the optimal control distribution
        
        Returns:
            Tuple of (means, covariances)
        """
        batch_size = x.shape[0]
        means = torch.zeros((batch_size, 2))
        covariances = []

        for i in range(batch_size):
            t_i = t[i].item()
            idx = self.find_nearest_time_index(t_i)
            S_t = self.S_values[idx]
            means[i] = -torch.inverse(self.D + (self.tau / (2 * self.gamma ** 2)) * torch.eye(2)) @ self.M.T @ S_t @ x[i]
            cov = self.tau * (self.D + (self.tau / (2 * self.gamma ** 2)) * torch.eye(2))
            covariances.append(cov)

        return means, covariances

    def sample_control(self, t, x):
        """
        Sample control action from the control distribution
        """
        means, covariances = self.control_distribution(t, x)
        batch_size = x.shape[0]
        samples = torch.zeros((batch_size, 2))

        for i in range(batch_size):
            mean_np = means[i].numpy()
            cov_np = covariances[i].numpy()
            sample = np.random.multivariate_normal(mean_np, cov_np)
            samples[i] = torch.tensor(sample, dtype=torch.float32)

        return samples


def simulate_trajectory(controller, x0, T, time_grid, use_sampling=False, fixed_brownian=None):
    """
    Simulate state trajectory using Euler method
    
    Args:
        controller: instance of LQRSolver or SoftLQRSolver
        x0: initial state vector
        T: final time
        time_grid: time discretization
        use_sampling: whether to sample control (for Soft LQR)
        fixed_brownian: fixed Brownian increments (for fair comparison)

    Returns:
        time series of states, controls, and noise
    """
    n_steps = len(time_grid) - 1
    states = torch.zeros((len(time_grid), 2))
    controls = torch.zeros((n_steps, 2))
    states[0] = x0
    dt = (time_grid[1] - time_grid[0]).item()

    dW = torch.randn(n_steps, controller.sigma.shape[1]) * torch.sqrt(torch.tensor(dt)) if fixed_brownian is None else fixed_brownian

    for i in range(n_steps):
        t_i = time_grid[i].reshape(1)
        x_i = states[i].reshape(1, 2)
        if use_sampling and hasattr(controller, 'sample_control'):
            a_i = controller.sample_control(t_i, x_i)[0]
        else:
            a_i = controller.optimal_control(t_i, x_i)[0]

        controls[i] = a_i
        drift = controller.H @ states[i] + controller.M @ a_i
        diffusion = controller.sigma @ dW[i]
        states[i+1] = states[i] + drift * dt + diffusion

    return time_grid, states, controls, dW


def compare_trajectories(standard_lqr, soft_lqr, starting_points, T, time_grid):
    """
    Compare trajectories between strict LQR and soft LQR
    """
    for i, x0 in enumerate(starting_points):
        print(f"\nSimulating from initial point {i+1}/4: {x0.tolist()}")
        plt.figure(figsize=(10, 8))

        n_steps = len(time_grid) - 1
        dt = (time_grid[1] - time_grid[0]).item()
        np.random.seed(i)
        dW = torch.tensor(np.random.normal(0, np.sqrt(dt), (n_steps, 2)), dtype=torch.float32)

        print("  Simulating strict LQR trajectory...")
        _, states_std, _, _ = simulate_trajectory(standard_lqr, x0, T, time_grid, fixed_brownian=dW)

        print("  Simulating soft LQR trajectory...")
        _, states_soft, _, _ = simulate_trajectory(soft_lqr, x0, T, time_grid, use_sampling=True, fixed_brownian=dW)

        plt.plot(states_std[:, 0].numpy(), states_std[:, 1].numpy(), 'b-', linewidth=2, label='Strict LQR')
        plt.plot(states_soft[:, 0].numpy(), states_soft[:, 1].numpy(), 'r-', linewidth=2, label='Soft LQR (τ=5.0, γ=10)')
        plt.scatter([x0[0].item()], [x0[1].item()], c='k', s=100, marker='o', label='Initial Point')

        plt.title(f'Trajectory from initial point ({x0[0].item()}, {x0[1].item()})')
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'trajectory_from_{x0[0].item()}_{x0[1].item()}.png')
        plt.show()

    print("\nAll simulations completed!")

def main_exercise2():
    # === Model parameters ===
    H = torch.tensor([[1.0, 1.0], [0.0, 1.0]]) * 0.5
    M = torch.tensor([[1.0, 1.0], [0.0, 1.0]])
    sigma = torch.eye(2) * 0.5
    C = torch.tensor([[1.0, 0.1], [0.1, 1.0]]) * 1.0
    D = torch.tensor([[1.0, 0.1], [0.1, 1.0]]) * 0.1
    R = torch.tensor([[1.0, 0.3], [0.3, 1.0]]) * 10.0
    T = 0.5
    time_grid = torch.linspace(0, T, 1000)

    standard_lqr = LQRSolver(H, M, sigma, C, D, R, T, time_grid)
    soft_lqr = SoftLQRSolver(H, M, sigma, C, D, R, T, tau=5.0, gamma=10.0, time_grid=time_grid)

    starting_points = [
        torch.tensor([2.0, 2.0]),
        torch.tensor([2.0, -2.0]),
        torch.tensor([-2.0, -2.0]),
        torch.tensor([-2.0, 2.0]),
    ]

    compare_trajectories(standard_lqr, soft_lqr, starting_points, T, time_grid)
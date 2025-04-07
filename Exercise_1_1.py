import subprocess
import sys

def install_requirements():
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "Requirements.txt"])
        print("Dependencies installed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error installing dependencies: {e}")
        sys.exit(1)

install_requirements()

import torch
import numpy as np
import scipy
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

"""
    Exercise 1
"""

class LQRSolver:
    """
    Linear Quadratic Regulator (LQR) solver: satisfies Exercise 1.1 requirements
    """
    def __init__(self, H, M, sigma, C, D, R, T, time_grid=None):
        """
        Initialize the LQR solver

        Args:
            H, M, sigma, C, D, R: LQR problem matrices
            T: terminal time
            time_grid: time grid (numpy array or torch tensor)
        """
        self.H = torch.tensor(H, dtype=torch.float32) if not isinstance(H, torch.Tensor) else H
        self.M = torch.tensor(M, dtype=torch.float32) if not isinstance(M, torch.Tensor) else M
        self.sigma = torch.tensor(sigma, dtype=torch.float32) if not isinstance(sigma, torch.Tensor) else sigma
        self.C = torch.tensor(C, dtype=torch.float32) if not isinstance(C, torch.Tensor) else C
        self.D = torch.tensor(D, dtype=torch.float32) if not isinstance(D, torch.Tensor) else D
        self.R = torch.tensor(R, dtype=torch.float32) if not isinstance(R, torch.Tensor) else R
        self.T = T

        # Compute inverse of D
        self.D_inv = torch.inverse(self.D)

        # Create or use time grid
        if time_grid is None:
            self.time_grid = torch.linspace(0, T, 100)
        else:
            self.time_grid = torch.tensor(time_grid, dtype=torch.float32) if not isinstance(time_grid, torch.Tensor) else time_grid

        # Solve Riccati equation
        self.S_values = self.solve_riccati_ode()

    def riccati_ode(self, t, S_flat):
        """Define Riccati ODE"""
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
        """Find the closest time index to t"""
        if t >= self.T:
            return len(self.time_grid) - 1
        if t <= 0:
            return 0

        time_array = self.time_grid.numpy()
        idx = np.searchsorted(time_array, t, side='right') - 1
        return idx

    def value_function(self, t, x):
        """
        Compute the value function V(t, x)

        Args:
            t: 1D torch tensor of time points
            x: 2D torch tensor of states (batch_size x 2)

        Returns:
            1D tensor: estimated value
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
                sigma_sigma_T = self.sigma @ self.sigma.T
                traces = torch.tensor([torch.trace(sigma_sigma_T @ S) for S in self.S_values[idx:]])
                dt = self.time_grid[idx+1:] - self.time_grid[idx:-1]
                integral_term = torch.sum(0.5 * (traces[:-1] + traces[1:]) * dt)
                values[i] += integral_term

        return values

    def optimal_control(self, t, x):
        """
        Compute the optimal control a(t,x)

        Args:
            t: 1D torch tensor of time points
            x: 2D torch tensor of states (batch_size x 2)

        Returns:
            2D tensor of controls (batch_size x 2)
        """
        batch_size = x.shape[0]
        controls = torch.zeros((batch_size, 2))

        for i in range(batch_size):
            t_i = t[i].item()
            idx = self.find_nearest_time_index(t_i)
            S_t = self.S_values[idx]
            controls[i] = -self.D_inv @ self.M.T @ S_t @ x[i]

        return controls

import os

def main_exercise1_1():
    # Define model parameters
    H = torch.tensor([[1.0, 1.0],
                     [0.0, 1.0]]) * 0.5
    M = torch.tensor([[1.0, 1.0],
                    [0.0, 1.0]])
    sigma = torch.eye(2) * 0.5
    C = torch.tensor([[1.0, 0.1],
                     [0.1, 1.0]]) * 1.0
    D = torch.tensor([[1.0, 0.1],
                    [0.1, 1.0]]) * 0.1
    R = torch.tensor([[1.0, 0.3],
                     [0.3, 1.0]]) * 10.0
    T = 0.5

    # Create a time grid
    time_grid = torch.linspace(0, T, 100)

    # Initialize the solver
    lqr_solver = LQRSolver(H, M, sigma, C, D, R, T, time_grid)

    # Evaluation
    test_point_t = torch.tensor([0.0])
    test_point_x = torch.tensor([[1.0, 1.0]])

    value = lqr_solver.value_function(test_point_t, test_point_x)
    control = lqr_solver.optimal_control(test_point_t, test_point_x)

    # Make sure the directory exists, if not, create it
    output_dir = "Exercise_1_1_results"
    os.makedirs(output_dir, exist_ok=True)

    # Path for the output text file
    output_file = os.path.join(output_dir, "results.txt")

    # Open the file in write mode
    with open(output_file, 'w') as file:
        # Redirect print statements to the file
        print(f"Value function at t=0, x=[1,1]: {value.item()}", file=file)
        print(f"Optimal control at t=0, x=[1,1]: {control.squeeze(0).tolist()}", file=file)

    print("Results have been saved to 'Exercise_1_1_results/results.txt'.")

if __name__ == "__main__":
    main_exercise1_1()

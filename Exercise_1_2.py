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
    Exercise 1.2
"""

torch.manual_seed(60)
np.random.seed(60)

class LQRSolver:
    """
    Linear Quadratic Regulator (LQR) Solver for Exercise 1.1
    """

    def __init__(self, H, M, sigma, C, D, R, T, time_grid=None):
        """
        Initialize the LQR solver

        Args:
            H, M, sigma, C, D, R: Matrices for the LQR problem
            T: Terminal time
            time_grid: Time grid (numpy array or torch tensor)
        """
        # Ensure all inputs are torch tensors
        self.H = torch.tensor(H, dtype=torch.float32) if not isinstance(H, torch.Tensor) else H
        self.M = torch.tensor(M, dtype=torch.float32) if not isinstance(M, torch.Tensor) else M
        self.sigma = torch.tensor(sigma, dtype=torch.float32) if not isinstance(sigma, torch.Tensor) else sigma
        self.C = torch.tensor(C, dtype=torch.float32) if not isinstance(C, torch.Tensor) else C
        self.D = torch.tensor(D, dtype=torch.float32) if not isinstance(D, torch.Tensor) else D
        self.R = torch.tensor(R, dtype=torch.float32) if not isinstance(R, torch.Tensor) else R
        self.T = T

        # Compute the inverse of D
        self.D_inv = torch.inverse(self.D)

        # Create or use provided time grid
        if time_grid is None:
            self.time_grid = torch.linspace(0, T, 100)
        else:
            self.time_grid = torch.tensor(time_grid, dtype=torch.float32) if not isinstance(time_grid, torch.Tensor) else time_grid

        # Solve the Riccati ODE
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
            [self.T, 0],  # Solve backward from T to 0
            S_T,
            t_eval=np.flip(t_grid),
            method='RK45',
            rtol=1e-8,
            atol=1e-8
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
            1D torch tensor: value function estimate
        """
        batch_size = x.shape[0]
        values = torch.zeros(batch_size)
        for i in range(batch_size):
            t_i = t[i].item()
            idx = self.find_nearest_time_index(t_i)
            S_t = self.S_values[idx]
            x_i = x[i]
            values[i] = x_i @ S_t @ x_i

            # Add integral term if needed
            if idx < len(self.time_grid) - 1:
                integral_term = 0.0
                sigma_sigma_T = self.sigma @ self.sigma.T
                for j in range(idx, len(self.time_grid) - 1):
                    dt = self.time_grid[j + 1] - self.time_grid[j]
                    S_j = self.S_values[j]
                    S_j_plus_1 = self.S_values[j + 1]
                    trace_j = torch.trace(sigma_sigma_T @ S_j)
                    trace_j_plus_1 = torch.trace(sigma_sigma_T @ S_j_plus_1)
                    integral_term += 0.5 * (trace_j + trace_j_plus_1) * dt
                values[i] += integral_term
        return values

    def optimal_control(self, t, x):
        """
        Compute the optimal control a(t, x)

        Args:
            t: 1D torch tensor of time points
            x: 2D torch tensor of states (batch_size x 2)

        Returns:
            2D torch tensor of controls (batch_size x 2)
        """
        batch_size = x.shape[0]
        controls = torch.zeros((batch_size, 2))
        for i in range(batch_size):
            t_i = t[i].item()
            idx = self.find_nearest_time_index(t_i)
            S_t = self.S_values[idx]
            controls[i] = -self.D_inv @ self.M.T @ S_t @ x[i]
        return controls

def simulate_lqr(solver, t0, x0, N, M):
    """
    Simulate the LQR system using Euler method to estimate cost Ĵ(t,x)

    Args:
        solver: instance of LQRSolver
        t0: initial time (usually 0)
        x0: initial state (torch.tensor([x1, x2]))
        N: number of time steps
        M: number of Monte Carlo samples

    Returns:
        float: estimated average cost
    """
    T = solver.T
    dt = (T - t0) / N
    sqrt_dt = torch.sqrt(torch.tensor(dt))
    d = x0.shape[0]
    X = x0.repeat(M, 1)
    cost = torch.zeros(M)
    time_grid = torch.linspace(t0, T, N + 1)
    H = solver.H
    M_mat = solver.M
    sigma = solver.sigma
    C = solver.C
    D = solver.D
    R = solver.R

    for n in range(N):
        t_n = time_grid[n].repeat(M)
        a_n = solver.optimal_control(t_n, X)
        drift = (H @ X.T).T + (M_mat @ a_n.T).T
        dW = torch.randn(M, d) * sqrt_dt
        X = X + dt * drift + dW @ sigma.T
        cost += dt * (
                torch.einsum("bi,ij,bj->b", X, C, X)
                + torch.einsum("bi,ij,bj->b", a_n, D, a_n)
        )

    cost += torch.einsum("bi,ij,bj->b", X, R, X)
    return cost.mean().item()

def plot_error_vs_N(solver, x0, M=10000, N_list=None):
    if N_list is None:
        N_list = [2**i for i in range(1, 12)]  # 2 to 2048
    errors = []
    for N in N_list:
        J_hat = simulate_lqr(solver, 0.0, x0, N, M)
        v_true = solver.value_function(torch.tensor([0.0]), x0.unsqueeze(0))[0].item()
        error = abs(J_hat - v_true)
        errors.append(error)
        print(f"N = {N:<4} | Ĵ: {J_hat:.6f} | v: {v_true:.6f} | Error: {error:.2e}")
    
    plt.figure()
    plt.loglog(N_list, errors, marker='o', label='|Ĵ - v|')
    ref = errors[0] * np.array(N_list[0]) / np.array(N_list)  # O(1/N)
    plt.loglog(N_list, ref, linestyle='--', label='O(1/N) ref')
    plt.xlabel("Time steps N")
    plt.ylabel("Absolute error")
    plt.title(f'Error vs Time Steps (fixed M) {x0}')
    plt.grid(True, which="both")
    plt.legend()
    plt.tight_layout()
    save_path = f'Exercise_1_2_results/Error_vs_Time_Steps_fixed_M_{tuple(x0.tolist())}.png'
    plt.savefig(save_path)
    plt.show()

def plot_error_vs_M(solver, x0, N=10000, M_list=None):
    if M_list is None:
        M_list = [2 * 4**i for i in range(0,6)]  # 2 to 2048
        
    errors = []
    for M in M_list:
        J_hat = simulate_lqr(solver, 0.0, x0, N, M)
        v_true = solver.value_function(torch.tensor([0.0]), x0.unsqueeze(0))[0].item()
        error = abs(J_hat - v_true)
        errors.append(error)
        print(f"M = {M:<6} | Ĵ: {J_hat:.6f} | v: {v_true:.6f} | Error: {error:.2e}")
    
    plt.figure()
    plt.loglog(M_list, errors, marker='s', label='|Ĵ - v|')
    ref = errors[0] * np.sqrt(M_list[0]) / np.sqrt(np.array(M_list))  # O(1/sqrt(M))
    plt.loglog(M_list, ref, linestyle='--', label='O(1/√M) ref')
    plt.xlabel("Sample size M")
    plt.ylabel("Absolute error")
    plt.title(f'Error vs Sample Size (fixed N) {x0}')
    plt.grid(True, which="both")
    plt.legend()
    plt.tight_layout()
    save_path = f'Exercise_1_2_results/Error_vs_Sample_Size_fixed_N_{tuple(x0.tolist())}.png'
    plt.savefig(save_path)
    plt.show()

def main_exercise1_2_initial_1():
    H = torch.tensor([[1.0, 1.0], [0.0, 1.0]]) * 0.5
    M = torch.tensor([[1.0, 1.0], [0.0, 1.0]])
    sigma = torch.eye(2) * 0.5
    C = torch.tensor([[1.0, 0.1], [0.1, 1.0]]) * 1.0
    D = torch.tensor([[1.0, 0.1], [0.1, 1.0]]) * 0.1
    R = torch.tensor([[1.0, 0.3], [0.3, 1.0]]) * 10.0
    T = 0.5
    time_grid = torch.linspace(0, T, 100)

    solver = LQRSolver(H, M, sigma, C, D, R, T, time_grid)

    # Test points
    test_points = [
        torch.tensor([1.0, 1.0]),
    ]
    
    print("\n=== Error: x = (1,1) ===")
#    plot_error_vs_N(solver, torch.tensor([1.0, 1.0]))
    plot_error_vs_M(solver, torch.tensor([1.0, 1.0]))

    print("End")

def main_exercise1_2_initial_2():
    H = torch.tensor([[1.0, 1.0], [0.0, 1.0]]) * 0.5
    M = torch.tensor([[1.0, 1.0], [0.0, 1.0]])
    sigma = torch.eye(2) * 0.5
    C = torch.tensor([[1.0, 0.1], [0.1, 1.0]]) * 1.0
    D = torch.tensor([[1.0, 0.1], [0.1, 1.0]]) * 0.1
    R = torch.tensor([[1.0, 0.3], [0.3, 1.0]]) * 10.0
    T = 0.5
    time_grid = torch.linspace(0, T, 100)

    solver = LQRSolver(H, M, sigma, C, D, R, T, time_grid)

    # Test points
    test_points = [
        torch.tensor([2.0, 2.0]),
    ]

    print("\n=== Error: x = (2,2) ===")
#    plot_error_vs_N(solver, torch.tensor([2.0, 2.0]))
    plot_error_vs_M(solver, torch.tensor([2.0, 2.0]))

    print("End")

# Make sure this is executed when running the script
if __name__ == "__main__":
    main_exercise1_2_initial_1()

if __name__ == "__main__":
    main_exercise1_2_initial_2()
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

######V4
from Exercise_2 import LQRSolver, SoftLQRSolver

class OnlyLinearValueNN(nn.Module):
    def __init__(self, hidden_dim=512, device=torch.device("cpu")):
        super(OnlyLinearValueNN, self).__init__()
        # Remember the device
        self.device = device

        # Add hidden layer for matrix parameters
        self.matrix_network = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2*2)
        ).to(device)

        # Add hidden layer for bias parameters
        self.offset_network = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        ).to(device)

    def forward(self, t):
        """
        Input: time t (batch_size, 1)
        Output:
            - matrix: a matrix of shape (batch_size, 2, 2)
            - offset: a bias of shape (batch_size, 1)
        """
        matrix_elements = self.matrix_network(t)
        matrix = matrix_elements.view(-1, 2, 2)

        # Ensure the matrix is symmetric positive definite
        matrix = torch.bmm(matrix, matrix.transpose(1, 2)) + 1e-3 * torch.eye(2).to(matrix.device)

        # Calculate the bias
        offset = self.offset_network(t)

        return matrix, offset

def evaluate_value(model, t_tensor, x_tensor):
    """Calculate the value function v(t, x) = x^T S(t) x + b(t)"""
    matrix, offset = model(t_tensor)  # Get S(t) and b(t)
    quad_term = torch.einsum('bi,bij,bj->b', x_tensor, matrix, x_tensor)
    return quad_term + offset.view(-1)

# ===== Initial Sampling Function =====
def sample_x0_uniform(batch_size, low=-2.0, high=2.0):
    return torch.empty(batch_size, 2).uniform_(low, high)

# ===== Simulate Trajectory Function (Single) =====
def simulate_trajectory_with_cost(controller, x0, time_grid):
    N = len(time_grid) - 1
    dt = (time_grid[1] - time_grid[0]).item()

    x_seq = torch.zeros((N + 1, 2))
    a_seq = torch.zeros((N, 2))
    cost_seq = torch.zeros(N)
    log_prob_seq = torch.zeros(N)
    t_seq = time_grid[:-1].unsqueeze(1)  # (N, 1)

    x = x0.clone()
    x_seq[0] = x

    dW = torch.randn(N, 2) * np.sqrt(dt)

    for i in range(N):  # Loop over each time step
        t_i = time_grid[i].reshape(1)
        x_i = x.reshape(1, 2)

        a_i = controller.sample_control(t_i, x_i)[0]

        a_seq[i] = a_i

        drift = controller.H @ x + controller.M @ a_i
        diffusion = controller.sigma @ dW[i]
        x_new = x + drift * dt + diffusion

        if i < N:
            x_seq[i + 1] = x_new

        cost = x_i @ controller.C @ x_i.T + a_i @ controller.D @ a_i.T
        cost_seq[i] = cost.item()

        if hasattr(controller, 'control_distribution'):
            mean, cov = controller.control_distribution(t_i, x_i)
            dist = torch.distributions.MultivariateNormal(mean[0], cov[0])
            log_prob_seq[i] = dist.log_prob(a_i)

    terminal_cost = x @ controller.R @ x
    return t_seq, x_seq, a_seq, cost_seq, log_prob_seq, terminal_cost.item()

def compute_target(costs, log_probs, g_T, dt, tau):
    """
    Construct the target value for each trajectory according to Algorithm 2's formula (14)
    """
    N = costs.shape[0]
    cumulative = torch.zeros(N, dtype=costs.dtype, device=costs.device)
    total = 0.0
    for n in reversed(range(N)):
        total += (costs[n] + tau * log_probs[n]) * dt
        cumulative[n] = total
    cumulative += torch.tensor(g_T, dtype=cumulative.dtype, device=cumulative.device)
    return cumulative

# ===== Batch Training of Critic =====
def train_critic_batch(model, controller, time_grid, tau, n_epochs, batch_size, lr, print_every):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Calculate dt at the beginning of the function
    dt = (time_grid[1] - time_grid[0]).item()

    loss_list = []

    for epoch in range(n_epochs):
        x0_batch = sample_x0_uniform(batch_size)
        loss = 0.0

        for i in range(batch_size):
            x0 = x0_batch[i]
            t_seq, x_seq, a_seq, costs, log_probs, g_T = simulate_trajectory_with_cost(
                controller, x0, time_grid)
            t_input = t_seq
            x_input = x_seq[:-1]

            v_pred = evaluate_value(model, t_input, x_input)
            target = compute_target(costs, log_probs, g_T, dt, tau)

            for n in range(len(t_seq)):
                loss += (v_pred[n] - target[n].detach()) ** 2

        # loss = loss / batch_size
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_list.append(loss.item())

        if epoch % print_every == 0 or epoch == n_epochs - 1:
            print(f"[Epoch {epoch:03d}] Critic Loss = {loss.item():.4e}")

    return loss_list

# ===== Evaluate Accuracy =====
def evaluate_critic_accuracy(model, solver, time_grid):
    from itertools import product
    time_list = [0.0, 1/6, 2/6, 0.5]
    x_vals = torch.linspace(-3, 3, 100)
    grid = torch.cartesian_prod(x_vals, x_vals)

    max_error = 0.0
    for t_scalar in time_list:
        t_tensor = torch.tensor([t_scalar]).repeat(grid.shape[0]).view(-1, 1)
        v_hat = evaluate_value(model, t_tensor, grid)
        v_true = solver.value_function(t_tensor.view(-1), grid)
        error = torch.abs(v_hat - v_true)
        max_error = max(max_error, torch.max(error).item())
        print(f"[t={t_scalar:.3f}] Max error = {torch.max(error):.4e}")

    print(f"Overall max error across all t and x: {max_error:.4e}")

def plot_value_function_1d(model, solver, t_scalar=0.2):
    """
    Plot the value function against x_1 while keeping t fixed and x_2 = 0
    """
    x1_vals = torch.linspace(-3, 3, 200)
    x2_fixed = torch.zeros_like(x1_vals)
    x_grid = torch.stack([x1_vals, x2_fixed], dim=1)

    t_tensor = torch.full((x_grid.shape[0], 1), t_scalar)

    v_hat = evaluate_value(model, t_tensor, x_grid).detach()
    v_true = solver.value_function(t_tensor.view(-1), x_grid)

    plt.figure(figsize=(8, 5))
    plt.plot(x1_vals.numpy(), v_true.numpy(), label="True $v^*(t, x)$", linewidth=2)
    plt.plot(x1_vals.numpy(), v_hat.numpy(), label="Predicted $\hat{v}(t, x)$", linestyle='--')
    plt.xlabel("$x_1$ (with $x_2 = 0$)")
    plt.ylabel("Value Function")
    plt.title(f"Value Function Comparison at t = {t_scalar}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    save_path = f'Exercise_3_results/Value_Function_Comparison_at_t = {t_scalar}.png'
    plt.savefig(save_path)
    plt.show()


def main_exercise3_critic_loss():
    # === Parameter Configuration Required by the Instructor ===
    H = torch.tensor([[1.0, 1.0], [0.0, 1.0]]) * 0.5
    M = torch.tensor([[1.0, 1.0], [0.0, 1.0]])
    sigma = torch.eye(2) * 0.5
    C = torch.tensor([[1.0, 0.1], [0.1, 1.0]])
    D = torch.eye(2)  # Assignment requires D = identity
    R = torch.tensor([[1.0, 0.3], [0.3, 1.0]]) * 10.0

    T = 0.5
    tau = 0.5
    gamma = 1.0
    N = 100
    time_grid = torch.linspace(0, T, N + 1)

    # Training parameters
    batch_size = 200
    n_epochs = 50
    hidden_dim = 512
    lr = 1e-3
    print_every = 10

    # === Initialize Soft LQR Controller (Fixed Policy π) ===
    soft_lqr = SoftLQRSolver(H, M, sigma, C, D, R, T, tau=tau, gamma=gamma, time_grid=time_grid)

    # === Initialize Value Network ===
    model = OnlyLinearValueNN(hidden_dim=512)

    # === Train Critic Network ===
    loss_list = train_critic_batch(
        model=model,
        controller=soft_lqr,
        time_grid=time_grid,
        tau=tau,
        n_epochs=n_epochs,
        batch_size=batch_size,
        lr=lr,
        print_every=print_every
    )

    # === Validate Value Network Error ===
    print("\n=== Evaluate Critic Accuracy ===")
    evaluate_critic_accuracy(model, soft_lqr, time_grid)

    plt.figure(figsize=(8, 5))
    plt.plot(range(len(loss_list)), loss_list, label="Critic Loss")
    plt.yscale("log")
    plt.yticks([1e8, 1e7, 1e6], ['$10^8$', '$10^7$', '$10^6$'])
    plt.xlabel("Epoch")
    plt.ylabel("Loss (log scale)")
    plt.title("Critic Loss Over Epochs")
    plt.grid(True)
    plt.legend(loc='upper left')
    plt.legend()
    plt.tight_layout
    save_path = f'Exercise_3_results/Critic_Loss_Over_Epochs.png'
    plt.savefig(save_path)
    plt.show()

def main_exercise3_comparsion():
    # === Parameter Configuration Required by the Instructor ===
    H = torch.tensor([[1.0, 1.0], [0.0, 1.0]]) * 0.5
    M = torch.tensor([[1.0, 1.0], [0.0, 1.0]])
    sigma = torch.eye(2) * 0.5
    C = torch.tensor([[1.0, 0.1], [0.1, 1.0]])
    D = torch.eye(2)  # Assignment requires D = identity
    R = torch.tensor([[1.0, 0.3], [0.3, 1.0]]) * 10.0

    T = 0.5
    tau = 0.5
    gamma = 1.0
    N = 100
    time_grid = torch.linspace(0, T, N + 1)

    # Training parameters
    batch_size = 200
    n_epochs = 50
    hidden_dim = 512
    lr = 1e-3
    print_every = 10

    # === Initialize Soft LQR Controller (Fixed Policy π) ===
    soft_lqr = SoftLQRSolver(H, M, sigma, C, D, R, T, tau=tau, gamma=gamma, time_grid=time_grid)

    # === Initialize Value Network ===
    model = OnlyLinearValueNN(hidden_dim=512)

    # === Train Critic Network ===
    loss_list = train_critic_batch(
        model=model,
        controller=soft_lqr,
        time_grid=time_grid,
        tau=tau,
        n_epochs=n_epochs,
        batch_size=batch_size,
        lr=lr,
        print_every=print_every
    )

    plot_value_function_1d(model, soft_lqr, t_scalar=0.2)


if __name__ == "__main__":
    main_exercise3_critic_loss()

if __name__ == "__main__":
    main_exercise3_comparsion()
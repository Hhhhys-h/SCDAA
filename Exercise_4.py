from logging import logProcesses
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import StepLR

from Exercise_2 import LQRSolver, SoftLQRSolver

def simulate_trajectory(controller, x0, T, time_grid, use_sampling=False, fixed_brownian=None):
    """
    Simulate system trajectory

    Args:
        controller: LQRSolver or SoftLQRSolver object
        x0: Initial state vector
        T: Termination time
        time_grid: Time grid
        use_sampling: Whether to sample from the distribution (used for Soft LQR)
        fixed_brownian: Predefined Brownian motion (used for comparison)

    Returns:
        times, states, controls, dW
    """
    # Initialize arrays
    n_steps = len(time_grid) - 1
    states = torch.zeros((len(time_grid), 2))
    controls = torch.zeros((n_steps, 2))

    # Set initial state
    states[0] = x0

    # Calculate time step
    dt = (time_grid[1] - time_grid[0]).item()

    # Generate or use predefined Brownian motion
    if fixed_brownian is None:
        # Generate standard Brownian increments
        dW = torch.randn(n_steps, controller.sigma.shape[1]) * torch.sqrt(torch.tensor(dt))
    else:
        dW = fixed_brownian

    # Simulate trajectory
    for i in range(n_steps):
        t_i = time_grid[i].reshape(1)
        x_i = states[i].reshape(1, 2)

        # Calculate control action
        if use_sampling and hasattr(controller, 'sample_control'):
            a_i = controller.sample_control(t_i, x_i)[0]
        else:
            a_i = controller.optimal_control(t_i, x_i)[0]

        controls[i] = a_i

        # Update state (Explicit Euler method)
        drift = controller.H @ states[i] + controller.M @ a_i
        diffusion = controller.sigma @ dW[i]
        states[i + 1] = states[i] + drift * dt + diffusion

    return time_grid, states, controls, dW


def compare_trajectories(standard_lqr, soft_lqr, starting_points, T, time_grid):
    """Compare the trajectories of standard LQR and soft LQR"""
    for i, x0 in enumerate(starting_points):
        print(f"\nSimulating starting point {i + 1}/4: {x0.tolist()}")

        # Create figure
        plt.figure(figsize=(10, 8))

        # Generate Brownian motion (used by both methods)
        n_steps = len(time_grid) - 1
        dt = (time_grid[1] - time_grid[0]).item()
        np.random.seed(i)  # Use a different seed for each starting point, but ensure both methods use the same Brownian motion
        dW = torch.tensor(np.random.normal(0, np.sqrt(dt), (n_steps, 2)), dtype=torch.float32)

        # Simulate standard LQR trajectory
        print("  Simulating standard LQR trajectory...")
        _, states_std, _, _ = simulate_trajectory(
            standard_lqr, x0, T, time_grid, fixed_brownian=dW)

        # Simulate soft LQR trajectory
        print("  Simulating soft LQR trajectory...")
        _, states_soft, _, _ = simulate_trajectory(
            soft_lqr, x0, T, time_grid, use_sampling=True, fixed_brownian=dW)

        # Plot trajectories
        plt.plot(states_std[:, 0].numpy(), states_std[:, 1].numpy(),
                 'b-', linewidth=2, label='Standard LQR')
        plt.plot(states_soft[:, 0].numpy(), states_soft[:, 1].numpy(),
                 'r-', linewidth=2, label='Soft LQR (τ=0.1, γ=10)')
        plt.scatter([x0[0].item()], [x0[1].item()],
                    c='k', s=100, marker='o', label='Starting Point')

        plt.title(f'Trajectory starting from point ({x0[0].item()}, {x0[1].item()})')
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.legend()
        plt.grid(True)

        # Save figure
        save_path = f'Exercise_4_results/trajectory_from_{x0[0].item()}_{x0[1].item()}.png'
        plt.savefig(save_path)
        plt.show()

    print("\nAll simulations completed!")

# def main_exercise_4(): 
# # === Add call to compare_trajectories in main ===
#     # === Define model parameters ===
#     H = torch.tensor([[1.0, 1.0], [0.0, 1.0]]) * 0.5
#     M = torch.tensor([[1.0, 1.0], [0.0, 1.0]])
#     sigma = torch.eye(2) * 0.5
#     C = torch.tensor([[1.0, 0.1], [0.1, 1.0]]) * 1.0
#     D = torch.tensor([[1.0, 0.1], [0.1, 1.0]]) * 0.1
#     R = torch.tensor([[1.0, 0.3], [0.3, 1.0]]) * 10.0
#     T = 0.5
#     time_grid = torch.linspace(0, T, 1000)

#     # === Initialize standard and soft LQR ===
#     standard_lqr = LQRSolver(H, M, sigma, C, D, R, T, time_grid)
#     soft_lqr = SoftLQRSolver(H, M, sigma, C, D, R, T, tau=0.1, gamma=10.0, time_grid=time_grid)

#     # === Set starting points ===
#     starting_points = [
#         torch.tensor([2.0, 2.0]),
#         torch.tensor([2.0, -2.0]),
#         torch.tensor([-2.0, -2.0]),
#         torch.tensor([-2.0, 2.0]),
#     ]

#     # === Call the trajectory comparison function ===
#     compare_trajectories(standard_lqr, soft_lqr, starting_points, T, time_grid)

# ---- Policy Network from Appendix ----
class PolicyNeuralNetwork(nn.Module):
    def __init__(self, hidden_size, d, device="cpu"):
        super(PolicyNeuralNetwork, self).__init__()
        self.hidden_layer1 = nn.Linear(1, hidden_size, device=device)
        self.hidden_layer2 = nn.Linear(hidden_size, hidden_size, device=device)

        # Output for phi
        self.phi_output = nn.Linear(hidden_size, d * d).to(device)

        # Output for L matrix for Sigma
        self.sigma_output_L = nn.Linear(hidden_size, d * (d + 1) // 2).to(device)

        torch.nn.init.xavier_uniform_(self.sigma_output_L.weight)
        torch.nn.init.xavier_uniform_(self.phi_output.weight)
        #torch.nn.init.zeros_(self.sigma_output_L.bias)
        #torch.nn.init.zeros_(self.phi_output.bias)
        #torch.nn.init.xavier_normal_(self.sigma_output_L.weight)
        #torch.nn.init.xavier_normal_(self.phi_output.weight)

        self.d = d
        self.tri_indices = torch.tril_indices(self.d, self.d).to(device)

    def forward(self, t, x):
        t = t.view(-1, 1)  # Ensure t is a column vector
        hidden = torch.relu(self.hidden_layer1(t))
        hidden = torch.sigmoid(self.hidden_layer2(hidden))

        # Compute phi
        phi_flat = self.phi_output(hidden)
        phi = phi_flat.view(-1, self.d, self.d)

        # Compute Sigma
        L_flat = self.sigma_output_L(hidden).squeeze(0)

        L = torch.zeros(self.d, self.d, device=L_flat.device)
        L[self.tri_indices[0], self.tri_indices[1]] = L_flat
        Sigma = L @ L.T + 1e-3 * torch.eye(self.d, device=L.device) # Ensure PSD

        # Compute mean
        mean = torch.bmm(phi, x.unsqueeze(-1)).squeeze(-1)

        return mean, Sigma

# ---- Training loop ----
def train_actor_only(policy_net, lqr_solver, epochs=50, episodes=100, tau=0.5, T=1, N=20):
    optimizer = optim.Adam(policy_net.parameters(), lr=1e-4)
    # scheduler = StepLR(optimizer, step_size=10, gamma=0.5)
    loss_history = []

    for epoch in range(epochs):

        epoch_loss = 0.0
        epoch_cost = 0.0

        for ep in range(episodes):
            loss = 0
            x0 = torch.empty(2).uniform_(-2, 2)
            dt = T / N
            t_grid = torch.linspace(0, T, N + 1)

            d = x0.shape[0]  # State dimension
            x = x0.clone()  # Current state
            traj = [(0.0, x0.clone())]  # Save the trajectory with each time point and state
            actions, log_probs, costs, deltas = [], [], [], []

            for n in range(N):
                t = t_grid[n]
                t1 = t_grid[n + 1]

                # Output control distribution parameters, sample an action, calculate log probability for policy gradient
                mean, cov = policy_net(t, x.unsqueeze(0))
                dist = torch.distributions.MultivariateNormal(mean.squeeze(0), cov)
                a = dist.sample()
                logp = dist.log_prob(a)

                # Simulate environment dynamics, corresponding to SDE formula (17)
                dx = (
                    (lqr_solver.H @ x + lqr_solver.M @ a) * dt +
                    lqr_solver.sigma @ torch.randn(d) * np.sqrt(dt)
                )
                x_new = x + dx

                # Calculate instantaneous cost
                cost = x @ lqr_solver.C @ x + a @ lqr_solver.D @ a

                # Use the analytical solution from ex2 to compute the value function, estimate the value function difference delta_v
                v_t1 = lqr_solver.value_function(t1.expand(1), x_new.unsqueeze(0)).detach()
                v_t = lqr_solver.value_function(t.expand(1), x.unsqueeze(0)).detach()
                delta_v = v_t1 - v_t

                # Update trajectory, record control actions, log probabilities, costs, deltas, and update state
                traj.append((t_grid[n + 1].item(), x_new.clone()))
                actions.append(a)
                log_probs.append(logp)
                costs.append(cost)
                deltas.append(delta_v)
                x = x_new  # State update

            # === Loss accumulation based on Equation (8) ===
            # deltas = torch.tensor(deltas)
            # deltas = (deltas - deltas.mean()) / (deltas.std() + 1e-6)

            for logp, delta_v, cost in zip(log_probs, deltas, costs):
                weight = delta_v + (cost + tau * logp.detach()) * dt
                loss += -logp * weight

            optimizer.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=2.0)
            optimizer.step()
            epoch_loss += loss.item()
            epoch_cost += np.mean(costs)

            # Calculate gradients
            # total_norm = torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=1.0)
            # total_grad_norm += total_norm.item()

            # Calculate policy entropy
            # dist = torch.distributions.MultivariateNormal(mean.squeeze(0), cov)
            # entropy = dist.entropy().mean()
            # total_entropy += entropy.item()

        avg_epoch_loss = epoch_loss / episodes
        loss_history.append(avg_epoch_loss)
        # scheduler.step()

        if epoch % 5 == 0:
            print(f"[Epoch {epoch}] Avg loss: {avg_epoch_loss:.2e}")
            print(f"logp: {logp.item():.2f}, entropy: {dist.entropy().item():.2f}")
            print(f"x norm: {x.norm().item():.2f}, a norm: {a.norm().item():.2f}")

    return loss_history


# ---- Rollout Comparison ----
def rollout_trajectory(policy_net, lqr_solver, x0, T=1.0, N=20):
    dt = T / N
    x = x0.clone()
    #x_actor = x0.clone()
    #x_opt = x0.clone()
    #traj_actor = [x0.clone()]
    #traj_opt = [x0.clone()]
    cost_actor = []
    cost_opt = []

    for n in range(N):
        t = torch.tensor(n * dt).unsqueeze(0)

        mean, cov = policy_net(t, x.unsqueeze(0))
        dist = torch.distributions.MultivariateNormal(mean.squeeze(0), cov)
        a = dist.sample()

        dx_actor = (lqr_solver.H @ x + lqr_solver.M @ a) * dt
        x_actor = x + dx_actor
        #traj_actor.append(x_actor.clone())
        cost_actor.append((x @ lqr_solver.C @ x + a @ lqr_solver.D @ a).item())

        a_opt = lqr_solver.sample_control(t, x.unsqueeze(0))[0]  # Use Exercise 2 optimal control
        dx_opt = (lqr_solver.H @ x + lqr_solver.M @ a_opt) * dt
        x_opt = x + dx_opt
        #traj_opt.append(x_opt.clone())
        cost_opt.append((x @ lqr_solver.C @ x + a_opt @ lqr_solver.D @ a_opt).item())

        x = x_actor

    return cost_actor, cost_opt



# ---- Main ----
def main_exercise4_loss():
    # Problem parameters
    H = torch.tensor([[1.0, 0.1], [0.8, 0.3]])
    M = torch.tensor([[0.3, 0.8], [0.1, 1.0]])
    sigma = torch.eye(2) * 0.5
    C = torch.eye(2) * 10
    D = torch.eye(2) * 0.005
    R = torch.eye(2) * 100
    T = 1.0
    tau = 0.2
    gamma = 10

    solver = SoftLQRSolver(H, M, sigma, C, D, R, T, tau = tau, gamma = gamma)
    policy = PolicyNeuralNetwork(hidden_size=512, d=2, device="cpu")

    losses = train_actor_only(policy, solver, epochs=50, episodes=100, tau=0.2, T=1.0, N=20)

    # Plot
    plt.plot(losses, label='Loss')
    plt.yscale("log")
    plt.legend()
    plt.title("Actor-only Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (log scale)")
    plt.grid(True)
    plt.tight_layout()
    save_path = f'Exercise_4_results/Actor_only_Training_Loss.png'
    plt.savefig(save_path)
    plt.show()

def main_exercise4_cost():
    # === Initial points ===
    initial_states = torch.tensor([[2., 2.], [2., -2.], [-2., -2.], [-2., 2.]])
    colors = ['b', 'g', 'c', 'm']

    # Problem parameters
    H = torch.tensor([[1.0, 0.1], [0.8, 0.3]])
    M = torch.tensor([[0.3, 0.8], [0.1, 1.0]])
    sigma = torch.eye(2) * 0.5
    C = torch.eye(2) * 10
    D = torch.eye(2) * 0.005
    R = torch.eye(2) * 100
    T = 1.0
    tau = 0.2
    gamma = 10

    solver = SoftLQRSolver(H, M, sigma, C, D, R, T, tau = tau, gamma = gamma)
    policy = PolicyNeuralNetwork(hidden_size=512, d=2, device="cpu")

    # === Plot ===
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    axs = axs.flatten()

    for i, x0 in enumerate(initial_states):
        cost_actor, cost_opt = rollout_trajectory(policy, solver, x0)
        t = torch.linspace(0, 1, len(cost_actor))

        axs[i].plot(t, torch.cumsum(torch.tensor(cost_actor), dim=0), '--', color=colors[i], label='Actor')
        axs[i].plot(t, torch.cumsum(torch.tensor(cost_opt), dim=0), '-', color=colors[i], label='Optimal')
        axs[i].set_title(f"Cumulative Cost from Initial State {x0.tolist()}")
        axs[i].set_xlabel("Time")
        axs[i].set_ylabel("Cost")
        axs[i].grid()
        axs[i].legend()

    plt.tight_layout()
    save_path = f'Exercise_4_results/Cumulative_Cost_from_Initial_State {x0.tolist()}.png'
    plt.savefig(save_path)
    plt.show()


if __name__ == "__main__":
    main_exercise4_loss()

if __name__ == "__main__":
    main_exercise4_cost()
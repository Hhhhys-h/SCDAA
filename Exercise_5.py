import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

from torch.distributions import MultivariateNormal
from copy import deepcopy
from Exercise_2 import LQRSolver, SoftLQRSolver

class OnlyLinearValueNN(nn.Module):
    def __init__(self, hidden_dim=512, device=torch.device("cpu")):
        super(OnlyLinearValueNN, self).__init__()
        self.device = device

        # Add hidden layers for the matrix parameters
        self.matrix_network = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * 2)
        ).to(device)

        # Add hidden layers for the bias parameters
        self.offset_network = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        ).to(device)

    def forward(self, t):
        """
        Input: Time t (batch_size, 1)
        Output:
            - matrix: A matrix with shape (batch_size, 2, 2)
            - offset: A bias with shape (batch_size, 1)
        """
        matrix_elements = self.matrix_network(t)
        matrix = matrix_elements.view(-1, 2, 2)

        # Ensure the matrix is symmetric positive definite
        matrix = torch.bmm(matrix, matrix.transpose(1, 2)) + 1e-3 * torch.eye(2).to(matrix.device)

        # Calculate the bias
        offset = self.offset_network(t)

        return matrix, offset

def evaluate_value(model, t_tensor, x_tensor):
    """Compute the value function v(t, x) = x^T S(t) x + b(t)"""
    matrix, offset = model(t_tensor)  # Get S(t) and b(t)
    quad_term = torch.einsum('bi,bij,bj->b', x_tensor, matrix, x_tensor)
    return quad_term + offset.view(-1)


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

            for logp, delta_v, cost in zip(log_probs, deltas, costs):
                weight = delta_v + (cost + tau * logp.detach()) * dt
                loss += -logp * weight

            optimizer.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=2.0)
            optimizer.step()
            epoch_loss += loss.item()
            epoch_cost += np.mean(costs)

        avg_epoch_loss = epoch_loss / episodes
        loss_history.append(avg_epoch_loss)
        # scheduler.step()

        if epoch % 5 == 0:
            print(f"[Epoch {epoch}] Avg loss: {avg_epoch_loss:.2e}")
            print(f"logp: {logp.item():.2f}, entropy: {dist.entropy().item():.2f}")
            print(f"x norm: {x.norm().item():.2f}, a norm: {a.norm().item():.2f}")

    return loss_history

def compute_target(costs, log_probs, g_T, dt, tau):
    """
    Construct the target values for each trajectory according to Algorithm 2, formula (14).
    """
    N = costs.shape[0]
    cumulative = torch.zeros(N, dtype=costs.dtype, device=costs.device)
    total = 0.0
    for n in reversed(range(N)):
        total += (costs[n] + tau * log_probs[n]) * dt
        cumulative[n] = total
    cumulative += torch.tensor(g_T, dtype=cumulative.dtype, device=cumulative.device)
    return cumulative



# === Module Description ===
# EX3: OnlyLinearValueNN, compute_target() have been implemented
# EX4: PolicyNeuralNetwork has been implemented
# SoftLQRSolver: Provides value_function and environment dynamics

# === 1️⃣ Initialize Parameters and Components ===
def train_actor_critic(policy_net, value_net, solver,
                       T=1.0, N=20, tau=0.1, gamma=10.0,
                       epochs=50, episodes_per_epoch=500,
                       actor_lr=1e-4, critic_lr=1e-1,
                       device="cpu"):

    dt = T / N
    time_grid = torch.linspace(0, T, N + 1)

    actor_optimizer = optim.Adam(policy_net.parameters(), lr=actor_lr)
    critic_optimizer = optim.Adam(value_net.parameters(), lr=critic_lr)

    actor_losses, critic_losses = [], []

    for epoch in range(epochs):
        total_actor_loss = 0.0
        total_critic_loss = 0.0

        for ep in range(episodes_per_epoch):
            x0 = torch.tensor([-2.0, 2.0], device=device)
            x = x0.clone()

            t_seq = []
            x_seq = []
            a_seq = []
            f_seq = []
            logp_seq = []

            # === 2️⃣ Environment Interaction, Sample Trajectory (Algorithm 3, line 2) ===
            for n in range(N):
                t = time_grid[n].to(device)
                t_next = time_grid[n + 1].to(device)

                mean, Sigma = policy_net(t.view(1), x.view(1, -1))
                dist = MultivariateNormal(mean[0], Sigma)
                a = dist.sample()
                logp = dist.log_prob(a)

                drift = solver.H @ x + solver.M @ a
                diffusion = solver.sigma @ torch.randn(2, device=device) * np.sqrt(dt)
                x_next = x + drift * dt + diffusion

                cost = x @ solver.C @ x + a @ solver.D @ a

                # Store data
                t_seq.append(t)
                x_seq.append(x)
                a_seq.append(a)
                f_seq.append(cost)
                logp_seq.append(logp)

                x = x_next

            # ✅ Add the last t_N and x_N to prevent out-of-bounds access             
            t_seq.append(time_grid[N].to(device))  # t_N             
            x_seq.append(x.clone())  # x_N

            # Terminal term g_T
            g_T = x @ solver.R @ x

            # === 3️⃣ Update Critic Loss (Formula (15), from EX3) ===
            critic_loss = 0.0
            targets = compute_target(torch.stack(f_seq), torch.stack(logp_seq), g_T, dt, tau)

            for n in range(N):
                t_input = t_seq[n].view(1, 1)
                x_input = x_seq[n].view(1, -1)
                v_pred = evaluate_value(value_net, t_input, x_input)[0]
                critic_loss += (v_pred - targets[n].detach()) ** 2

            critic_optimizer.zero_grad()
            critic_loss.backward()
            critic_optimizer.step()

            total_critic_loss += critic_loss.item()

            # === 4️⃣ Update Actor Loss (Formula (13), from EX4) ===
            actor_loss = 0.0
            for n in range(N):
                t_input = t_seq[n].view(1, 1)
                x_input = x_seq[n].view(1, -1)

                mean, Sigma = policy_net(t_input, x_input)
                dist = MultivariateNormal(mean[0], Sigma)

                logp = dist.log_prob(a_seq[n])  # Do not detach!

                v_next = evaluate_value(value_net, t_seq[n + 1].view(1, 1), x_seq[n + 1]
                                        .view(1, -1)).detach()
                v_now = evaluate_value(value_net, t_input, x_input).detach()
                delta_v = v_next - v_now

                weight = delta_v + (f_seq[n] + tau * logp) * dt
                actor_loss += -logp * weight

            actor_optimizer.zero_grad()
            actor_loss.backward()
            actor_optimizer.step()

            total_actor_loss += actor_loss.item()

        # === 5️⃣ Record the loss for each epoch for plotting ===
        actor_losses.append(total_actor_loss / episodes_per_epoch)
        critic_losses.append(total_critic_loss / episodes_per_epoch)

        if epoch % 5 == 0:
            print(f"[Epoch {epoch}] Actor Loss: {actor_losses[-1]:.2e}, Critic Loss: {critic_losses[-1]:.2e}")

    return actor_losses, critic_losses


# === 🖼 Figures 2 & 3: Policy Trajectory vs. Soft LQR Optimal Trajectory ===
def compare_trajectories(policy_net, solver, time_grid, epoch_label, device="cpu"):
    N = len(time_grid) - 1
    dt = (time_grid[1] - time_grid[0]).item()

    # Initial state
    x0 = torch.tensor([-2.0, 2.0], device=device)

    # Simulate policy trajectory
    x_policy = [x0.clone()]
    x = x0.clone()
    for n in range(N):
        t = time_grid[n].to(device)
        mean, Sigma = policy_net(t.view(1), x.view(1, -1))
        dist = MultivariateNormal(mean[0], Sigma)
        a = dist.sample()
        drift = solver.H @ x + solver.M @ a
        diffusion = solver.sigma @ torch.randn(2, device=device) * np.sqrt(dt)
        x = x + drift * dt + diffusion
        x_policy.append(x.clone())

    x_policy = torch.stack(x_policy)

    # Simulate Soft LQR optimal trajectory (analytical policy)
    x_soft = [x0.clone()]
    x = x0.clone()
    for n in range(N):
        t = time_grid[n].view(1)
        x_input = x.view(1, -1)
        a = solver.sample_control(t, x_input)[0]
        drift = solver.H @ x + solver.M @ a
        diffusion = solver.sigma @ torch.randn(2, device=device) * np.sqrt(dt)
        x = x + drift * dt + diffusion
        x_soft.append(x.clone())

    x_soft = torch.stack(x_soft)

    # === Plotting ===
    plt.figure(figsize=(6, 6))
    plt.plot(x_soft[:, 0].cpu(), x_soft[:, 1].cpu(), label="Soft LQR (optimal)", linewidth=2)
    plt.plot(x_policy[:, 0].cpu(), x_policy[:, 1].cpu(), label=f"Actor Policy (epoch {epoch_label})", linestyle="--")
    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$")
    plt.title("Trajectory Comparison")
    plt.legend()
    plt.grid(True)
    plt.axis("equal")
    plt.tight_layout()
    plt.show()

def main_exercise5():
    # === Parameter Settings (Section 5 of the Paper) ===
    H = torch.tensor([[1.0, 0.8], [0.1, 0.3]])
    M = torch.tensor([[0.3, 0.1], [0.8, 1.0]])
    sigma = 0.5 * torch.eye(2)
    C = 10 * torch.eye(2)
    D = 0.005 * torch.eye(2)
    R = 100 * torch.eye(2)
    T = 1.0
    N = 20
    tau = 0.1
    gamma = 10.0
    device = torch.device("cpu")

    time_grid = torch.linspace(0, T, N + 1)

    # === Initialize Policy Network and Value Network ===
    policy_net = PolicyNeuralNetwork(hidden_size=256, d=2, device=device)
    value_net = OnlyLinearValueNN(hidden_dim=512, device=device)
    solver = SoftLQRSolver(H, M, sigma, C, D, R, T, tau=tau, gamma=gamma, time_grid=time_grid)

    # === Train Actor-Critic ===
    actor_losses, critic_losses = train_actor_critic(
        policy_net, value_net, solver,
        T=T, N=N, tau=tau, gamma=gamma,
        epochs=50, episodes_per_epoch=100,
        actor_lr=1e-4, critic_lr=1e-3,
        device=device
    )

    # === Figure 1: Loss Curves ===
    plt.figure(figsize=(8, 5))
    plt.plot(actor_losses, label='Actor Loss')
    plt.plot(critic_losses, label='Critic Loss')
    plt.yscale('log')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (log scale)')
    plt.title('Actor and Critic Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    save_path = f'Exercise_5_results/Actor_and_Critic_Loss.png'
    plt.savefig(save_path)
    plt.show()

if __name__ == "__main__":
    main_exercise5()
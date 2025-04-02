""" 
This file is used to output the required charts in the assignment
"""

pip install -r requirements.txt

"""Exercise 1.1"""
def main_exercise1():
    # Define parameters
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
    
    # Create time grid
    time_grid = torch.linspace(0, T, 100)
    
    # Create an instance of LQR solver
    lqr_solver = LQRSolver(H, M, sigma, C, D, R, T, time_grid)
    
    # Test value function and optimal control
    test_point_t = torch.tensor([0.0])
    test_point_x = torch.tensor([[1.0, 1.0]])
    
    value = lqr_solver.value_function(test_point_t, test_point_x)
    control = lqr_solver.optimal_control(test_point_t, test_point_x)
    
    print(f"Value function at t=0, x=[1,1]: {value.item()}")
    print(f"Optimal control at t=0, x=[1,1]: {control.squeeze(0).tolist()}")


if __name__ == "__main__":
    main_exercise1()


"""Exercise 2"""
def compare_trajectories(standard_lqr, soft_lqr, starting_points, T, time_grid):
    """比较标准LQR和软LQR的轨迹"""
    for i, x0 in enumerate(starting_points):
        print(f"\n模拟起始点 {i+1}/4: {x0.tolist()}")
        
        # 创建图形
        plt.figure(figsize=(10, 8))
        
        # 生成布朗运动 (两种方法共用)
        n_steps = len(time_grid) - 1
        dt = (time_grid[1] - time_grid[0]).item()
        np.random.seed(i)  # 每个起始点使用不同的种子，但确保两种方法使用相同的布朗运动
        dW = torch.tensor(np.random.normal(0, np.sqrt(dt), (n_steps, 2)), dtype=torch.float32)
        
        # 模拟标准LQR轨迹
        print("  模拟标准LQR轨迹...")
        _, states_std, _, _ = simulate_trajectory(
            standard_lqr, x0, T, time_grid, fixed_brownian=dW)
        
        # 模拟软LQR轨迹
        print("  模拟软LQR轨迹...")
        _, states_soft, _, _ = simulate_trajectory(
            soft_lqr, x0, T, time_grid, use_sampling=True, fixed_brownian=dW)
        
        # 绘制轨迹
        plt.plot(states_std[:, 0].numpy(), states_std[:, 1].numpy(), 
                'b-', linewidth=2, label='标准LQR')
        plt.plot(states_soft[:, 0].numpy(), states_soft[:, 1].numpy(), 
                'r-', linewidth=2, label='软LQR (τ=0.1, γ=10)')
        plt.scatter([x0[0].item()], [x0[1].item()], 
                   c='k', s=100, marker='o', label='起始点')
        
        plt.title(f'从起始点 ({x0[0].item()}, {x0[1].item()}) 出发的轨迹')
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.legend()
        plt.grid(True)
        
        # 保存图形
        plt.savefig(f'trajectory_from_{x0[0].item()}_{x0[1].item()}.png')
        plt.show()
    
    print("\n所有模拟完成!")


# === Add call to compare_trajectories in main ===
if __name__ == "__main__":
    # === 定义模型参数 ===
    H = torch.tensor([[1.0, 1.0], [0.0, 1.0]]) * 0.5
    M = torch.tensor([[1.0, 1.0], [0.0, 1.0]])
    sigma = torch.eye(2) * 0.5
    C = torch.tensor([[1.0, 0.1], [0.1, 1.0]]) * 1.0
    D = torch.tensor([[1.0, 0.1], [0.1, 1.0]]) * 0.1
    R = torch.tensor([[1.0, 0.3], [0.3, 1.0]]) * 10.0
    T = 0.5
    time_grid = torch.linspace(0, T, 1000)

    # === 初始化 strict 和 soft LQR ===
    standard_lqr = LQRSolver(H, M, sigma, C, D, R, T, time_grid)
    soft_lqr = SoftLQRSolver(H, M, sigma, C, D, R, T, tau=0.1, gamma=10.0, time_grid=time_grid)

    # === 设置起始点 ===
    starting_points = [
        torch.tensor([2.0, 2.0]),
        torch.tensor([2.0, -2.0]),
        torch.tensor([-2.0, -2.0]),
        torch.tensor([-2.0, 2.0]),
    ]

    # === 调用轨迹比较函数 ===
    compare_trajectories(standard_lqr, soft_lqr, starting_points, T, time_grid)


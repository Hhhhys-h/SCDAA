pip install -r requirements.txt

"""
    Exercise 2
"""
class LQRSolver:
    """
    线性二次调节器求解器：满足Exercise 1.1要求
    """
    def __init__(self, H, M, sigma, C, D, R, T, time_grid=None):
        """
        初始化LQR求解器
        
        Args:
            H, M, sigma, C, D, R: LQR问题的矩阵
            T: 终止时间
            time_grid: 时间网格 (numpy array或torch tensor)
        """
        # 确保所有输入都是torch张量
        self.H = torch.tensor(H, dtype=torch.float32) if not isinstance(H, torch.Tensor) else H
        self.M = torch.tensor(M, dtype=torch.float32) if not isinstance(M, torch.Tensor) else M
        self.sigma = torch.tensor(sigma, dtype=torch.float32) if not isinstance(sigma, torch.Tensor) else sigma
        self.C = torch.tensor(C, dtype=torch.float32) if not isinstance(C, torch.Tensor) else C
        self.D = torch.tensor(D, dtype=torch.float32) if not isinstance(D, torch.Tensor) else D
        self.R = torch.tensor(R, dtype=torch.float32) if not isinstance(R, torch.Tensor) else R
        self.T = T
        
        # 计算D的逆矩阵
        self.D_inv = torch.inverse(self.D)
        
        # 创建或使用时间网格
        if time_grid is None:
            self.time_grid = torch.linspace(0, T, 100)
        else:
            # 确保time_grid是torch张量
            self.time_grid = torch.tensor(time_grid, dtype=torch.float32) if not isinstance(time_grid, torch.Tensor) else time_grid
        
        # 求解Riccati方程
        self.S_values = self.solve_riccati_ode()
    
    def riccati_ode(self, t, S_flat):
        """定义Riccati ODE"""
        # 将展平的S重塑为2x2矩阵
        S = S_flat.reshape(2, 2)
        
        # 转换为numpy进行计算
        S_np = S
        H_np = self.H.numpy()
        M_np = self.M.numpy()
        D_inv_np = np.linalg.inv(self.D.numpy())
        C_np = self.C.numpy()
        
        # 计算Riccati方程的右侧
        term1 = S_np @ M_np @ D_inv_np @ M_np.T @ S_np
        term2 = H_np.T @ S_np
        term3 = S_np @ H_np
        dSdt = term1 - term2 - term3 - C_np
        
        return dSdt.flatten()
    
    def solve_riccati_ode(self):
        """求解Riccati ODE"""
        # 将时间网格转换为numpy数组
        t_grid = self.time_grid.numpy()
        
        # 终端条件: S(T) = R
        S_T = self.R.numpy().flatten()
        
        # 求解ODE (逆向求解)
        sol = solve_ivp(
            self.riccati_ode,
            [self.T, 0],  # 从T到0逆向求解
            S_T,
            t_eval=np.flip(t_grid),  # 按逆序评估
            method='RK45',
            rtol=1e-13,
            atol=1e-15
        )
        
        # 将解转换为torch张量并重塑为矩阵
        S_values = []
        for i in range(sol.y.shape[1]):
            S = sol.y[:, i].reshape(2, 2)
            S_values.append(torch.tensor(S, dtype=torch.float32))
        
        # 反转顺序，使其与时间网格对应
        return S_values[::-1]
    
    def find_nearest_time_index(self, t):
        """找到最接近t的时间索引"""
        if t >= self.T:
            return len(self.time_grid) - 1
        
        if t <= 0:
            return 0
        
        # 找到最接近的索引
        time_array = self.time_grid.numpy()
        idx = np.searchsorted(time_array, t, side='right') - 1
        return idx
    
    def value_function(self, t, x):
        """
        计算t时刻在状态x的值函数
        
        Args:
            t: 1D torch张量，表示时间点
            x: 2D torch张量，表示状态点 (batch_size x 2)
            
        Returns:
            1D torch张量，表示值函数
        """
        batch_size = x.shape[0]
        values = torch.zeros(batch_size)
        
        for i in range(batch_size):
            # 找到最接近的时间点
            t_i = t[i].item()
            idx = self.find_nearest_time_index(t_i)
            
            # 获取对应的S矩阵
            S_t = self.S_values[idx]
            
            # 计算二次型 x^T S x
            x_i = x[i]
            values[i] = x_i @ S_t @ x_i
            
            # 计算积分项 (如果需要)
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
        计算最优控制
        
        Args:
            t: 1D torch张量，表示时间点
            x: 2D torch张量，表示状态点 (batch_size x 2)
            
        Returns:
            2D torch张量，表示最优控制 (batch_size x 2)
        """
        batch_size = x.shape[0]
        controls = torch.zeros((batch_size, 2))
        
        for i in range(batch_size):
            # 找到最接近的时间点
            t_i = t[i].item()
            idx = self.find_nearest_time_index(t_i)
            
            # 获取对应的S矩阵
            S_t = self.S_values[idx]
            
            # 计算最优控制: a(t,x) = -D^(-1) M^T S(t) x
            controls[i] = -self.D_inv @ self.M.T @ S_t @ x[i]
        
        return controls


class SoftLQRSolver(LQRSolver):
    """
    软线性二次调节器求解器：满足Exercise 2.1要求
    """
    def compute_value_constant(self):
        """
        计算 Soft LQR 的值函数常数项 C_{D,τ,γ}，对应文档中公式 (22)：
        C = -τ ( (m/2)·log(τ) - m·log(γ) + 0.5·log(det(Σ)) )
        """
        d = self.D.shape[0]
        Sigma = torch.inverse(self.D + (self.tau / (2 * self.gamma ** 2)) * torch.eye(d))
        det_sigma = torch.det(Sigma)
        C = -self.tau * (0.5 * d * torch.log(torch.tensor(self.tau)) - d * torch.log(torch.tensor(self.gamma)) + 0.5 * torch.log(det_sigma))
        return C.item()

    def __init__(self, H, M, sigma, C, D, R, T, time_grid=None, tau=0.1, gamma=10.0):
        """
        初始化软LQR求解器 - 完全重新实现，不调用父类初始化
        
        Args:
            H, M, sigma, C, D, R: LQR问题的矩阵
            T: 终止时间
            time_grid: 时间网格 (numpy array或torch tensor)
            tau: 熵正则化强度
            gamma: 先验正态分布方差
        """
        # 确保所有输入都是torch张量
        self.H = torch.tensor(H, dtype=torch.float32) if not isinstance(H, torch.Tensor) else H
        self.M = torch.tensor(M, dtype=torch.float32) if not isinstance(M, torch.Tensor) else M
        self.sigma = torch.tensor(sigma, dtype=torch.float32) if not isinstance(sigma, torch.Tensor) else sigma
        self.C = torch.tensor(C, dtype=torch.float32) if not isinstance(C, torch.Tensor) else C
        self.D = torch.tensor(D, dtype=torch.float32) if not isinstance(D, torch.Tensor) else D
        self.R = torch.tensor(R, dtype=torch.float32) if not isinstance(R, torch.Tensor) else R
        self.T = T
        
        # 计算D的逆矩阵
        self.D_inv = torch.inverse(self.D)
        
        # 存储软LQR特定参数
        self.tau = tau
        self.gamma = gamma
        
        # 创建或使用时间网格
        if time_grid is None:
            self.time_grid = torch.linspace(0, T, 100)
        else:
            # 确保time_grid是torch张量
            self.time_grid = torch.tensor(time_grid, dtype=torch.float32) if not isinstance(time_grid, torch.Tensor) else time_grid
        
        # 求解Riccati方程
        self.S_values = self.solve_riccati_ode()
    
    def riccati_ode(self, t, S_flat):
        """定义软LQR的Riccati ODE，包含熵正则化项"""
        # 将展平的S重塑为2x2矩阵
        S = S_flat.reshape(2, 2)
        
        # 转换为numpy进行计算
        S_np = S
        H_np = self.H.numpy()
        M_np = self.M.numpy()
        C_np = self.C.numpy()
        sigma_np = self.sigma.numpy()

        Sigma = np.linalg.inv(self.D.numpy() + (self.tau / (2 * self.gamma**2)) * np.eye(2))
        
        # 计算标准Riccati方程的右侧
        term1 = S_np.T @ M_np @ Sigma @ M_np.T @ S
        term2 = H_np.T @ S
        term3 = S @ H_np

        # 删除 entropy_term，改用文档中的控制修正项
        sigma_sigma_T = sigma_np @ sigma_np.T
                
        # 总的导数
        dSdt = term1 - term2 - term3 - C_np
        
        return dSdt.flatten()
    
    def control_distribution(self, t, x):
        """
        计算最优控制分布的参数
        
        Args:
            t: 1D torch张量，表示时间点
            x: 2D torch张量，表示状态点 (batch_size x 2)
            
        Returns:
            tuple: (means, covariances)，分别是均值和协方差
        """
        batch_size = x.shape[0]
        means = torch.zeros((batch_size, 2))
        covariances = []
        
        for i in range(batch_size):
            # 找到最接近的时间点
            t_i = t[i].item()
            idx = self.find_nearest_time_index(t_i)
            
            # 获取对应的S矩阵
            S_t = self.S_values[idx]
            
            # Soft LQR 均值
            means[i] = -torch.inverse(self.D + (self.tau / (2 * self.gamma ** 2)) * torch.eye(2)) @ self.M.T @ S_t @ x[i]  # Soft LQR 均值
            
            # Soft LQR 协方差
            cov = self.tau * (self.D + (self.tau / (2 * self.gamma ** 2)) * torch.eye(2))  # Soft LQR 协方差
            covariances.append(cov)
        
        return means, covariances
    
    def sample_control(self, t, x):
        """
        从控制分布中采样具体的控制动作
        
        Args:
            t: 1D torch张量，表示时间点
            x: 2D torch张量，表示状态点 (batch_size x 2)
            
        Returns:
            2D torch张量，表示采样的控制动作 (batch_size x 2)
        """
        means, covariances = self.control_distribution(t, x)
        batch_size = x.shape[0]
        samples = torch.zeros((batch_size, 2))
        
        for i in range(batch_size):
            # 将均值和协方差转换为numpy
            mean_np = means[i].numpy()
            cov_np = covariances[i].numpy()
            
            # 从多元正态分布采样
            sample = np.random.multivariate_normal(mean_np, cov_np)
            samples[i] = torch.tensor(sample, dtype=torch.float32)
        
        return samples


def simulate_trajectory(controller, x0, T, time_grid, use_sampling=True, fixed_brownian=None):
    """
    模拟系统轨迹
    
    Args:
        controller: LQRSolver或SoftLQRSolver对象
        x0: 初始状态向量
        T: 终止时间
        time_grid: 时间网格
        use_sampling: 是否从分布中采样(用于软LQR)
        fixed_brownian: 预定义的布朗运动(用于比较)
        
    Returns:
        times, states, controls, dW
    """
    # 初始化数组
    n_steps = len(time_grid) - 1
    states = torch.zeros((len(time_grid), 2))
    controls = torch.zeros((n_steps, 2))
    
    # 设置初始状态
    states[0] = x0
    
    # 计算时间步长
    dt = (time_grid[1] - time_grid[0]).item()
    
    # 生成或使用预定义的布朗运动
    if fixed_brownian is None:
        # 生成标准布朗运动增量
        dW = torch.randn(n_steps, controller.sigma.shape[1]) * torch.sqrt(torch.tensor(dt))
    else:
        dW = fixed_brownian
    
    # 模拟轨迹
    for i in range(n_steps):
        t_i = time_grid[i].reshape(1)
        x_i = states[i].reshape(1, 2)
        
        # 计算控制动作
        if use_sampling and hasattr(controller, 'sample_control'):
            a_i = controller.sample_control(t_i, x_i)[0]
        else:
            a_i = controller.optimal_control(t_i, x_i)[0]
        
        controls[i] = a_i
        
        # 更新状态(显式Euler方法)
        drift = controller.H @ states[i] + controller.M @ a_i
        diffusion = controller.sigma @ dW[i]
        states[i+1] = states[i] + drift * dt + diffusion
    
    return time_grid, states, controls, dW

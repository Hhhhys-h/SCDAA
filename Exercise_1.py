
import subprocess
import sys

def install_requirements():
    try:
        # 使用subprocess模块运行pip install命令
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "Requirements.txt"])
        print("依赖项安装完成。")
    except subprocess.CalledProcessError as e:
        print(f"安装依赖项时出错: {e}")
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
                # 预先计算sigma_sigma_T
                sigma_sigma_T = self.sigma @ self.sigma.T
                
                # 计算所有时间点的trace(sigma*sigma^T*S)
                traces = torch.tensor([torch.trace(sigma_sigma_T @ S) for S in self.S_values[idx:]])
                
                # 计算时间差
                dt = self.time_grid[idx+1:] - self.time_grid[idx:-1]
                
                # 使用梯形法则计算积分
                integral_term = torch.sum(0.5 * (traces[:-1] + traces[1:]) * dt)
                
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

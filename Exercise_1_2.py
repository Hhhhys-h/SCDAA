
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
    Exercise 1_2
"""
class LQRMonteCarlo:
    """
    用蒙特卡洛方法验证LQR求解器的结果: Exercise 1.2
    """
    def __init__(self, lqr_solver):
        """
        初始化蒙特卡洛求解器
        
        Args:
            lqr_solver: LQRSolver实例
        """
        self.lqr_solver = lqr_solver
    
    def monte_carlo_estimate(self, x0, num_steps, num_samples):
        """
        使用向量化的显式欧拉方法来估计LQR代价的期望值
        
        Args:
            x0: 初始状态 (2,) 的torch张量
            num_steps: 时间步数
            num_samples: 蒙特卡洛样本数
            
        Returns:
            估计值(平均成本, float)
        """
        print(f"[Vectorized MC] Starting simulation: num_steps={num_steps}, num_samples={num_samples}")
        
        dt = self.lqr_solver.T / num_steps
        time_points = torch.linspace(0, self.lqr_solver.T, num_steps + 1)
        
        # 1) 初始化所有样本的状态: shape = [num_samples, 2]
        states = x0.unsqueeze(0).repeat(num_samples, 1).clone()
        
        # 2) 累加每个样本的成本
        costs = torch.zeros(num_samples)
        
        # 3) 预先生成随机增量 dW: shape = [num_steps, num_samples, 2]
        dW = torch.randn(num_steps, num_samples, 2) * torch.sqrt(torch.tensor(dt))
        
        for n in range(num_steps):
            t_n = time_points[n].item()
            # 找到与 t_n 最接近的时间索引, 取对应的 S(t)
            t_idx = self.lqr_solver.find_nearest_time_index(t_n)
            S_t = self.lqr_solver.S_values[t_idx]  # [2, 2]
            
            # a(t,x) = -D^(-1) M^T S(t) x
            # states shape: [num_samples, 2]
            # S_t shape: [2,2], M shape: [2,2], D_inv shape: [2,2]
            a_n = -((states @ S_t) @ self.lqr_solver.M.T) @ self.lqr_solver.D_inv  # [num_samples, 2]
            
            # 过程成本: x^T C x + a^T D a
            cost_x = ((states @ self.lqr_solver.C) * states).sum(dim=1)
            cost_a = ((a_n    @ self.lqr_solver.D) * a_n   ).sum(dim=1)
            costs += (cost_x + cost_a) * dt
            
            # 显式欧拉更新
            drift = (states @ self.lqr_solver.H.T) + (a_n @ self.lqr_solver.M.T)
            # dW[n] shape: [num_samples, 2]
            # sigma shape: [2,2] => dW[n] @ sigma.T => [num_samples, 2]
            states = states + dt * drift + (dW[n] @ self.lqr_solver.sigma.T)
        
        # 终端成本: X_T^T R X_T
        cost_term = ((states @ self.lqr_solver.R) * states).sum(dim=1)
        costs += cost_term
        
        # 返回所有样本的平均成本
        return costs.mean().item()
    
    def convergence_study_time_steps(self, x0, num_steps_list, num_samples=10000):
        """
        研究时间步长的收敛性
        
        Args:
            x0: 初始状态 (2,)
            num_steps_list: 时间步长列表 (如 [2,4,8,...])
            num_samples: 每个步长的样本数量 (默认10000)
            
        Returns:
            结果字典
        """
        exact_value = self.lqr_solver.value_function(torch.tensor([0.0]), x0.unsqueeze(0)).item()
        print(f"Exact value from Riccati solver: {exact_value:.6f}")
        
        estimated_values = []
        errors = []
        
        for N in num_steps_list:
            print(f"\n[Time steps: {N}] Monte Carlo with {num_samples} samples...")
            mean_cost = self.monte_carlo_estimate(x0, N, num_samples)
            err = abs(mean_cost - exact_value)
            
            estimated_values.append(mean_cost)
            errors.append(err)
            
            print(f"  -> Estimate: {mean_cost:.6f}, Error: {err:.6f}")
        
        return {
            'num_steps': num_steps_list,
            'estimated_values': estimated_values,
            'errors': errors,
            'exact_value': exact_value
        }
    
    def convergence_study_samples(self, x0, num_samples_list, num_steps=10000):
        """
        研究样本数的收敛性
        
        Args:
            x0: 初始状态 (2,)
            num_samples_list: 样本数列表 (如 [2,8,32,128,...])
            num_steps: 固定时间步数 (默认10000)
            
        Returns:
            结果字典
        """
        exact_value = self.lqr_solver.value_function(torch.tensor([0.0]), x0.unsqueeze(0)).item()
        print(f"Exact value from Riccati solver: {exact_value:.6f}")
        
        estimated_values = []
        errors = []
        
        for samples in num_samples_list:
            print(f"\n[Samples: {samples}] Monte Carlo with {num_steps} time steps...")
            mean_cost = self.monte_carlo_estimate(x0, num_steps, samples)
            err = abs(mean_cost - exact_value)
            
            estimated_values.append(mean_cost)
            errors.append(err)
            
            print(f"  -> Estimate: {mean_cost:.6f}, Error: {err:.6f}")
        
        return {
            'num_samples': num_samples_list,
            'estimated_values': estimated_values,
            'errors': errors,
            'exact_value': exact_value
        }
    
    def plot_convergence_time_steps(self, results):
        """
        绘制时间步长收敛性图 (log-log)
        
        Args:
            results: convergence_study_time_steps的结果
        """
        plt.figure(figsize=(8, 5))
        
        plt.loglog(results['num_steps'], results['errors'], 'o-', label='Error')
        
        # 添加参考斜率线 O(1/N)
        if len(results['errors']) > 1:
            max_error = results['errors'][0]
            min_step = min(results['num_steps'])
            max_step = max(results['num_steps'])
            ref_x = [min_step, max_step]
            ref_y = [max_error * (min_step / x) for x in ref_x]
            plt.loglog(ref_x, ref_y, '--', label='Slope -1')
        
        plt.xlabel('Number of Time Steps (log scale)')
        plt.ylabel('Absolute Error (log scale)')
        plt.title('Convergence w.r.t. Time Steps')
        plt.grid(True, which='both', ls='--')
        plt.legend()
        return plt
    
    def plot_convergence_samples(self, results):
        """
        绘制样本数收敛性图 (log-log)
        
        Args:
            results: convergence_study_samples的结果
        """
        plt.figure(figsize=(8, 5))
        
        plt.loglog(results['num_samples'], results['errors'], 'o-', label='Error')
        
        # 添加参考斜率线 O(1/sqrt(N))
        if len(results['errors']) > 1:
            max_error = results['errors'][0]
            min_samples = min(results['num_samples'])
            max_samples = max(results['num_samples'])
            ref_x = [min_samples, max_samples]
            ref_y = [max_error * (min_samples / x)**0.5 for x in ref_x]
            plt.loglog(ref_x, ref_y, '--', label='Slope -1/2')
        
        plt.xlabel('Number of Monte Carlo Samples (log scale)')
        plt.ylabel('Absolute Error (log scale)')
        plt.title('Convergence w.r.t. Monte Carlo Samples')
        plt.grid(True, which='both', ls='--')
        plt.legend()
        return plt


def main_exercise1_2():
    """
    这是Exercise 1.2的示例主函数，可根据需要修改或删除。
    假设你已在lqr_solver.py里定义并实现了LQRSolver。
    """
    print("Starting LQR Monte Carlo simulation for Exercise 1.2...")
    
    # 定义问题参数
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
    
    # 创建时间网格
    time_grid = torch.linspace(0, T, 100)
    
    # 从EX1.1导入的LQRSolver
    from Exercise_1 import LQRSolver
    lqr_solver = LQRSolver(H, M, sigma, C, D, R, T, time_grid)
    mc_simulator = LQRMonteCarlo(lqr_solver)
    
    # 测试点
    x0 = torch.tensor([1.0, 1.0])
    exact_value = lqr_solver.value_function(torch.tensor([0.0]), x0.unsqueeze(0)).item()
    print(f"\nExact value at x0={x0.tolist()}: {exact_value:.6f}")
    
    # 小规模测试
    mean_cost = mc_simulator.monte_carlo_estimate(x0, num_steps=100, num_samples=1000)
    print(f"MC estimate (steps=100, samples=1000): {mean_cost:.6f}")
    print(f"Error: {abs(mean_cost - exact_value):.6f}")
    
    # 时间步数收敛性研究 (例子: 2,4,8,16)
    num_steps_list = [2**i for i in range(1, 12)]
    time_steps_results = mc_simulator.convergence_study_time_steps(x0, num_steps_list, num_samples=1000)
    plt_time = mc_simulator.plot_convergence_time_steps(time_steps_results)
    plt_time.savefig('convergence_time_steps.png')
    
    # 样本数收敛性研究 (例子: 2,8,32)
    num_samples_list = [2 * 4**i for i in range(0, 6)]
    samples_results = mc_simulator.convergence_study_samples(x0, num_samples_list, num_steps=100)
    plt_samp = mc_simulator.plot_convergence_samples(samples_results)
    plt_samp.savefig('convergence_samples.png')
    
    print("\nAll done. Results saved to PNG files.")

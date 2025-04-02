""" 
This file is used to output the required charts in the assignment
"""
import subprocess
import sys

def install_requirements():
    try:
        # 使用subprocess模块运行pip install命令
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

"""Exercise 1.1"""
from Exercise_1_1 import main_exercise1
if __name__ == "__main__":
    main_exercise1_1()


"""Exercise 1.2"""
from Exercise_1_2 import main_exercise2
if __name__ == "__main__":
    main_exercise1_2()

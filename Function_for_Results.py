""" 
This file is used to output the required charts in the assignment
"""
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

"""Exercise 1.1"""
from Exercise_1_1 import main_exercise1_1
if __name__ == "__main__":
    main_exercise1_1()

"""Exercise 1.2"""
from Exercise_1_2 import main_exercise1_2_initial_1, main_exercise1_2_initial_2
if __name__ == "__main__":
    main_exercise1_2_initial_1()

if __name__ == "__main__":
    main_exercise1_2_initial_2()

"""Exercise 2.1"""
from Exercise_2 import main_exercise2
if __name__ == "__main__":
    main_exercise2()

"""Exercise 3.1"""
from Exercise_3 import main_exercise3_comparsion,main_exercise3_critic_loss
if __name__ == "__main__":
    main_exercise3_critic_loss()

if __name__ == "__main__":
    main_exercise3_comparsion()

"""Exercise 4.1"""
from Exercise_4 import main_exercise4_cost, main_exercise4_loss
if __name__ == "__main__":
    main_exercise4_loss()

if __name__ == "__main__":
    main_exercise4_cost()

# """Exercise 5.1"""
# from Exercise_5 import main_exercise5
# if __name__ == "__main__":
#     main_exercise5()


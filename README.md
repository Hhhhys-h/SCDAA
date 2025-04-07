# SCDAA
We put the definition of classes and functions in 6 documents, which are named by their corresponding exercise title. Markdowns and some necessary explanations are included to make the process clearer, you can view all details of our code here and check how we work to solve the problems. The graphs and results are shown in the relative Exercise_results document, the code of calling the function for output is placed in the Function_for_Results.py document, you can browse the functions here and copy the code you want to run separately for a specific answer. To help you better call the code, we will introduce the functions necessary to adjust variables and output results.

**Exercise 1.1: Solving LQR using Ricatti ODE**

To get the results of value function and optimal control, you can run the code below:
```
if __name__ == "__main__":
    main_exercise1_1()
```

**Exercise 1.2：LQR MC checks**

To run the functions and get the convergence of time steps and Monte Carlo samples, you can copy and run the code below:
```
if __name__ == "__main__":
    main_exercise1_2()
```
The initial points are [1,1] and [2,2].

**Exercise 2.1：Solving soft LQR**

To plot the figure of controlled trajectories for the strict LQR and relaxed LQR from four different starting points, you can run the code below:
```
if __name__ == "__main__":
    main_exercise2()
```

**Exercise 3.1：Critic algorithm**

```
if __name__ == "__main__":
    main_exercise3()
```

**Exercise 4.1：Actor algorithm**
```
if __name__ == "__main__":
    main_exercise1_4()
```
**Exercise 5.1：Actor-critic algorithm**
```
if __name__ == "__main__":
    main_exercise1_5()
```

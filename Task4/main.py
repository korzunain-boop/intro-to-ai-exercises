import numpy as np
import matplotlib.pyplot as plt

E = np.exp(1)
PI = np.pi 

def ackley_1d(x):
    if isinstance(x, np.ndarray) and x.ndim > 0 and x.size == 1:
        x_val = x[0]
    else:
        x_val = x
        
    term1 = -20.0 * np.exp(-0.2 * np.sqrt(x_val**2))
    term2 = -np.exp(np.cos(2.0 * PI * x_val))
    return term1 + term2 + 20.0 + E

def ackley_1d_grad(x):
    x_val = x[0]
    x_abs = np.abs(x_val)
    
    if x_abs == 0.0:
        return np.array([0.0])
    
    sign_x = x_val / x_abs
    grad_term1 = 4.0 * sign_x * np.exp(-0.2 * x_abs)
    grad_term2 = 2.0 * PI * np.sin(2.0 * PI * x_val) * np.exp(np.cos(2.0 * PI * x_val))
    
    return np.array([grad_term1 + grad_term2])

def ackley_2d(x):
    x_val, y_val = x[0], x[1]
    r = np.sqrt((x_val**2 + y_val**2) / 2.0)
    term1 = -20.0 * np.exp(-0.2 * r)
    c = (np.cos(2.0 * PI * x_val) + np.cos(2.0 * PI * y_val)) / 2.0
    term2 = -np.exp(c)
    return term1 + term2 + 20.0 + E

def ackley_2d_grad(x):
    x_val, y_val = x[0], x[1]
    
    if np.all(x == 0.0):
        return np.array([0.0, 0.0])
    
    r = np.sqrt((x_val**2 + y_val**2) / 2.0)
    exp_r = np.exp(-0.2 * r)
    
    c = (np.cos(2.0 * PI * x_val) + np.cos(2.0 * PI * y_val)) / 2.0
    exp_c = np.exp(c)
    
    df_dx = 2.0 * (x_val / r) * exp_r + PI * np.sin(2.0 * PI * x_val) * exp_c
    df_dy = 2.0 * (y_val / r) * exp_r + PI * np.sin(2.0 * PI * y_val) * exp_c
    
    return np.array([df_dx, df_dy])


def gradient_descent_best(func, func_grad, x_start, learning_rate, n_iterations):
    x = np.array(x_start, dtype=float)
    history = [x.copy()]
    
    current_f = func(x)
    best_f = current_f
    best_x = x.copy()
    
    for _ in range(n_iterations):
        grad = func_grad(x)
        x -= learning_rate * grad
        history.append(x.copy())
        
        current_f = func(x)
        if current_f < best_f:
            best_f = current_f
            best_x = x.copy()
        
    return best_x, best_f, history

N_ITER = 2500
START_1D = [5.0]
START_2D = [5.0, 5.0]

learning_rates_1d = [0.01, 0.1, 0.5] 
results_1d = {}

print("--- 1D Ackley results ---")

for alpha in learning_rates_1d:
    best_x, best_f, history = gradient_descent_best(ackley_1d, ackley_1d_grad, START_1D, alpha, N_ITER)
    results_1d[alpha] = (best_x[0], best_f, history)
    print(f"Alpha={alpha}: Best x={best_x[0]:.4f}, Best f(x)={best_f:.4f}")

learning_rates_2d = [0.01, 0.1, 0.2] 
results_2d = {}

print("\n--- 2D Ackley results ---")

for alpha in learning_rates_2d:
    best_x, best_f, history = gradient_descent_best(ackley_2d, ackley_2d_grad, START_2D, alpha, N_ITER)
    results_2d[alpha] = (best_x, best_f, history)
    print(f"Alpha={alpha}: Best x={best_x}, Best f(x)={best_f:.4f}")

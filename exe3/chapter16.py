#2 my_ls_params Function
import numpy as np
import matplotlib.pyplot as plt
def my_ls_params(f, x, y):
    A = np.vstack([func(x) for func in f]).T
    beta = np.linalg.inv(A.T @ A) @ A.T @ y
    return beta
x = np.linspace(0, 2*np.pi, 1000)
y = 3*np.sin(x) - 2*np.cos(x) + np.random.random(len(x))
f = [np.sin, np.cos, lambda x: np.ones_like(x)]
beta = my_ls_params(f, x, y)
plt.figure(figsize=(10, 8))
plt.plot(x, y, 'b.', label='Data')
plt.plot(x, beta[0]*np.sin(x) + beta[1]*np.cos(x) + beta[2], 'r', label='Regression')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Least Squares Regression Example')
plt.legend()
plt.show()

#3 my_func_fit
def my_func_fit(x, y):
    log_x = np.log(x)
    log_y = np.log(y)
    A = np.vstack([log_x, np.ones_like(log_x)]).T
    beta, log_alpha = np.linalg.lstsq(A, log_y, rcond=None)[0]
    alpha = np.exp(log_alpha)
    return alpha, beta
x = np.array([1, 2, 3, 4, 5])
y = np.array([2.3, 4.8, 7.5, 10.1, 12.8])
alpha, beta = my_func_fit(x, y)
print(f"alpha: {alpha}, beta: {beta}")
plt.figure(figsize=(10, 8))
plt.plot(x, y, 'ro', label='Data points')
plt.plot(x, alpha * x**beta, 'b', label='Fitted function')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Function Fitting Example')
plt.legend()
plt.show()

#4 Python implementation that calculates the total error and checks if a new point will incur additional error
a = 1.0
b = -2.0
c = 3.0
d = -4.0
data_points = np.array([
    [1, -2],
    [2, -1],
    [3, 0],
    [4, 1]
])
def cubic_polynomial(x, a, b, c, d):
    return a*x**3 + b*x**2 + c*x + d
def total_error(data_points, a, b, c, d):
    error = 0.0
    for x_i, y_i in data_points:
        y_hat = cubic_polynomial(x_i, a, b, c, d)
        error += (y_i - y_hat) ** 2
    return error
error = total_error(data_points, a, b, c, d)
print(f"Total Error: {error}")
def check_new_point(x_new, y_new, a, b, c, d):
    y_hat = cubic_polynomial(x_new, a, b, c, d)
    return np.isclose(y_new, y_hat)
x_new = 5.0
y_new = 2.0
is_on_polynomial = check_new_point(x_new, y_new, a, b, c, d)
print(f"New point lies on the polynomial: {is_on_polynomial}")

#5 my_lin_regression

def my_lin_regression(f, x, y):
    m = len(x)
    n = len(f)
    A = np.zeros((m, n))
    for i in range(n):
        A[:, i] = f[i](x)
    A_T_A_inv = np.linalg.inv(A.T @ A)
    beta = A_T_A_inv @ A.T @ y
    return beta
x = np.linspace(0, 2 * np.pi, 1000)
y = 3 * np.sin(x) - 2 * np.cos(x) + np.random.random(len(x))
f = [np.sin, np.cos, lambda x: np.ones_like(x)]  # Include a bias term

beta = my_lin_regression(f, x, y)

plt.figure(figsize=(10, 8))
plt.plot(x, y, 'b.', label='Data')
plt.plot(x, beta[0] * f[0](x) + beta[1] * f[1](x) + beta[2], 'r', label='Regression')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Least Square Regression Example')
plt.legend()
plt.show()

#5.1 my_exp_regression
def my_exp_regression(x, y):
    Y = np.log(y)
    A = np.vstack([x, np.ones_like(x)]).T
    A_T_A_inv = np.linalg.inv(A.T @ A)
    beta_C = A_T_A_inv @ A.T @ Y
    beta = beta_C[0]
    C = beta_C[1]
    alpha = np.exp(C)
    return alpha, beta
x = np.linspace(0, 1, 1000)
y = 2 * np.exp(-0.5 * x) + 0.25 * np.random.random(len(x))
alpha, beta = my_exp_regression(x, y)
plt.figure(figsize=(10, 8))
plt.plot(x, y, 'b.', label='Data')
plt.plot(x, alpha * np.exp(beta * x), 'r', label='Regression')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Least Square Regression on Exponential Model')
plt.legend()
plt.show()

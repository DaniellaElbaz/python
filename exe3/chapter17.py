#1 Linear Interpolation Function
import numpy as np

def my_lin_interp(x, y, X):
    Y = np.zeros_like(X)
    for i in range(len(X)):
        if X[i] <= x[0]:
            Y[i] = y[0]
        elif X[i] >= x[-1]:
            Y[i] = y[-1]
        else:
            for j in range(len(x) - 1):
                if x[j] <= X[i] <= x[j + 1]:
                    Y[i] = y[j] + (X[i] - x[j]) * (y[j + 1] - y[j]) / (x[j + 1] - x[j])
                    break
    return Y

#2 Cubic Spline Interpolation Function
def my_cubic_spline(x, y, X):
    n = len(x)
    h = np.diff(x)
    al = [0] * (n-1)
    for i in range(1, n-1):
        al[i] = (3 / h[i]) * (y[i+1] - y[i]) - (3 / h[i-1]) * (y[i] - y[i-1])
    l = [1] * n
    mu = [0] * n
    z = [0] * n
    for i in range(1, n-1):
        l[i] = 2 * (x[i+1] - x[i-1]) - h[i-1] * mu[i-1]
        mu[i] = h[i] / l[i]
        z[i] = (al[i] - h[i-1] * z[i-1]) / l[i]
    b = [0] * (n-1)
    c = [0] * n
    d = [0] * (n-1)
    for j in range(n-2, -1, -1):
        c[j] = z[j] - mu[j] * c[j+1]
        b[j] = (y[j+1] - y[j]) / h[j] - h[j] * (c[j+1] + 2 * c[j]) / 3
        d[j] = (c[j+1] - c[j]) / (3 * h[j])
    Y = np.zeros_like(X)
    for i in range(len(X)):
        if X[i] <= x[0]:
            Y[i] = y[0]
        elif X[i] >= x[-1]:
            Y[i] = y[-1]
        else:
            for j in range(n - 1):
                if x[j] <= X[i] <= x[j + 1]:
                    Y[i] = y[j] + b[j] * (X[i] - x[j]) + c[j] * (X[i] - x[j])**2 + d[j] * (X[i] - x[j])**3
                    break
    return Y
#5  Cubic Spline with Flat Endpoints
def my_cubic_spline_flat(x, y, X):
    n = len(x)
    h = np.diff(x)
    al = [0] * (n-1)
    for i in range(1, n-1):
        al[i] = (3 / h[i]) * (y[i+1] - y[i]) - (3 / h[i-1]) * (y[i] - y[i-1])
    l = [1] * n
    mu = [0] * n
    z = [0] * n
    for i in range(1, n-1):
        l[i] = 2 * (x[i+1] - x[i-1]) - h[i-1] * mu[i-1]
        mu[i] = h[i] / l[i]
        z[i] = (al[i] - h[i-1] * z[i-1]) / l[i]
    c = [0] * n
    b = [0] * (n-1)
    d = [0] * (n-1)
    for j in range(n-2, -1, -1):
        c[j] = z[j] - mu[j] * c[j+1]
        b[j] = (y[j+1] - y[j]) / h[j] - h[j] * (c[j+1] + 2 * c[j]) / 3
        d[j] = (c[j+1] - c[j]) / (3 * h[j])
    Y = np.zeros_like(X)
    for i in range(len(X)):
        if X[i] <= x[0]:
            Y[i] = y[0]
        elif X[i] >= x[-1]:
            Y[i] = y[-1]
        else:
            for j in range(n - 1):
                if x[j] <= X[i] <= x[j + 1]:
                    Y[i] = y[j] + b[j] * (X[i] - x[j]) + c[j] * (X[i] - x[j])**2 + d[j] * (X[i] - x[j])**3
                    break
    return Y
#6 Quintic Spline Interpolation Function
import numpy as np

def my_quintic_spline(x, y, X):
    h = np.diff(x)
    A = np.zeros((len(x), len(x)))
    b = np.zeros(len(x))
    A[0, 0] = 1
    A[-1, -1] = 1
    for i in range(1, len(x) - 1):
        A[i, i-1] = h[i-1]
        A[i, i] = 2 * (h[i-1] + h[i])
        A[i, i+1] = h[i]
        b[i] = 3 * (y[i+1] - y[i]) / h[i] - 3 * (y[i] - y[i-1]) / h[i-1]
    c = np.linalg.solve(A, b)
    b = np.zeros(len(x) - 1)
    d = np.zeros(len(x) - 1)
    for i in range(len(x) - 1):
        b[i] = (y[i+1] - y[i]) / h[i] - h[i] * (c[i+1] + 2 * c[i]) / 3
        d[i] = (c[i+1] - c[i]) / (3 * h[i])
    Y = np.zeros(len(X))
    for j in range(len(X)):
        for i in range(len(x) - 1):
            if x[i] <= X[j] <= x[i+1]:
                dx = X[j] - x[i]
                Y[j] = y[i] + b[i] * dx + c[i] * dx**2 + d[i] * dx**3
                break
    return Y

#7 Interpolation Plotter Function
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

def my_interp_plotter(x, y, X, option):
    if option == 'nearest':
        f = interp1d(x, y, kind='nearest')
        Y = f(X)
    elif option == 'linear':
        f = interp1d(x, y, kind='linear')
        Y = f(X)
    elif option == 'cubic':
        f = interp1d(x, y, kind='cubic')
        Y = f(X)
    else:
        raise ValueError("Invalid interpolation option provided. Choose 'nearest', 'linear', or 'cubic'.")
    plt.plot(x, y, 'ro', label='Data points')
    plt.plot(X, Y, 'b-', label=f'{option.capitalize()} interpolation')
    plt.title(f'{option.capitalize()} Interpolation')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.show()

x = np.array([0, .1, .15, .35, .6, .7, .95, 1])
y = np.array([1, 0.8187, 0.7408, 0.4966, 0.3012, 0.2466, 0.1496, 0.1353])
my_interp_plotter(x, y, np.linspace(0, 1, 101), 'nearest')
my_interp_plotter(x, y, np.linspace(0, 1, 101), 'linear')
my_interp_plotter(x, y, np.linspace(0, 1, 101), 'cubic')
#my_D_cubic_spline Function
import numpy as np

def my_D_cubic_spline(x, y, X, D):
    n = len(x)
    h = np.diff(x)
    alpha = np.zeros(n)
    for i in range(1, n-1):
        alpha[i] = (3/h[i] * (y[i+1] - y[i])) - (3/h[i-1] * (y[i] - y[i-1]))

    l = np.ones(n)
    mu = np.zeros(n-1)
    z = np.zeros(n)
    l[0] = 1
    mu[0] = 0
    z[0] = D
    for i in range(1, n-1):
        l[i] = 2 * (x[i+1] - x[i-1]) - h[i-1] * mu[i-1]
        mu[i] = h[i] / l[i]
        z[i] = (alpha[i] - h[i-1] * z[i-1]) / l[i]
    l[n-1] = 1
    z[n-1] = D

    b = np.zeros(n-1)
    c = np.zeros(n)
    d = np.zeros(n-1)
    for j in range(n-2, -1, -1):
        c[j] = z[j] - mu[j] * c[j+1]
        b[j] = (y[j+1] - y[j])/h[j] - h[j] * (c[j+1] + 2*c[j]) / 3
        d[j] = (c[j+1] - c[j]) / (3*h[j])
    Y = np.zeros_like(X)
    for i in range(len(X)):
        for j in range(n-1):
            if x[j] <= X[i] <= x[j+1]:
                dx = X[i] - x[j]
                Y[i] = y[j] + b[j]*dx + c[j]*dx**2 + d[j]*dx**3
                break
    return Y
x = [0, 1, 2, 3, 4]
y = [0, 0, 1, 0, 0]
X = np.linspace(0, 4, 101)
plt.figure(figsize = (10, 8))
plt.subplot(221)
plt.plot(x, y, 'ro', X, my_D_cubic_spline(x, y, X, 0), 'b')
plt.subplot(222)
plt.plot(x, y, 'ro', X, my_D_cubic_spline(x, y, X, 1), 'b')
plt.subplot(223)
plt.plot(x, y, 'ro', X, my_D_cubic_spline(x, y, X, -1), 'b')
plt.subplot(224)
plt.plot(x, y, 'ro', X, my_D_cubic_spline(x, y, X, 4), 'b')
plt.tight_layout()
plt.show()
#my_lagrange Function +fit
import numpy as np

def my_lagrange(x, y, X):
    def L(k, X):
        Lk = 1
        for i in range(len(x)):
            if i != k:
                Lk *= (X - x[i]) / (x[k] - x[i])
        return Lk
    Y = np.zeros_like(X)
    for i in range(len(X)):
        Y[i] = sum(y[k] * L(k, X[i]) for k in range(len(x)))
    return Y
x = [0, 1, 2, 3, 4]
y = [2, 1, 3, 5, 1]
X = np.linspace(0, 4, 101)

plt.figure(figsize=(10, 8))
plt.plot(X, my_lagrange(x, y, X), 'b', label='Interpolation')
plt.plot(x, y, 'ro', label='Data points')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Lagrange Interpolation of Data Points')
plt.legend()
plt.show()
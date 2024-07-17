#1 N-th Root using Newton-Raphson Method

def my_nth_root(x, n, tol):
    r = x
    while abs(r**n - x) > tol:
        r = ((n - 1) * r + x / r**(n - 1)) / n
    return r
print(my_nth_root(27, 3, 1e-6))
print(my_nth_root(16, 4, 1e-6))

#2 Fixed Point using Bisection Method
def my_fixed_point(f, g, tol, max_iter):
    def F(x):
        return f(x) - g(x)
    a, b = -1e10, 1e10
    for _ in range(max_iter):
        m = (a + b) / 2
        if abs(F(m)) < tol:
            return m
        elif F(a) * F(m) < 0:
            b = m
        else:
            a = m
    return []

f = lambda x: x**2
g = lambda x: x
print(my_fixed_point(f, g, 1e-6, 1000))
#3 The bisection method fails for f(x)= 1/x

#שיטת הביסקציה נכשלת עבור
#𝑓(𝑥)=1/𝑥
 #עקב אי המשכיות ב-x=0.
#חוסר המשכיות הזה גורם
#f(x) להפר את ה-IVT, שהוא חיוני לשיטת הביסקציה כדי להבטיח את קיומו של שורש בתוך המרווח.
#כתוצאה מכך, השיטה לא יכולה להפחית ביעילות את השגיאה או למצוא שורש במקרים כאלה

#4 Bisection Method
import numpy as np
def my_bisection(f, a, b, tol):
    R, E = [], []
    while abs(a - b) > tol:
        m = (a + b) / 2
        R.append(m)
        E.append(abs(f(m)))
        if f(a) * f(m) < 0:
            b = m
        else:
            a = m
    return R, E

f1 = lambda x: x**2 - 2
R1, E1 = my_bisection(f1, 0, 2, 1e-1)
print(R1, E1)

f2 = lambda x: np.sin(x) - np.cos(x)
R2, E2 = my_bisection(f2, 0, 2, 1e-2)
print(R2, E2)

#4.1 Newton-Raphson Method
def my_newton(f, df, x0, tol):
    R, E = [x0], [abs(f(x0))]
    x = x0
    while abs(f(x)) > tol:
        x = x - f(x) / df(x)
        R.append(x)
        E.append(abs(f(x)))
    return R, E

f1 = lambda x: x**2 - 2
df1 = lambda x: 2 * x
R1, E1 = my_newton(f1, df1, 2, 1e-5)
print(R1, E1)

f2 = lambda x: np.sin(x) - np.cos(x)
df2 = lambda x: np.cos(x) + np.sin(x)
R2, E2 = my_newton(f2, df2, 1, 1e-5)
print(R2, E2)

#my_pipe_builder
import math
def my_pipe_builder(C_ocean, C_land, L, H):
    def total_cost(x):
        return C_ocean * math.sqrt(H**2 + x**2) + C_land * (L - x)
    a = 0
    b = L
    tol = 1e-6

    while (b - a) > tol:
        mid = (a + b) / 2
        cost_mid = total_cost(mid)
        cost_left = total_cost(mid - tol)
        cost_right = total_cost(mid + tol)
        if cost_left < cost_mid:
            b = mid
        elif cost_right < cost_mid:
            a = mid
        else:
            break
    return (a + b) / 2
print(my_pipe_builder(20, 10, 100, 50))
print(my_pipe_builder(30, 10, 100, 50))
print(my_pipe_builder(30, 10, 100, 20))

# newton_raphson_oscillate
def newton_raphson_oscillate(f, df, x0, iterations=10):
    x = x0
    for _ in range(iterations):
        x = x - f(x) / df(x)
        print(f"x = {x}")

f = lambda x: x**3 - x
df = lambda x: 3 * x**2 - 1
newton_raphson_oscillate(f, df, 1)
newton_raphson_oscillate(f, df, -1)

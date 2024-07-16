
#Write a function my_sinh(x), where the output y is the hyperbolic sine computed on x. Assume that x is a 1 by 1 float
import numpy as np
def my_sinh(x):
    y = (np.exp(x) - np.exp(-x)) / 2
    return y
x = 1.0
y = my_sinh(x)
print(f"The hyperbolic sine of {x} is {y}")

#Checkerboard Pattern
import numpy as np

def my_checker_board(n):
    return np.array([[1 if (i + j) % 2 == 0 else 0 for j in range(n)] for i in range(n)])

print(my_checker_board(1))
print(my_checker_board(2))
print(my_checker_board(3))
print(my_checker_board(5))
#Triangle Area
def my_triangle(b, h):
    return 0.5 * b * h
print(my_triangle(1, 1))
print(my_triangle(2, 1))
print(my_triangle(12, 5))

#Split Matrix
def my_split_matrix(m):
    mid = (m.shape[1] + 1) // 2
    m1 = m[:, :mid]
    m2 = m[:, mid:]
    return m1, m2

m = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
m1, m2 = my_split_matrix(m)
print(m1)
print(m2)

m = np.ones((5, 5))
m1, m2 = my_split_matrix(m)
print(m1)
print(m2)

#Cylinder Surface Area and Volume

import math

def my_cylinder(r, h):
    surface_area = 2 * math.pi * r * (r + h)
    volume = math.pi * r ** 2 * h
    return [round(surface_area, 4), round(volume, 4)]

print(my_cylinder(1, 5))

#Count Odd Numbers
def my_n_odds(a):
    return sum(1 for x in a if x % 2 != 0)

print(my_n_odds(np.arange(100)))
print(my_n_odds(np.arange(2, 100, 2)))

#Matrix of Twos
def my_twos(m, n):
    return np.full((m, n), 2)
print(my_twos(3, 2))
print(my_twos(1, 4))
# Lambda Function and String Concatenation
# 1. Lambda function for subtraction
subtract = lambda x, y: x - y

# 2. String concatenation
def add_string(s1, s2):
    return s1 + s2
s1 = add_string('Programming', ' ')
s2 = add_string('is', ' fun!')
print(add_string(s1, s2))

#Generate Errors and Greeting Function

# 1. Generate Errors
try:
    def fun(a):
        pass
    fun()
except TypeError as e:
    print(e)

def wrong_indent():
 pass

# 2. Greeting Function
def greeting(name, age):
    return f'Hi, my name is {name} and I am {age} years old.'

print(greeting('John', 26))
print(greeting('Kate', 19))


#Donut Area

def my_donut_area(r1, r2):
    return np.pi * (r2**2 - r1**2)

print(my_donut_area(np.arange(1, 4), np.arange(2, 7, 2)))

#Within Tolerance

def my_within_tolerance(A, a, tol):
    return [i for i, x in enumerate(A) if abs(x - a) < tol]

print(my_within_tolerance([0, 1, 2, 3], 1.5, 0.75))
print(my_within_tolerance(np.arange(0, 1.01, 0.01), 0.5, 0.03))

#Bounding Array

def bounding_array(A, top, bottom):
    return [min(max(x, bottom), top) for x in A]

print(bounding_array(np.arange(-5, 6, 1), 3, -3))
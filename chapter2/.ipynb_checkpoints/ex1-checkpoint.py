
#Write a function my_sinh(x), where the output y is the hyperbolic sine computed on x. Assume that x is a 1 by 1 float
import numpy as np
def my_sinh(x):
    y = (np.exp(x) - np.exp(-x)) / 2
    return y
x = 1.0
y = my_sinh(x)
print(f"The hyperbolic sine of {x} is {y}")

#2)
S = '123'
N = float(S)
type_S = type(S)
type_N = type(N)
print(f"Type of S: {type_S}")
print(f"Type of N: {type_N}")

#3)
s1 = 'HELLO'
s2 = 'hello'
print(f"s1 == s2: {s1 == s2}")
print(f"s1.lower() == s2: {s1.lower() == s2}")
print(f"s1 == s2.upper(): {s1 == s2.upper()}")

#1)
#jupyter notebook
#%whos

#4)
print("The word 'Engineering' has", len("Engineering"), "letters.")
print("The word 'Book' has", len("Book"), "letters.")

#5)
string_5 = 'Python is great!'
substring = 'Python'

if substring in string_5:
    print(f"'{substring}' is in '{string_5}'")
else:
    print(f"'{substring}' is not in '{string_5}'")
#6)
string_6 = 'Python is great!'
last_word = string_6.split()[-1]
print("Last word:", last_word)
#7)
list_a = [1, 8, 9, 15]
list_a.insert(1, 2)
list_a.append(4)

print("Modified list_a after insert and append operations:", list_a)
#8)
list_a.sort()
print("Sorted list_a in ascending order:", list_a)
#17)
import numpy as np
array_17 = np.linspace(-10, 10, 100)
print(array_17)
#18)
array_a = np.array([-1, 0, 1, 2, 0, 3])
array_18 = array_a[array_a > 0]
print(array_18)
#19)
y = np.array([
    [3, 5, 3],
    [2, 2, 5],
    [3, 8, 9]
])
y_transpose = np.transpose(y)
print(y_transpose)
#20)
zero_array = np.zeros((2, 4))
print(zero_array)
#21)
zero_array[:, 1] = 1
print(zero_array)
#22)
#jupyter notebook
#%reset -f
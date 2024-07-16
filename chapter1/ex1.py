import math
#9)
print(math.factorial(6))
#10)
def is_leap_year(year):
    if year % 400 == 0:
        return True
    if year % 100 == 0:
        return False
    if year % 4 == 0:
        return True
    return False
start_year = 1500
end_year = 2010
leap_years_count = sum(is_leap_year(year) for year in range(start_year, end_year + 1))
print("Number of leap years between 1500 and 2010:", leap_years_count)
#11)
factor = 2 * math.sqrt(2) / 9801
def ramanujan_sum(N):
    total_sum = 0
    for k in range(N + 1):
        numerator = math.factorial(4 * k) * (1103 + 26390 * k)
        denominator = (math.factorial(k) ** 4) * (396 ** (4 * k))
        total_sum += numerator / denominator
    return total_sum
def approximate_pi(N):
    ramanujan_sum_value = ramanujan_sum(N)
    return 1 / (factor * ramanujan_sum_value)
approx_pi_N0 = approximate_pi(0)
approx_pi_N1 = approximate_pi(1)
true_pi = math.pi
print(f"N=0: {approx_pi_N0}")
print(f"N=1: {approx_pi_N1}")
print(f"pi: {true_pi}")
print(f"Difference for N=0: {abs(true_pi - approx_pi_N0)}")
print(f"Difference for N=1: {abs(true_pi - approx_pi_N1)}")
#13)
x_values = [math.pi, math.pi / 2, math.pi / 4, math.pi / 6]
def verify_identity(x):
    sin_squared = math.sin(x) ** 2
    cos_squared = math.cos(x) ** 2
    return sin_squared + cos_squared
results = {x: verify_identity(x) for x in x_values}
for x, result in results.items():
    print(f"sin^2({x}) + cos^2({x}) = {result}")
for x, result in results.items():
    assert math.isclose(result, 1, rel_tol=1e-9), f"not hold for x = {x}"
#14)
degrees = 87
radians = math.radians(degrees)
sin_value = math.sin(radians)
sin_value
print(f"The sine of 87 degrees is {sin_value}")
#19)

def logical_expression(P, Q):
    return (P and Q) or (P and (not Q))
results = []
for P in [True, False]:
    for Q in [True, False]:
        result = logical_expression(P, Q)
        results.append((P, Q, result))
false_conditions = [(P, Q) for P, Q, result in results if not result]
print(" under Conditions  is false:")
for P, Q in false_conditions:
    print(f"P = {P}, Q = {Q}")
#22.1)
e_squared = math.exp(2)
sin_value = math.sin(math.pi / 6)
log_value = math.log(3)
cos_value = math.cos(math.pi / 9)
power_value = 5 ** 3
result = e_squared * sin_value + log_value * cos_value - power_value
result
#22.2)
P = 1
Q = 1
not_P = not P
not_Q = not Q
result = not_P and not_Q
result

a = 10
b = 25
condition1 = a < b
condition2 = a == b
result = condition1 and condition2
result
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
print(dir())
print(globals())

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
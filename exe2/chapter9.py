#Binary to Decimal Conversion

def my_bin_2_dec(b):
    d = 0
    for i, bit in enumerate(reversed(b)):
        d += bit * (2 ** i)
    return d

print(my_bin_2_dec([1, 1, 1]))
print(my_bin_2_dec([1, 0, 1, 0, 1, 0, 1]))
print(my_bin_2_dec([1] * 25))
#Decimal to Binary Conversion
def my_dec_2_bin(d):
    if d == 0:
        return [0]
    b = []
    while d > 0:
        b.append(d % 2)
        d //= 2
    b.reverse()
    return b

print(my_dec_2_bin(0))
print(my_dec_2_bin(23))
print(my_dec_2_bin(2097))

#Combine Functions
d = my_bin_2_dec(my_dec_2_bin(12654))
print(d)

#Binary Addition
def my_bin_adder(b1, b2):
    max_len = max(len(b1), len(b2))
    b1 = [0] * (max_len - len(b1)) + b1
    b2 = [0] * (max_len - len(b2)) + b2
    result = []
    carry = 0
    for i in range(max_len - 1, -1, -1):
        total = b1[i] + b2[i] + carry
        result.append(total % 2)
        carry = total // 2
    if carry:
        result.append(carry)
    result.reverse()
    return result

print(my_bin_adder([1, 1, 1, 1, 1], [1]))
print(my_bin_adder([1, 1, 1, 1, 1], [1, 0, 1, 0, 1, 0, 0]))
print(my_bin_adder([1, 1, 0], [1, 0, 1]))

#Bit Allocation Effects
#הקצאת ביטים נוספים למונה מגדילה את דיוק המספר
#הקצאת יותר ביטים למכנה מקטינה את דיוק המספר
#הקצאת ביטים נוספים לסימן אינה משפיעה ישירות על דיוק או טווח, היא משמשת לייצג מספרים חיוביים ושליליים .

#IEEE 754 to Decimal
def my_ieee_2_dec(ieee):
    sign = int(ieee[0])
    exponent = int(ieee[1:12], 2) - 1023
    mantissa_bin = ieee[12:]
    mantissa = 1
    for i, bit in enumerate(mantissa_bin):
        mantissa += int(bit) * (2 ** -(i + 1))
    decimal = mantissa * (2 ** exponent)
    if sign == 1:
        decimal *= -1
    return decimal

print(my_ieee_2_dec('1100000001001000000000000000000000000000000000000000000000000000'))
print(my_ieee_2_dec('0100000000001011001100110011001100110011001100110011001100110011'))

#Decimal to IEEE 754
import struct

def my_dec_2_ieee(d):
    ieee = ''.join(f'{c:08b}' for c in struct.pack('!d', d))
    return ieee
print(my_dec_2_ieee(1.518484199625))
print(my_dec_2_ieee(-309.141740))
print(my_dec_2_ieee(-25252))

#Smallest Number for Gap of 1
import numpy as np

smallest_gap = np.spacing(1.0)
print(smallest_gap)

#Advantages and Disadvantages of Binary vs. Decimal
#יתרונות בינאריים: גורם לפעולות אריתמטיות ולוגיות פשוטות

#חסרונות בינאריים: יכול להוביל לייצוגים ארוכים יותר עבור מספרים מסוימים

#יתרונות עשרוניים: לרוב ייצוגים קצרים יותר עבור מספרים

#חסרונות עשרוניים: מורכב יותר עבור פעולות אריתמטיות

#Base-1 Representation
#1 בבסיס-13 מציג את-||||||||||| היצוג
#1-סופר בצורה פשוטה את מספר הסימנים בחיבור וכפל בבסיס

#Counting in Binary on Fingers
#כל אצבע מייצגת ספרה בינארית (0 או 1).
#עם 10 אצבעות, המספר הגבוה ביותר שאפשר לספור הוא 2^10 - 1 = 1023.

#Multiplying and Dividing by 2
#הכפלת מספר בינארי ב-2: העבר את כל הביטים שנשארו במיקום אחד.
#חלוקת מספר בינארי ב-2: העבר את כל הביטים ימינה במיקום אחד.
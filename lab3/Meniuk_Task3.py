# Створення прикладних програм на мові Python
# Лабораторна робота №2
# Менюк Володимир, № Заліковки 
from math import *
print ('[1] Створення прикладних програм на мові Python, Лабораторна №3')
print ('[2] Менюк Володимир, № Заліковки')
print('[3] Введіть:')
x = float(input('	x = '))
z = float(input('	z = '))
if not (2*z) or not (1):
	print('Значення змінних виходять за область визначення функції!')
else:
	res = (sqrt(fabs(x**2-4)))/2*z
	print('Результат обчислення = ', res)
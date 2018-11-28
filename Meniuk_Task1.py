cours_name = 'Створення прикладних програм на мові Python'
lab_num = 1
print ('[1] ' + cours_name, lab_num, sep = ', ')

name = 'Vova'
surname = 'Meniuk'
var_num = 'FI-47867'
print ('[2] ' + name + ' ' + surname, var_num, sep = ', ')

teacher = 'Ivan Yaroslavovich'
result = '[3] '
for i in range(45):
	if i == 0:
		result += teacher
	else:
		result += ', ' + teacher
print (result)

x = (176-9.3/7)-(18.2+179.3)-323.8**2
print('[4] x = ', x)

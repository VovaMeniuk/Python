def f(*x, **y):
	res = 0
	if len(x) >= len(y.values()):
		for val in x:
			res += val
		return res/len(x)
	else:
		for val in list(y.values()): 
		 	res += val 
		return res / len(y.values())

x = [-2, 1,0,3,4]
y = {'1':1,'2':-3,'3':4,'4':0}

print(x)
print(list(y.values()))
print(f(*x,**y))

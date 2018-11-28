a = ['h', 'h', 'h', 'h']
b = {'h' : 'H', 'H' : 'h'}
for item in a:
    if item in b:
        a[a.index(item)]=b[item]
        
print (a)
print (b)

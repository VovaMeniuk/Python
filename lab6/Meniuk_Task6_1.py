spysok = input('Введіть список: ').split()
count = 0
for i in spysok:
    if int(i):
        print(i,end=' ')
        count += 1       
print("\nКількість елементів: ", count)

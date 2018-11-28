from math import*
from tkinter import*

root = Tk()
root.title('Meniuk Vova')
root.geometry('500x500')

label=Label(root, text='Graf', font='Arial 14 bold', bg='red')
label.pack(fill=X)

canvas = Canvas(root, height=360, width = 480, bg='black')
canvas.pack()

x0, y0 = 200, 200
x1 =50; x2 = 350; dx=10
canvas.create_line(0, y0,470, y0, fill = 'green', arrow = LAST)
canvas.create_line(x0, 10,x0, 350, fill = 'green', arrow = FIRST)

canvas.create_text(209,210, text='10')
canvas.create_text(300,210, text='100')
canvas.create_text(x0+5,10, text='y', anchor=W)
canvas.create_text(400,200, text='x', anchor = NW)
canvas.create_text(x0+10,100, text='100')

p = 50
while p<=350:
    canvas.create_line(p, 195,p, 205, fill = 'green')
    canvas.create_line(195, p, 205, p, fill = 'green')
    p+=10

points=[]
for x in range(x1+30,x2+dx+30,dx):
    y = y0-(x-x0-30)**2/100
    z = (x,y)
    points.append(z)

print(points)
print(len(points))
canvas.create_line(points, fill='blue', smooth=1, width=2)
button = Button(root, text='Close', command = quit)
button.pack()
root.mainloop()

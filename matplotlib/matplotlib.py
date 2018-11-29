from numpy import *
import matplotlib.pyplot as plt
import math

def f(x):
    return 5*sin(10*x)*sin(3*x)/(x**x)

x=linspace(1,10,250)

plt.plot(x,f(x),label='5*sin(10*x)*sin(3*x)/(x**x)')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.savefig('z12.png',dpi=200)
plt.show()

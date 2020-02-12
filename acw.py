import numpy as np
import matplotlib.pyplot as plt
import math
import random
show_actual = False

noise = lambda : random.gauss(0.0, 0.001) #TODO: Change to your own function using box muller...
euler = lambda x, yn, dy, h : yn + (dy(yn, x) * h)

#given that dy/dt = 2t and y0 = 2 -> y = t^2 + 2
def u(t):
    if t <= 5:
        return 2
    elif t <= 10:
        return 1
    elif t <= 15:
        return 3
    raise Exception("Not implemented")

dy = lambda y, t :  (-2 * y) + (2 * u(t))

x0 = 1
x0_noise = 1
#0.5 is poor
h = 0.01

simulation_length = 15
x = list()
y = list()
y_noise = list()
f = list()

for k in range(0, int(simulation_length / h) + 1):
    t = k * h
    x.append(t)
    y.append(x0)
    y_noise.append(x0_noise)
    x0 = euler(t, x0, dy, h)
    x0_noise = euler(t, x0_noise, dy, h) + noise()
    f.append((-2 * t))

print("Summary:")
for a, b in zip(x, y):
    print(a, b, sep="\t\t\t\t\t\t\t")

plt.plot(x, y)
plt.plot(x, y_noise)
plt.plot(x, [u(t) for t in x], color="green")
if show_actual:
    plt.plot(x, [u(t) - math.exp(-2 * t) for t in x], color="black")
plt.legend(["~f(x)", "~f(x) + N", "f(x)"])
plt.grid(which="major", color="r")
plt.grid(which="minor", color="b")
plt.show()
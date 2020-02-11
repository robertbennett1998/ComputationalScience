import numpy as np
import matplotlib.pyplot as plt
import math
import random

# box_muller = lambda mu, sigma : 
# random.uniform(0.0, 1.0)

noise = lambda : random.gauss(0.0, 0.001)
euler = lambda t, h, y, dy : y +  (((-2 * y) + dy(t)) * h)

#0.5 is poor

#given that dy/dt = 2t and y0 = 2 -> y = t^2 + 2
def u(t):
    if t <= 5:
        return 2
    elif t <= 10:
        return 1
    elif t <= 15:
        return 3
    raise Exception("Not implemented")

dy = lambda t : 2 * u(t)

x0 = 0
x0_noise = 0
h = 0.1

simulation_length = 15
x = list()
y_noise = list()
y = list()
f = list()

for k in range(0, int(simulation_length / h) + 1):
    t = k * h
    x.append(t)
    y.append(x0)
    y_noise.append(x0_noise)
    x0 = euler(t, h, x0, dy)
    x0_noise = euler(t, h, x0_noise, dy) + noise()
    f.append((-2 * t))

print("Summary:")
for a, b in zip(x, y):
    print(a, b, sep="\t\t\t\t\t\t\t")

plt.plot(x, y)
plt.plot(x, y_noise)
plt.plot(x, [dy(t) for t in x], color="g")
#plt.plot(x, f, color="g")
plt.legend(["y", "y + noise", "dy"])#, "f"])
plt.grid(which="major", color="r")
plt.grid(which="minor", color="b")
plt.show()
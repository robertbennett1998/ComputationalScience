import numpy as np
import matplotlib.pyplot as plt

s0 = -5
s1 = 5
f0 = 10
f1 = 10
a0 = 1
a1 = -1
mass_loss_factor = .1

masses0 = list()
masses1 = list()

def m0(t):
    loss = ((mass_loss_factor + 1) * t**2)

    r = 25 - loss

    if (loss >= 25):
        return None

    if r < 0.1:
        return None

    return r

def m1(t):
    loss = (mass_loss_factor * t**2)
    if (loss >= 15):
        return None
    m = 15 - loss
    if m < 0.1:
        return None

    return m

forward_difference = lambda x, h, f : (f(x + h) - f(x)) / h

euler = lambda x, h, y, dy : y + (dy(x) * h)

def dy0(t):
    m = m0(t)
    masses0.append(m)
    print(m)
    if (m == None):
        print("zero 0")
        return 0
    return (a0*(f0/m) * t)

def dy1(t):
    m = m1(t)
    masses1.append(m)
    if (m == None):
        print("zero 1")
        return 0
    return (a1*(f1/m) * t)

h = 0.01
x = list()
y0 = list()
y1 = list()
sim_length = 15
e = 0.1
for k in range(0, int(sim_length / h) + 1):
    t = k * h
    x.append(t)
    y0.append(s0)
    y1.append(s1)
    s0 = euler(t, h, s0, dy0)
    s1 = euler(t, h, s1, dy1)
    if abs(s1 - s0) <= e:   
        #print("swap")
        a0 *= -1
        a1 *= -1

plt.plot(x, y0)
plt.plot(x, y1)
plt.legend(["m0", "m1"])
plt.grid(which="major", color="r")
plt.grid(which="minor", color="b")
plt.show()

plt.plot(x, masses0)
plt.plot(x, masses1)
plt.grid(which="major", color="r")
plt.grid(which="minor", color="b")
plt.show()
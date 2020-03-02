import numpy as np
import matplotlib.pyplot as plt
import math
import random
show_actual = True
show_u = True

noise = lambda : random.gauss(0.0, 0.001) #TODO: Change to your own function using box muller...
euler = lambda x, yn, dy, h : yn + (dy(yn, x) * h)

def run_simulation(y0, step_length, sim_length, dy, noise=lambda : 0, method=euler) -> (list, list):
    x = list()
    y = list()

    for k in range(0, int(sim_length / step_length) + 1):
        t = k * step_length
        x.append(t)
        y.append(y0)
        y0 = euler(t, y0, dy, step_length)

    return x, y

def u(t):
    if t <= 5:
        return 2
    elif t <= 10:
        return 1
    elif t <= 15:
        return 3
    raise Exception("Not implemented")
dy = lambda y, t :  (-2 * y) + (2 * u(t))
h = 0.4
simulation_length = 15

x_small_values, y_small_values = run_simulation(y0=1, step_length=h, sim_length=simulation_length, dy=dy)
actual_small_values = [u(t) - math.exp(-2 * t) for t in x_small_values]
h = 0.01
x_small_values_small_h, y_small_values_small_h = run_simulation(y0=1, step_length=h, sim_length=simulation_length, dy=dy)

def u(t):
    if t <= 5:
        return 2000000
    elif t <= 10:
        return 1000000
    elif t <= 15:
        return 3000000
    raise Exception("Not implemented")



h = 0.4
x_large_values, y_large_values = run_simulation(y0=1, step_length=h, sim_length=simulation_length, dy=dy)
actual_large_values = [u(t) - math.exp(-2 * t) for t in x_large_values]
h = 0.01
x_large_values_small_h, y_large_values_small_h = run_simulation(y0=1, step_length=h, sim_length=simulation_length, dy=dy)
actual_large_values_small_h = [u(t) - math.exp(-2 * t) for t in x_large_values_small_h]
color = 'tab:red'
fig, small_ax = plt.subplots()
small_ax.set_xlabel("time / s")
small_ax.set_ylabel("displacement for small distances between input (m)")
small_ax.plot(x_small_values, y_small_values, color=color)
color = 'b'
small_ax.plot(x_small_values_small_h, y_small_values_small_h, color=color)
small_ax.tick_params(axis='y', labelcolor=color)

large_ax = small_ax.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:blue'
large_ax.set_ylabel("displacement for large distances between input (m)")
large_ax.plot(x_large_values, y_large_values, color=color)
color = 'tab:green'
large_ax.plot(x_large_values_small_h, y_large_values_small_h, color=color)
large_ax.tick_params(axis='y', labelcolor=color)

plt.show()

abs_error = lambda actual_value, simulated_value : abs(actual_value - simulated_value)
avg_abs_error = lambda actual_values, simulated_values : sum([abs_error(actual_values[i], simulated_values[i]) for i in range(len(actual_values))], 0) / len(actual_values)
avg_abs_error_over_t = lambda actual_values, simulated_values : [avg_abs_error(actual_values[0:i], simulated_values[0:i]) for i in range(1, len(actual_values))]

#plt.plot(x_small_values[1:], avg_abs_error_over_t(actual_small_values, y_small_values))
plt.plot(x_large_values[1:], avg_abs_error_over_t(actual_large_values, y_large_values))
print(actual_large_values[:5])
print(y_large_values[:5])
plt.plot(x_large_values_small_h[1:], avg_abs_error_over_t(actual_large_values_small_h, y_large_values_small_h))
plt.legend(["LV LH", "LV SH"])
plt.show()
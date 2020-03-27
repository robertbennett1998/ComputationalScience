import numpy as np
import matplotlib.pyplot as plt
import math
import random
import pandas as pd
import os
show_actual = True
show_u = False

def save(x, y, path):
    results = pd.DataFrame({ "x": x, "y": y})
    results.to_csv(path)
    return results

euler = lambda x, yn, dy, h : yn + (dy(yn, x) * h)
def run_simulation(y0, step_length, sim_length, dy, noise=lambda t : 0, method=euler) -> (list, list):
    x = list()
    y = list()

    for k in range(0, int(sim_length / step_length) + 1):
        t = k * step_length
        x.append(t)
        y.append(y0)
        y0 = euler(t, y0, dy, step_length) + noise(t)

    return x, y

#Actual function
def lock(t):
    if t <= 5:
        return t
    elif t <= 10:
        return t - 5
    elif t <= 15:
        return t - 10
def flip(t):
    if t <= 5:
        return 1
    elif t <= 10:
        return -1
    elif t <= 15:
        return 1
def u(t):
    if t <= 5:
        return 2
    elif t <= 10:
        return 1
    elif t <= 15:
        return 3
    raise Exception("Not implemented")
#actual = [u(t) - (flip(t) * math.exp(-2 * lock(t))) for t in x]

save_dir = os.path.join(os.getcwd(), "results")
if not os.path.exists(save_dir):
    os.mkdir(save_dir)

def run_simulations(step_lengths, noise=None, save_results=False, sample_interval=0.1, save_sample=False):
    results = list()

    dy = lambda y, t :  (-2 * y) + (2 * u(t))
    simulation_length = 15
    y0 = 1
    filename = "result_"
    if noise is None:
        filename = "result_no_noise_"
        noise = lambda t : 0

    for h in step_lengths:
        x, y = run_simulation(y0, h, simulation_length, dy, noise)
        results.append((x,  y))

        if save_results:
            save(x, y, os.path.join(save_dir, filename + str(h).replace(".", "_") + ".csv"))

        if save_sample:
            x_sample = list()
            y_sample = list()
            iter_count = sample_interval / h
            count = iter_count
            for _x, _y in zip(x, y):
                if count >= iter_count:
                    x_sample.append(_x)
                    y_sample.append(_y)
                    count = 0
                count += 1
            save(x_sample, y_sample, os.path.join(save_dir, filename + str(h).replace(".", "_") + "_sample" + ".csv"))

    return results

#noise = lambda t : random.gauss(0.0, 0.001) #TODO: Change to your own function using box muller...

cached_noise = None
use_cached_noise = False
def noise(t=None, mu=0.0, sigma=0.01):
    if not use_cached_noise:
        u0 = random.uniform(0, 1)
        u1 = random.uniform(0, 1)
        z0 = math.sqrt(-2 * math.log(u0)) * math.sin(2*math.pi*u1)
        z1 = math.sqrt(-2 * math.log(u0)) * math.cos(2*math.pi*u1)
        cached_noise = z1

        return (z0 * sigma) + mu

    return (cached_noise * sigma) + mu

h_values = [0.1]

results = run_simulations(h_values, noise=noise, save_sample=False, save_results=True)
actual_function = lambda t : u(t) - (flip(t) * math.exp(-2 * lock(t)))

for result in results:
    plt.plot(result[0], result[1])

plt.plot(results[-1][0], [actual_function(t) for t in results[-1][0]])
h_values = list(h_values)
h_values.append("actual")
plt.legend(h_values)
plt.show()

to_plt = 0
marker = None #'o'
plt.plot(results[to_plt][0], results[to_plt][1], marker=marker)
plt.plot(results[to_plt][0], [actual_function(t) for t in results[to_plt][0]], marker=marker)
plt.plot(results[to_plt][0], [(y - actual_function(t)) for t, y in zip(results[to_plt][0], results[to_plt][1])], marker=marker)
plt.legend(["simulated", "actual", "absolute error"])
plt.show()

print(actual_function(0.2))
print(results[to_plt][0][2])
print(results[to_plt][1][1])
print(results[to_plt][1][2] - actual_function(0.2))
import numpy as np
import matplotlib.pyplot as plt
import math
import random
import pandas as pd
import os
show_actual = True
show_u = False

def u(t):
    if t <= 5:
        return 2
    elif t <= 10:
        return 1
    elif t <= 15:
        return 3

def actual_function(t):
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

        raise Exception("Not implemented")
    return u(t) - (flip(t) * math.exp(-2 * lock(t)))

def save(x, y, path):
    results = pd.DataFrame({ "x": x, "y": y})
    results["actual"] = results["x"].apply(actual_function)
    results["absolute_error"] = results["actual"] - results["y"]
    results.to_csv(path)
    return results

save_dir = os.path.join(os.getcwd(), "results")
if not os.path.exists(save_dir):
    os.mkdir(save_dir)

euler = lambda x, yn, dy, h : yn + (dy(yn, x) * h)
def run_simulation(y0, step_length, sim_length, dy, method=euler) -> (list, list):
    x = list()
    y = list()

    for k in range(0, int(sim_length / step_length) + 1):
        t = k * step_length
        x.append(t)
        y.append(y0)
        y0 = euler(t, y0, dy, step_length)

    return x, y

def run_simulations(step_lengths, save_results=False):
    results = list()

    dy = lambda y, t :  (-2 * y) + (2 * u(t))
    simulation_length = 15
    y0 = 1
    filename = "result_"

    for h in step_lengths:
        x, y = run_simulation(y0, h, simulation_length, dy)
        results.append((x,  y))

        if save_results:
            save(x, y, os.path.join(save_dir, filename + str(h).replace(".", "_") + ".csv"))

    return results

h_values = [1.0, 0.75, 0.5, 0.25, 0.2, 0.1, 0.01]

results = run_simulations(h_values, save_results=True)

#Results plot
for result in results:
    plt.plot(result[0], result[1])

plt.plot(results[-1][0], [actual_function(t) for t in results[-1][0]])
h_values = list(h_values)
h_values.append("actual")
plt.legend(h_values)
plt.show()

#Error plot
to_plt = 4
marker = None #'o'
plt.plot(results[to_plt][0], results[to_plt][1], marker=marker)
plt.plot(results[to_plt][0], [actual_function(t) for t in results[to_plt][0]], marker=marker)
plt.plot(results[to_plt][0], [(actual_function(t) - y) for t, y in zip(results[to_plt][0], results[to_plt][1])], marker=marker)
plt.legend(["simulated (h=0.1)", "actual", "absolute error"])
plt.show()


#Error plot 2
for result in results:
    plt.plot(result[0], [(actual_function(t) - y) for t, y in zip(result[0], result[1])])

plt.legend(h_values)
plt.show()
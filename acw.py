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

def run_simulations(step_lengths, noise=None, save_results=False):
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

    return results

noise = lambda t : random.gauss(0.0, 0.001) #TODO: Change to your own function using box muller...
h_values = [0.75, 0.55, 0.51, 0.5, 0.45, 0.3, 0.15] #from 1.5 to 

results = run_simulations(h_values, noise, save_results=True)

for result in results:
    plt.plot(result[0], result[1])
plt.plot(results[-1][0], [u(t) - (flip(t) * math.exp(-2 * lock(t))) for t in results[-1][0]])
h_values = list(h_values)
h_values.append("actual")
plt.legend(h_values)
plt.show()

# plt.plot(x, y)
# if show_u:
#     plt.plot(x, [u(t) for t in x], color="green")

# if show_actual:
#     plt.plot(x, actual, color="black")
# plt.legend(["~f(x) + N(0, 0.001)", "f(x)"])
# plt.grid(which="major", color="r")
# plt.grid(which="minor", color="b") 
# plt.show()

# abs_error = lambda actual_value, simulated_value : abs(actual_value - simulated_value)
# avg_abs_error = lambda actual_values, simulated_values : sum([abs_error(actual_values[i], simulated_values[i]) for i in range(len(actual_values))], 0) / len(actual_values)
# avg_abs_error_over_t = lambda actual_values, simulated_values : [avg_abs_error(actual_values[0:i], simulated_values[0:i]) for i in range(1, len(actual))]

# plt.plot(x[1:], avg_abs_error_over_t(actual, y))
# plt.show()
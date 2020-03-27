import pandas as pd
import os
import math
import random
import glob
import matplotlib.pyplot as plt

results_dir = os.path.join(os.getcwd(), "results")
if not os.path.exists(results_dir):
    raise Exception("Results directory doesn't exist")

cached_noise = None
use_cached_noise = False
def box_muller(x, mu=0.0, sigma=0.01):
    if not use_cached_noise:
        u0 = random.uniform(0, 1)
        u1 = random.uniform(0, 1)
        z0 = math.sqrt(-2 * math.log(u0)) * math.sin(2*math.pi*u1)
        z1 = math.sqrt(-2 * math.log(u0)) * math.cos(2*math.pi*u1)
        cached_noise = z1

        return x + ((z0 * sigma) + mu)

    return x + ((cached_noise * sigma) + mu)

legend = list()
for filepath in glob.glob(os.path.join(os.getcwd(), "results/*.csv")):
    results = pd.read_csv(filepath)
    results["y"] = results["y"].apply(box_muller)
    results.to_csv(filepath)
    plt.plot(results["x"].values, results["y"].values)
    legend.append(os.path.basename(filepath))

plt.legend(legend)
plt.show()
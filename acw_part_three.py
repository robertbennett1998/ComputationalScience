import pandas as pd
import numpy as np
import os
from neuron import Neuron
import math
import matplotlib.pyplot as plt
from sklearn import preprocessing

results = pd.read_csv(os.path.join(os.getcwd(), "results/result_0_01.csv"))

linear = lambda x : x
linear_cost_derivative = lambda x, w, y, y_prime : (y_prime - y) * -x
sigmoid = lambda x : 1 / (1 + math.exp(-x))
sigmoid_cost_derivative = lambda x, w, y, y_prime : (y_prime - y) * (-x * ( y * (1 - y)))

n = 3
preceptron_sigmoid = Neuron(input_dimension=n, use_bias=True, learning_rate=0.0001, activation_function=sigmoid, cost_function_derivative=sigmoid_cost_derivative)#lambda x, w, y, y_prime : y_prime - y)
print("Initial Weights (sigmoid):", preceptron_sigmoid._weights)

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

def get_tuple(results, i, n):
    data = list()
    for j in range(i-n, i):
        if j < 0:
            data.append(1)
        else:
            data.append(results["y"].values[j])

    return data

training_data = list()
training_labels = list()
for i in range(len(results["y"].values)):
    training_data.append(get_tuple(results, i, n))
    training_labels.append(results["y"].values[i])

def map_range(x, in_min, in_max, out_min, out_max):
    return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min

training_data_sigmoid = training_data.copy()
for data_point in training_data:
    for x in data_point: 
        x = x

training_labels_sigmoid = training_labels.copy()

min_max_scaler = preprocessing.MinMaxScaler()
training_labels_sigmoid = min_max_scaler.fit_transform(results[['y']].values.astype(float))

preceptron_sigmoid.train(training_data_sigmoid, training_labels_sigmoid, epochs=10)

print("Trained Weights (sigmoid):", preceptron_sigmoid._weights)

min_max_scaler = preprocessing.MinMaxScaler(feature_range=(1, 3))

plt.plot(results["x"].values, min_max_scaler.fit_transform(np.array([preceptron_sigmoid.predict(get_tuple(results, i, n)) for i in range(len(results["y"].values))]).reshape(-1, 1)))
plt.plot(results["x"].values, results["y"].values)
plt.plot(results["x"].values, [actual_function(x) for x in results["x"].values])

plt.legend(["preceptron (sigmoid)", "simulated function", "actual function"])
plt.show()
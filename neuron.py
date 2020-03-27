#%%imports
import numpy as np
import random as rand
import math
import matplotlib.pyplot as plt

#%% Preceptron class definition
class Neuron():
    def __init__(self, input_dimension, learning_rate, activation_function, cost_function_derivative, use_bias=True):
        self._learning_rate = learning_rate
        self._activation_function = activation_function
        self._cost_function_derivative = cost_function_derivative
        self._use_bias = use_bias

        self._weights = np.array([rand.gauss(0.0, 0.01) for i in range(input_dimension + (1 if use_bias else 0))])
        if use_bias:
            self._weights[0] = 0

    def train(self, training_data, training_labels, epochs=10):
        for i in range(epochs):
            for x, y_prime in zip(training_data, training_labels):
                x = np.array(x)
                #print(x)

                if self._use_bias:
                    x = np.insert(x, 0, 1)

                y = self._activation_function((self._weights @ x.transpose()))
                delta = -self._learning_rate * self._cost_function_derivative(x=x, w=self._weights, y=y, y_prime=y_prime)

                self._weights += delta
                #print("X:", x, "Y:", y, "Expected Y:", y_prime, "Delta:", delta, "Weights:", self._weights)

    def predict(self, x): 
        x = np.array(x)

        if self._use_bias:
            x = np.insert(x, 0, 1)

        return self._activation_function(self._weights @ x.transpose())

# #%% OR preceptron example
# print("OR example:")
# training_data = list()
# training_labels = list()
# training_example_count = 100

# for i in range(training_example_count):
#     sample = (rand.randint(0, 1), rand.randint(0, 1))
#     training_data.append(sample)
#     if sample[0] == 1 or sample[1] == 1:
#         training_labels.append(1)
#     else:
#         training_labels.append(0)


# preceptron = Neuron(input_dimension=2, learning_rate=0.01, activation_function=lambda x : 1.0 if x >= 0 else 0.0, cost_function_derivative=lambda x, w, y, y_prime : ((y_prime - y) * x))
# preceptron.train(training_data, training_labels)
# print(preceptron.predict((0, 0)))
# print(preceptron.predict((1, 0)))
# print(preceptron.predict((0, 1)))
# print(preceptron.predict((1, 1)))

# #%% AND preceptron example
# print("AND example:")
# training_data = list()
# training_labels = list()
# training_example_count = 100

# for i in range(training_example_count):
#     sample = (rand.randint(0, 1), rand.randint(0, 1))
#     training_data.append(sample)
#     if sample[0] == 1 and sample[1] == 1:
#         training_labels.append(1)
#     else:
#         training_labels.append(0)


# preceptron = Neuron(input_dimension=2, learning_rate=0.01, activation_function=lambda x : 1.0 if x >= 0 else 0.0, cost_function_derivative=lambda x, w, y, y_prime : ((y_prime - y) * x))
# preceptron.train(training_data, training_labels)
# print(preceptron.predict((0, 0)))
# print(preceptron.predict((1, 0)))
# print(preceptron.predict((0, 1)))
# print(preceptron.predict((1, 1)))
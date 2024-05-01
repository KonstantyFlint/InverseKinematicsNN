from typing import List

import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


def initialize_weights(input_size, output_size):
    return np.random.randn(input_size, output_size)


class NeuralNetwork:
    def __init__(self, layer_sizes: List[int]):
        self.weights = []
        self.biases = []
        self.pre_activation = []
        self.post_activation = []
        for i in range(1, len(layer_sizes)):
            input_size = layer_sizes[i - 1]
            output_size = layer_sizes[i]
            self.weights.append(initialize_weights(input_size, output_size))
            self.biases.append(np.zeros((1, output_size)))

    def forward(self, X):
        self.pre_activation = [X, ]
        self.post_activation = [X, ]

        for i in range(len(self.weights)):
            weights = self.weights[i]
            bias = self.biases[i]
            self.pre_activation.append(np.dot(self.post_activation[-1], weights) + bias)
            self.post_activation.append(sigmoid(self.pre_activation[-1]))
        return self.post_activation[-1]

    def backward(self, X, y, learning_rate):
        error = y - self.post_activation[-1]
        deltas = []
        for i in range(len(self.weights)):
            weights = self.weights[-i - 1]
            output = self.post_activation[-i - 1]
            deltas.append(error * sigmoid_derivative(output))
            error = deltas[-1].dot(weights.T)

        for i in range(len(deltas)):
            delta = deltas[-i - 1]
            self.weights[i] += self.post_activation[i].T.dot(delta) * learning_rate
            self.biases[i] += np.sum(delta, axis=0, keepdims=True) * learning_rate

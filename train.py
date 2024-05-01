import numpy as np

from input_output import save, load
from neural_network import NeuralNetwork
from test_generator import generate_test_cases

input_size = 2
hidden_size1 = 4
hidden_size2 = 3
output_size = 1

FILENAME = "network.pickle"

#nn = NeuralNetwork([2, 10, 25, 50, 25, 10, 2])

nn = load(FILENAME)

epochs = 10000
learning_rate = 0.0001
for epoch in range(epochs):
    X, y = generate_test_cases(1000)
    output = nn.forward(X)
    nn.backward(X, y, learning_rate)

    if epoch % 1000 == 0:
        error = np.mean(np.square(y - output))
        print(error)

save(nn, FILENAME)

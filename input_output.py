from neural_network import NeuralNetwork
import pickle


def save(network: NeuralNetwork, filename) -> None:
    with open(filename, 'wb') as file:
        pickle.dump(network, file)


def load(filename) -> NeuralNetwork:
    with open(filename, 'rb') as file:
        return pickle.load(file)

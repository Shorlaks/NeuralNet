import math

# Functions


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def relu(x):
    return x if x > 0 else 0


def tanh(x):
    return math.tanh(x)

# Derivatives


def sigmoid_derivative(x):
    return x * (1 - x)


def relu_derivative(x):
    return 1 if x > 0 else 0


def tanh_derivative(x):
    return 1 - math.pow(math.tanh(x), 2)

import random
import math
import activation_functions


class NeuralNet:
    def __init__(self, layers):
        self._layers = self.init_layers(layers)
        self._weights = self.init_weights()
        self._activations = self.init_activations()

    def init_activations(self):
        activations = [[0 for _ in range(self._layers[i+1])]
                       for i in range(len(self._layers)-1)]
        return activations

    def init_weights(self):
        weights = [[[random.uniform(-1, 1) for _ in range(self._layers[i])]
                    for __ in range(self._layers[i+1])]
                   for i in range(len(self._layers)-1)]
        return weights

    @staticmethod
    # checking user's input for correctness
    def init_layers(layers):
        try:
            assert isinstance(layers, list), "layers should be a list"
            for obj in layers:
                assert isinstance(obj, int), "'layers' items should be ints"
            assert len(layers) > 1, "should be at least 2 layers"
        except AssertionError as e:
            raise Exception(e)
        return layers

    def predict(self, input_list):
        self.forward_pass(input_list)
        return self._activations[-1]

    def train(self, input_data_set, labels):
        for i, input_list in enumerate(input_data_set):
            self.forward_pass(input_list)
            self.calculate_error(labels[i])
            self.backward_pass()

    def forward_pass(self, input_list):
        for i in range(len(self._activations[0])):
            dot_product = self.lists_dot_product(input_list, self._weights[0][i])
            self._activations[0][i] = activation_functions.sigmoid(dot_product)
        for i in range(1, len(self._activations)):
            for j in range(len(self._activations[i])):
                dot_product = self.lists_dot_product(self._activations[i-1], self._weights[i][j])
                self._activations[i][j] = activation_functions.sigmoid(dot_product)

    def binary_cross_entropy(self, label):
        if label == 0:
            return -math.log(1-p)
        elif label == 1:
            return -math.log(p)


    def calculate_error(self, label):
        pass

    def backward_pass(self):
        pass

    @staticmethod
    def lists_dot_product(l1, l2):
        return sum(i[0] * i[1] for i in zip(l1, l2))


nn = NeuralNet([4, 8, 2])
nn.predict([0.5, 0.5, 0.5, 0.5])
pass

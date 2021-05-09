import random
import activation_functions
import loss_functions
import math


class NeuralNet:
    def __init__(self, layers, funcs, lose_func):
        self._layers = self.init_layers(layers)
        self._activation_functions = self.init_activation_functions(funcs)
        self._weights = self.init_weights()
        self._biases = self.init_biases()
        self._delta_weights = self.init_delta_weights()
        self._activations = self.init_activations()
        self._errors = self.init_errors()
        self._loss_function = lose_func
        self._learning_rate = 0.8
        self._loss_function_derivative_dispatcher = {'se': loss_functions.squared_error_derivative,
                                                     'ae': loss_functions.absolute_error_derivative,
                                                     'sle': loss_functions.squared_logarithmic_error_derivative,
                                                     'bce': loss_functions.binary_cross_entropy_derivative
                                                     }
        self._activation_function_dispatcher = {'sigmoid': activation_functions.sigmoid,
                                                'relu': activation_functions.relu,
                                                'tanh': activation_functions.tanh,
                                                'leaky_relu': activation_functions.leaky_relu
                                                }
        self._activation_function_derivative_dispatcher = {'sigmoid': activation_functions.sigmoid_derivative,
                                                           'relu': activation_functions.relu_derivative,
                                                           'tanh': activation_functions.tanh_derivative,
                                                           'leaky_relu': activation_functions.leaky_relu_derivative
                                                           }

    # checking user's input for correctness
    @staticmethod
    def init_layers(layers):
        try:
            assert isinstance(layers, list), "layers should be a list"
            for obj in layers:
                assert isinstance(obj, int), "'layers' items should be ints"
            assert len(layers) > 1, "should be at least 2 layers"
        except AssertionError as e:
            raise Exception(e)
        return layers

    @staticmethod
    def init_activation_functions(funcs):
        return funcs

    def init_activations(self):
        activations = [[0 for _ in range(self._layers[i+1])]
                       for i in range(len(self._layers)-1)]
        return activations

    def init_errors(self):
        errors = [[0 for _ in range(self._layers[i+1])]
                  for i in range(len(self._layers)-1)]
        return errors

    def init_weights(self):
        weights = [[[random.uniform(-1, 1) for _ in range(self._layers[i])]
                    for __ in range(self._layers[i+1])]
                   for i in range(len(self._layers)-1)]
        return weights

    def init_biases(self):
        biases = [[random.uniform(-1, 1) for _ in range(self._layers[i+1])]
                  for i in range(len(self._layers)-1)]
        return biases

    def init_delta_weights(self):
        weights = [[[0 for _ in range(self._layers[i])]
                    for __ in range(self._layers[i+1])]
                   for i in range(len(self._layers)-1)]
        return weights

    def predict(self, input_list):
        self.forward_pass(input_list)
        return self._activations[-1]

    def train(self, input_data_set, labels, epochs=1):
        for j in range(epochs):
            for i, input_list in enumerate(input_data_set):
                self.forward_pass(input_list)
                self.backward_pass(labels[i], self._activations[-1], input_list)

    def forward_pass(self, input_list):
        for i in range(len(self._activations[0])):
            dot_product = self.lists_dot_product(input_list, self._weights[0][i]) + self._biases[0][i]
            self._activations[0][i] = self._activation_function_dispatcher[self._activation_functions[0]](dot_product)
        for i in range(1, len(self._activations)):
            for j in range(len(self._activations[i])):
                dot_product = self.lists_dot_product(self._activations[i-1], self._weights[i][j]) + self._biases[i][j]
                self._activations[i][j] = self._activation_function_dispatcher[self._activation_functions[i]](dot_product)

    def calculate_output_layer_errors(self, label, output):
        for i in range(len(self._errors[-1])):
            self._errors[-1][i] = self._loss_function_derivative_dispatcher[self._loss_function](label[i], output[i])
            self._errors[-1][i] *= self._activation_function_derivative_dispatcher[self._activation_functions[-1]](self._activations[-1][i])

    def backward_pass(self, label, output, training_input):
        self.calculate_output_layer_errors(label, output)
        for i in range(len(self._activations) - 1, 0, -1):
            self.calculate_delta_weights(i)
            self.calculate_hidden_layer_errors(i-1)
        self.calculate_input_layer_delta_weights(training_input)
        self.update_weights()
        self.update_biases()

    def calculate_delta_weights(self, layer):
        for i in range(len(self._weights[layer])):
            for j in range(len(self._weights[layer][i])):
                dw = self._activations[layer-1][j]
                self._delta_weights[layer][i][j] = dw * self._errors[layer][i]

    def calculate_input_layer_delta_weights(self, training_input):
        for i in range(len(self._weights[0])):
            for j in range(len(self._weights[0][i])):
                dw = training_input[j]
                self._delta_weights[0][i][j] = dw * self._errors[0][i]

    def calculate_hidden_layer_errors(self, layer):
        for i in range(len(self._errors[layer])):
            for j in range(len(self._errors[layer+1])):
                self._errors[layer][i] += self._errors[layer+1][j] * self._weights[layer+1][j][i]
            self._errors[layer][i] *= self._activation_function_derivative_dispatcher[self._activation_functions[layer]](self._activations[layer][i])

    def update_weights(self):
        for i in range(len(self._weights)):
            for j in range(len(self._weights[i])):
                for k in range(len(self._weights[i][j])):
                    self._weights[i][j][k] -= self._learning_rate * self._delta_weights[i][j][k]

    def update_biases(self):
        for i in range(len(self._biases)):
            for j in range(len(self._biases[i])):
                self._biases[i][j] -= self._learning_rate * self._biases[i][j] * self._errors[i][j]

    @staticmethod
    def lists_dot_product(l1, l2):
        s = sum(i[0] * i[1] for i in zip(l1, l2))
        if math.isinf(s):
            return 0
        return s

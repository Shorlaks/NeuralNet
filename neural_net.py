import random
import activation_functions
import loss_functions


class NeuralNet:
    def __init__(self, layers):
        self._layers = self.init_layers(layers)
        self._weights = self.init_weights()
        self._delta_weights = self.init_delta_weights()
        self._activations = self.init_activations()
        self._errors = self.init_errors()
        self._loss_function = 'se'
        self._learning_rate = 0.5
        self._loss_function_dispatcher = {'se': loss_functions.squared_error,
                                          'ae': loss_functions.absolute_error,
                                          'sle': loss_functions.squared_logarithmic_error,
                                          'binary_cross_entropy': loss_functions.binary_cross_entropy,
                                          'multi_class_cross_entropy': loss_functions.multi_class_cross_entropy
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

    def init_delta_weights(self):
        weights = [[[0 for _ in range(self._layers[i])]
                    for __ in range(self._layers[i+1])]
                   for i in range(len(self._layers)-1)]
        return weights

    def predict(self, input_list):
        self.forward_pass(input_list)
        return self._activations[-1]

    def train(self, input_data_set, labels):
        for i, input_list in enumerate(input_data_set):
            self.forward_pass(input_list)
            # error = self.calculate_total_error(labels[i], self._activations[-1])
            self.backward_pass(labels[i], self._activations[-1], input_list)

    def forward_pass(self, input_list):
        for i in range(len(self._activations[0])):
            dot_product = self.lists_dot_product(input_list, self._weights[0][i])
            self._activations[0][i] = activation_functions.sigmoid(dot_product)
        for i in range(1, len(self._activations)):
            for j in range(len(self._activations[i])):
                dot_product = self.lists_dot_product(self._activations[i-1], self._weights[i][j])
                self._activations[i][j] = activation_functions.sigmoid(dot_product)

    def calculate_output_layer_errors(self, label, output):
        for i in range(len(self._errors[-1])):
            self._errors[-1][i] = loss_functions.binary_cross_entropy_derivative(label[i], output[i])
            self._errors[-1][i] *= activation_functions.sigmoid_derivative(self._activations[-1][i])

    def backward_pass(self, label, output, training_input):
        self.calculate_output_layer_errors(label, output)
        for i in range(len(self._activations) - 1, 0, -1):
            self.calculate_delta_weights(i)
            self.calculate_hidden_layer_errors(i-1)
        self.calculate_input_layer_delta_weights(training_input)
        self.update_weights()

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
            self._errors[layer][i] *= activation_functions.sigmoid_derivative(self._activations[layer][i])

    def update_weights(self):
        for i in range(len(self._weights)):
            for j in range(len(self._weights[i])):
                for k in range(len(self._weights[i][j])):
                    self._weights[i][j][k] -= self._learning_rate * self._delta_weights[i][j][k]

    @staticmethod
    def lists_dot_product(l1, l2):
        return sum(i[0] * i[1] for i in zip(l1, l2))


data = []
labels = []
path = r"C:\Users\dank1\Desktop\pima.txt"
file = open(path, 'r')
lines = file.readlines()
for line in lines:
    arr = line.split(',')
    line_data = []
    for i in range(len(arr) - 1):
        line_data.append(float(arr[i]))
    data.append(line_data)
    labels.append([float(arr[-1][:1])])

for i in range(8):
    max = 0
    for d in data:
        if d[i] > max:
            max = d[i]
    for d in data:
        d[i] /= max


nn = NeuralNet([8, 8, 1])
nn.train(data, labels)

p = [0.35294117647058826, 0.7437185929648241, 0.5901639344262295, 0.35353535353535354, 0.0, 0.5007451564828614, 0.2590909090909091, 0.6172839506172839]
print(nn.predict(p))


print("finished")

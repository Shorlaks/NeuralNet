# Neural-Network (Python)
This is my implementation of a fully connected feedforward neural network using only lists (no numpy).

I noticed that the majority of neural network implementations on gitHub have only 3 layers (input, hidden output), 1 type of activation function (sigmoid) and 1 type of cost function (squared error), this minimalistic approach in my opinion prevents programmers from exploring the full potential of deep learning. 
For those reasons, I'm working on making this neural network as flexiable as possible by implementing several cost functions, different activation functions and unrestricting the depth of the network, giving the programmer the possibility to view how different setups affect the training process.

### Usage
To use the NeuralNet class, first import NeuralNet from neural_net.py:

`from neural_net import NeuralNet`

You can now create an instance of the NeuralNet class. The constructor takes three parameters:
- `layers`: A list representing the number of layers which contains the number of neurons for each layer.
- `activations`: A list containing the activation function for each layer.
- `cost_function` A string representing the desired loss function.

For example:

`nn = NeuralNet([8, 12, 8, 1], ["sigmoid", "relu", "sigmoid"], "se")`

As of now, 4 activation functions are supported:
- `"sigmoid"` - sigmoid.
- `"tanh"` - hyperbolic tangent.
- `"relu"` - rectified linear unit.
- `"leaky_relu"` - leaky rectified linear unit.

And 4 cost functions are supported:
- `"se"` - squared error.
- `"ae"` - absolute error.
- `"sle"` - squared logarithmic error.
- `"bce"` - binary cross entropy.

### Training
You can train the neural network using the `train` method. This method takes three parameters:
- `"data_set"` - A list of lists, where each sub-list contains the input vector.
- `"labels"` - A list of lists, where each sub-list contains the labels.
- `"epochs"` - An integer representing the number of times the entire dataset passes forwards and backwards through the network.

For example:
```
dataset = [[6, 148, 72, 35, 0, 33.6, 0.627, 50],
           [1, 85, 66, 29, 0, 26.6, 0.351, 31],
           [8, 183, 64, 0, 0, 23.3, 0.672, 32]]
labels = [[1], [0], [1]]
nn.train(dataset, labels, epochs=10)
```
Currently the optimizer used for training is Stochastic Gradient Descent, since it backpropagates for every input, in future I will add support for other optimizers.
### Prediction
To calculate the output of the network when it is given a certain set of inputs, use the `predict` method. This method takes a single parameter, input, which is a list of floats. The method returns a list of floats representing the output of the network.

For example:
```
input = [1, 89, 66, 23, 94, 28.1, 0.167, 21]
output = nn.predict(input)
```





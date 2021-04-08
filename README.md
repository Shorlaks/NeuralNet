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






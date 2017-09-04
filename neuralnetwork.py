"""
Files for the development of a Neural Network
"""

import math
import random
import json

class Neuron():
    """ Represents a single Neuron/Node """

    def __init__(self, alpha, weights):
        """ Initialise a neuron with INPUT weights and a specific alpha """
        self._weights = weights
        self._alpha = alpha

    def _input(self, inputs):
        """ Returns the sum of the inputs multiplied by the correct weight """
        return sum([w*i for w,i in zip(self._weights, inputs)])

    def _g(self, x):
        """
        Performs the sigmoid function equation from lectures with the defined
        alpha value corresponding to this object
        """
        e_x = math.exp(-self._alpha * x)
        return (1.0 / (1 + e_x))

    def _g_prime(self, x):
        """
        Performs the derivative of the sigmoid function for a given x
        value using the alpha value associated with this neuron
        """
        return self._g(x)*(1 - self._g(x))

    def output(self, inputs):
        """
        Outputs the sigmoid function applied to the weighted sum of the
        inputs, once more using the alpha value associated with this neuron
        """
        self._in_j = self._input(inputs) #Previous weighted inputs
        return self._g(self._in_j)

    def _update_weights(self, alpha, delta):
        """ Internal method for adapting weights """
        res = []
        for j, weight in enumerate(self._weights):
            self._weights[j] = weight + (alpha * delta * self._g_prime(self._in_j))
            #print("Prev weight: {} New weight: {}".format(weight, self._weights[j]))
            res.append(self._weights[j] - weight)
        return res[0]

class Layer():
    """ Represents a single layer (hidden or otherwise) """

    def __init__(self, weights, alphas):
        """ Weights is a 2-D list, alphas is a 1-D list """
        self._neurons = [Neuron(a, w) for w, a in zip(weights, alphas)]

    def calc(self, inputs):
        """ Calculates the result of these inputs on this layer """
        return [neuron.output(inputs) for neuron in self._neurons]

class NeuralNetwork():
    """ Represents the entire Neural Network """

    def __init__(self, weights=[], alphas=[]):
        """
        Weights is a 3-D list. First index for the layer, second for the neuron
        which will have a weight for each input.
        Alphas is a 2-D list. One for each layer, and each neuron has exactly
        1 alpha.
        """
        self._layers = [Layer(w, a) for w, a in zip(weights, alphas)]

    def new_layer(self, nodes, inputs, alpha=0.1):
        """
        The number of nodes in this layer, with the number of inputs to each node
        """
        weights = [[random.uniform(-0.1, 0.1)  for _ in range(inputs)] for i in range(nodes)]
        alphas = [alpha for _ in range(nodes)]
        self._layers.append(Layer(weights, alphas))

    def new_initial_layer(self, nodes, inputs, alpha=0.1):
        """ Creates weightings for the initial layer as 1 """
        weights = [[1 for _ in range(inputs)] for i in range(nodes)]
        alphas = [alpha for _ in range(nodes)]
        self._layers.insert(0, Layer(weights, alphas))

    def new_random_layer(self, nodes, inputs, alpha=0.1):
        """ Creates a random weighting for a layer in the range -1, 1 """
        weights = [[random.uniform(-1, 1) for _ in range(inputs)] for i in range(nodes)]
        alphas = [alpha for _ in range(nodes)]
        self._layers.append(Layer(weights, alphas))

    def run(self, inputs):
        """ Run a neural network with these given inputs """
        for layer in self._layers:
            inputs = layer.calc(inputs)
        return inputs

    def _back_prop_outer(self, layer, alpha, expected, outputs, inputs):
        """ Internal method for back propogation of the outer layer """
        res = []
        for k, outer_neuron in enumerate(layer._neurons):
            err_k = expected[k] - outputs[k]
            err_sig = err_k * outer_neuron._g_prime(outputs[k])
            res.append(outer_neuron._update_weights(alpha, err_sig))
        return res

    def _delta(self, output, err, neuron):
        """ Calculate the necessary delta value """
        return neuron._g_prime(output) * err

    def _back_prop_hidden(self, layer, alpha, next_layer, outputs, inputs, output_delta):
        """
        Internal method for calculating the back propogation for a hidden layer
        """
        res = []
        for i, neuron in enumerate(layer._neurons):
            err_sig = sum([neuron._weights[i] * output_delta[j] for j, neuron in enumerate(next_layer._neurons)])
            res.append(neuron._update_weights(alpha, err_sig))
        return res

    def back_prop(self, data, alpha=2):
        """
        Data is a list of (input, expected)
        Both input and expected should be lists
        """
        reversed_layers = reversed(self._layers[1:]) #Exclude the first layer from back prop
        for inputs, expected in data:
            outputs = self.run(inputs)
            output_delta = None
            next_layer = None
            for i, layer in enumerate(reversed_layers):
                if i == 0:
                    output_delta = self._back_prop_outer(layer, \
                                alpha, expected, outputs, inputs)
                    next_layer = layer
                else:
                    output_delta = self._back_prop_hidden(layer, alpha, next_layer, outputs, inputs, output_delta)
                    next_layer = layer

    def output(self, filename):
        """ Output the neural network as is to JSON """
        with open(filename, 'w') as f:
            op = {}
            layer_res = []
            alphas_res = []
            for layer in self._layers:
                weights = []
                alphas = []
                for neuron in layer._neurons:
                    weights.append(neuron._weights)
                    alphas.append(neuron._alpha)
                layer_res.append(weights)
                alphas_res.append(alphas)
            op['layers'] = layer_res
            op['alphas'] = alphas_res
            json.dump(op, f, indent='\t')

    def adapt_alpha(self, alpha):
        for layers in self._layers:
            for neurons in layers._neurons:
                neurons._alpha = alpha

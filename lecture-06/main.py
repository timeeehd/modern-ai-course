# Copy and pasted from https://github.com/rasmusbergpalm/nanograd/blob/main/nanograd.py

from typing import Union
from math import tanh


class Var:
    """
    A variable which holds a number and enables gradient computations.
    """

    def __init__(self, val: Union[float, int], parents=None):
        assert type(val) in {float, int}
        if parents is None:
            parents = []
        self.v = val
        self.parents = parents
        self.grad = 0.0

    def backprop(self, bp):
        self.grad += bp
        for parent, grad in self.parents:
            parent.backprop(grad * bp)

    def backward(self):
        self.backprop(1.0)

    def __add__(self: 'Var', other: 'Var') -> 'Var':
        return Var(self.v + other.v, [(self, 1.0), (other, 1.0)])

    def __mul__(self: 'Var', other: 'Var') -> 'Var':
        return Var(self.v * other.v, [(self, other.v), (other, self.v)])

    def __pow__(self, power: Union[float, int]) -> 'Var':
        assert type(power) in {float, int}, "power must be float or int"
        return Var(self.v ** power, [(self, power * self.v ** (power - 1))])

    def __neg__(self: 'Var') -> 'Var':
        return Var(-1.0) * self

    def __sub__(self: 'Var', other: 'Var') -> 'Var':
        return self + (-other)

    def __truediv__(self: 'Var', other: 'Var') -> 'Var':
        return self * other ** -1

    def tanh(self) -> 'Var':
        return Var(tanh(self.v), [(self, 1 - tanh(self.v) ** 2)])

    def relu(self) -> 'Var':
        return Var(self.v if self.v > 0.0 else 0.0, [(self, 1.0 if self.v > 0.0 else 0.0)])

    def __repr__(self):
        return "Var(v=%.4f, grad=%.4f)" % (self.v, self.grad)


a = Var(3.0)
b = Var(5.0)
f = a * b

f.backward()

for v in [a, b, f]:
    print(v)

print()

a = Var(3.0)
b = Var(5.0)
c = a * b
d = Var(9.0)
e = a * d
f = c + e

f.backward()

for v in [a, b, c, d, e, f]:
    print(v)

for v in [a, b, c, d, e, f]:
    print(v)


def finite_difference(fn, x_val, dx=1e-10):
    """
    Computes the finite difference numerical approximation to the derivative of fn(x) with respect to x at x_val: (fn(x_val + dx) - fn(x_val))/dx
    """
    return (fn(x_val + dx) - fn(x_val)) / dx


# test function - try to change into other functions as well

def f(a, b):
    return a * b + b


a_grad = finite_difference(lambda x: f(x, 5), 3)

print(a_grad)

b_grad = finite_difference(lambda x: f(3, x), 5)

print(b_grad)

from math import sin
import random
import tqdm as tqdm
import matplotlib.pyplot as plt

def sample_data(noise=0.3):
    x = (random.random() - 0.5) * 10
    return x, sin(x) + x + random.gauss(0, noise)

train_data = [sample_data() for _ in range(100)]
val_data = [sample_data() for _ in range(100)]

for x, y in train_data:
  plt.plot(x, y, 'b.')

plt.show()

from typing import Sequence


class Initializer:

    def init_weights(self, n_in, n_out) -> Sequence[Sequence[Var]]:
        raise NotImplementedError

    def init_bias(self, n_out) -> Sequence[Var]:
        raise NotImplementedError


class NormalInitializer(Initializer):

    def __init__(self, mean=0, std=0.1):
        self.mean = mean
        self.std = std

    def init_weights(self, n_in, n_out):
        return [[Var(random.gauss(self.mean, self.std)) for _ in range(n_out)] for _ in range(n_in)]

    def init_bias(self, n_out):
        return [Var(0.0) for _ in range(n_out)]


class DenseLayer:
    def __init__(self, n_in: int, n_out: int, act_fn, initializer: Initializer = NormalInitializer()):
        """
          n_in: the number of inputs to the layer
          n_out: the number of output neurons in the layer
          act_fn: the non-linear activation function for each neuron
          initializer: The initializer to use to initialize the weights and biases
        """
        self.weights = initializer.init_weights(n_in, n_out)
        self.bias = initializer.init_bias(n_out)
        self.act_fn = act_fn

    def __repr__(self):
        return 'Weights: ' + repr(self.weights) + ' Biases: ' + repr(self.bias)

    def parameters(self) -> Sequence[Var]:
        """Returns all the vars of the layer (weights + biases) as a single flat list"""
        parameter = []
        for i in range(len(self.weights)):
            parameter += self.weights[i]
        parameter += self.bias
        return parameter

    def forward(self, inputs: Sequence[Var]) -> Sequence[Var]:
        """
        inputs: A n_in length vector of Var's corresponding to the previous layer outputs or the data if it's the first layer.

        Computes the forward pass of the dense layer: For each output neuron, j, it computes: act_fn(weights[i][j]*inputs[i] + bias[j])
        Returns a vector of Vars that is n_out long.
        """
        assert len(self.weights) == len(inputs), "weights and inputs must match in first dimension"
        output = []
        for j in range(len(self.bias)):
            value = Var(0)
            for i in range(len(inputs)):
                value += self.weights[i][j] * inputs[i]
            output.append(self.act_fn(value + self.bias[j]))
        return output


import numpy as np

np.random.seed(0)

w = np.random.randn(3, 2)
b = np.random.randn(2)
x = np.random.randn(3)

expected = np.tanh(x @ w + b)


class FixedInit(Initializer):
    """
    An initializer used for debugging that will return the w and b variables defined above regardless of the input and output size.
    """

    def init_weights(self, n_in, n_out):
        return [list(map(Var, r.tolist())) for r in w]

    def init_bias(self, n_out):
        return list(map(Var, b.tolist()))


layer = DenseLayer(3, 2, lambda x: x.tanh(), FixedInit())
print('test')
print(layer.parameters())
var_x = list(map(Var, x.tolist()))
actual = layer.forward(var_x)
print(actual)
print(expected)


class MLP:
    def __init__(self, layers: Sequence[DenseLayer]):
        self.layers = layers

    def parameters(self) -> Sequence[Var]:
        """ Returns all the parameters of the layers as a flat list"""
        parameter = []
        for layer in self.layers:
            parameter += layer.parameters()
        return parameter

    def forward(self, x: Sequence[Var]) -> Sequence[Var]:
        """
        Computes the forward pass of the MLP: x = layer(x) for each layer in layers
        """
        for layer in self.layers:
            x = layer.forward(x)
        return x


class SGD:
    def __init__(self, parameters: Sequence[Var], learning_rate: float):
        self.parameters = parameters
        self.learning_rate = learning_rate

    def zero_grad(self):
        """ Set the gradient to zero for all parameters """
        for parameter in self.parameters:
            parameter.grad = 0

    def step(self):
        """Performs a single step of SGD for each parameter: p = p - learning_rate * grad_p """
        for parameter in self.parameters:
            parameter.v -= self.learning_rate * parameter.grad


def squared_loss(t: Var, y: Var) -> Var:
    return (t - y) ** 2


mlp = MLP([
    DenseLayer(1, 5, lambda x: x.tanh()),
    DenseLayer(5, 1, lambda x: x)
])
print('test')
print(mlp.parameters())
x, t = sample_data()
x = Var(x)
t = Var(t)
y = mlp.forward([x])

loss = squared_loss(t, y[0])
loss.backward()

for i, layer in enumerate(mlp.layers):
    print("layer", i, layer)

mlp = MLP([
    DenseLayer(1, 16, lambda x: x.tanh()),
    DenseLayer(16, 1, lambda x: x)
])  # What does this line do?
# Create the network

learning_rate = 0.01  # Try different learning rates
optim = SGD(mlp.parameters(),
            learning_rate)  # What does this line do? --> Initilize the optimizer with the respective parameters

batch_size = 64
losses = []
for i in tqdm.tqdm(range(100)):
    loss = Var(0.0)
    for _ in range(batch_size):  # What does this loop do? --> Uses batches to train the network
        x, y_target = random.choice(train_data)  # What does this line do? --> Choose random training data
        x = Var(x)
        y_target = Var(y_target)
        y = mlp.forward([x])
        loss += squared_loss(y_target, y[0])

    loss = loss / Var(batch_size)  # What does this line do? --> Calculate the average loss
    losses.append(loss.v)
    optim.zero_grad()  # Why do we need to call zero_grad here? --> To start from scratch again, and dont use the calculate gradients again
    loss.backward()  # What does this line do? --> Calculate the backward propagation
    optim.step()  # What does this line do? --> Update the weights by using the optimizer

plt.plot(losses, '.')
plt.ylabel('L2 loss')
plt.xlabel('Batches')
plt.show()

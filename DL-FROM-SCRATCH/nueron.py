import numpy as np

# 1. Activation function


def sigmoid(x):
    """
    this sigmoid function
    squashes values between 0 and 1.
    (σ(x) = 1 / (1 + e^(-x)))
    """
    return 1/(1+np.exp(-x))


def relu(x):
    """
    The Rectified Linear Unit (ReLU) activation function.
    Outputs x if x > 0, else outputs 0.
    """
    return np.maximum(0, x)


class Perceptron: 
    def __init__(self, num_inputs, activation_fn=sigmoid):
        """
        Initialize the single perceptron/nueron.

        ARGS: 
            num_inputs (int): the number of input features this nueron expects.
            activation_fn: the activation function to use.
        """
        # weights: one weight for each input
        # Initialize with small random values to break symmetry and avoid dead nuerons (for relu)
        self.weights = np.random.randn(num_inputs) * 0.01

        # bias : a single value that shifts the activation function output.
        # initialize with small random value
        self.bias = np.random.randn() * 0.01

        self.activation_fn = activation_fn

        print(f"Perceptron initialized with {num_inputs} inputs.")
        print(f"Initial weights : {self.weights}")
        print(f"Initial bias : {self.bias}")

    
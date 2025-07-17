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
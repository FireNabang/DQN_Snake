import numpy as np


def sigmoid(z):
    z = np.array(z, dtype=np.float128)
    z = np.clip(z, -709.78, 709.78)
    return np.reciprocal(np.add(1.0, np.exp(-z)))


def sigmoid_diff(z):
    return np.multiply(sigmoid(z), np.subtract(1, sigmoid(z)))


def none(z):
    z = np.array(z, dtype=np.float128)
    return z


def ReLU(z):
    mz = [max(x, 0) for x in z]
    result = np.array(mz, dtype=np.float128)
    return result.reshape(z.shape)


def ReLU_diff(z):
    result = np.array([int(x > 0) for x in z], dtype=np.float128)
    return result.reshape(z.shape)


def tanh(z):
    return 2*sigmoid(2*np.array(z)) - 1


def tanh_diff(z):
    return 1 - tanh(z)*tanh(z)


def soft_max(z):
    z = np.array(z, dtype=np.float128)
    z = np.clip(z, -709.78, 709.78)
    e = np.exp(z)
    return e / np.sum(e, dtype=np.float128)


def soft_max_diff(z):
    s = soft_max(z)
    result = np.ones((len(s), len(s)), dtype=np.float128)
    for i in range(len(s)):
        for j in range(len(s)):
            result[i][j] = s[i] * (int(i == j) - s[j])
    return result


activations = {
    'none': none,
    'sigmoid': sigmoid,
    'tanh': tanh,
    'relu': ReLU,
    'softmax': soft_max
}

activations_diff = {
    'none': none,
    'sigmoid': sigmoid_diff,
    'tanh': tanh_diff,
    'relu': ReLU_diff,
    'softmax': soft_max_diff
}


class ActivationFunction:
    def __init__(self, func: str):
        self.activation_function = activations[func]
        self.activation_function_diff = activations_diff[func]

    def get_activate(self, data):
        return self.activation_function(data)

    def get_activate_diff(self, data):
        return self.activation_function_diff(data)

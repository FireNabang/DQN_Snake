import numpy as np


def sigmoid(z):
    z = np.array(z, dtype=np.float32)
    z = np.clip(z, -88.72, 88.72)
    return np.reciprocal(np.add(1.0, np.exp(-z)))


def sigmoid_diff(z):
    z = np.array(z, dtype=np.float32)
    return np.multiply(sigmoid(z), np.subtract(1, sigmoid(z)))


def none(z):
    z = np.array(z, dtype=np.float32)
    return z


def ReLU(z):
    z = np.array(z, dtype=np.float32)
    try:
        result = [max(x, 0) for x in z]
    except:
        result = np.zeros(z.shape)
        for ch in range(z.shape[0]):
            for n in range(z.shape[1]):
                for m in range(z.shape[2]):
                    result[ch][n][m] = max(z[ch][n][m],0)
    return result


def ReLU_diff(z):
    z = np.array(z, dtype=np.float32)
    try:
        result = np.array([int(x > 0) for x in z], dtype=np.float32)
    except:
        result = np.zeros(z.shape)
        for ch in range(z.shape[0]):
            for n in range(z.shape[1]):
                for m in range(z.shape[2]):
                    result[ch][n][m] = int(z[ch][n][m]>0)
    return result


def tanh(z):
    return 2*sigmoid(2*np.array(z)) - 1


def tanh_diff(z):
    return 1 - tanh(z)*tanh(z)


def soft_max(z):
    z = np.array(z, dtype=np.float32)
    z = np.clip(z, -88.72, 88.72)
    e = np.exp(z)
    return e / np.sum(e, dtype=np.float32)


def soft_max_diff(z):
    s = soft_max(z)
    result = np.ones((len(s), len(s)), dtype=np.float32)
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

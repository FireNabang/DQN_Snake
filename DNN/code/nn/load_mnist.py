import six.moves.cPickle as pickle
import gzip
import numpy as np



def encode_label(j):
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e

def vector_shape_data(data):
    features = [np.reshape(x, (784, 1)) for x in data[0]]
    labels = [encode_label(y) for y in data[1]]
    return list(zip(features, labels))

def matrix_shape_data(data):
    features = [np.reshape(x, (28, 28)) for x in data[0]]
    labels = [encode_label(y) for y in data[1]]
    return list(zip(features, labels))

def load_data(shape = 'vector'):
    with gzip.open('mnist.pkl.gz', 'rb') as f:
        data = pickle._Unpickler(f)
        data.encoding = 'latin1'
        train_data, validation_data, test_data = data.load()
    if(shape == 'vector'):
        return vector_shape_data(train_data), vector_shape_data(test_data)
    return matrix_shape_data(train_data), matrix_shape_data(test_data)
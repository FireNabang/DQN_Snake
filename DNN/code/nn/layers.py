import numpy as np
from nn.functions import activations


def conv2D(src, kernel, stride=1):
    # kernel = np.flipud(np.fliplr(kernel))

    kernel_height, kernel_width = kernel.shape
    src_height, src_width = src.shape
    output_width = ((src_width - kernel_width) // stride) + 1
    output_height = ((src_height - kernel_height) // stride) + 1
    output = np.zeros((output_height, output_width))

    for n in range(src_height - kernel_height + 1):
        if n % stride == 0:
            for m in range(src_width - kernel_width + 1):
                try:
                    if m % stride == 0:
                        output[n, m] = (kernel * src[n: n + kernel_height, m: m + kernel_width]).sum()
                except:
                    break

    return output.transpose()

class Layer(object):
    def __init__(self, _activations='none'):
        self.params = []
        self.input_shape = None

        self.previous_layer = None
        self.next_layer = None

        self.input = None
        self.output = None
        self.accumulated_delta = None
        self.loss_delta = None

        self.activation_function = activations.ActivationFunction(_activations)

    def get_input(self):
        raise NotImplementedError

    def feed_forward(self):
        raise NotImplementedError

    def get_activate(self, data):
        return self.activation_function.get_activate(data)

    def get_activate_diff(self, data):
        return self.activation_function.get_activate_diff(data)

    def backpropagation(self):
        raise NotImplementedError

    def get_accumulated_delta(self):
        if self.next_layer is not None:
            return self.next_layer.accumulated_delta
        else:
            return self.loss_delta

    def clear_deltas(self):
        pass

    def update(self, learning_rate):
        pass

    def connect(self, layer):
        self.previous_layer = layer
        layer.next_layer = self


class DenseLayer(Layer):
    def __init__(self, dim=None, activations='none'):

        super(DenseLayer, self).__init__(activations)
        self.dim = dim

        self.weight = None
        self.bias = None
        self.params = [self.weight, self.bias]

        self.delta_w = None
        self.delta_b = None

    def get_input(self):
        if self.previous_layer is not None:
            return np.dot(self.weight, self.previous_layer.output) + self.bias
        else:
            return np.array(self.input, dtype=np.float128)

    def feed_forward(self):
        self.input = self.get_input()
        self.output = self.activation_function.get_activate(self.input)

    def backpropagation(self):
        data = self.get_input()
        accumulated_delta = self.get_accumulated_delta()
        activate_diff = self.get_activate_diff(data)

        if self.next_layer is None and (accumulated_delta.shape != activate_diff.shape):
            delta = np.dot(activate_diff.transpose(), accumulated_delta)
        else:
            delta = accumulated_delta * activate_diff

        self.delta_b += delta
        self.delta_w += np.dot(delta, self.previous_layer.output.transpose())
        self.accumulated_delta = np.dot(self.weight.transpose(), delta)

    def update(self, learning_rate):
        self.weight -= learning_rate * self.delta_w
        self.bias -= learning_rate * self.delta_b

    def clear_deltas(self):
        self.delta_w = np.zeros(self.weight.shape)
        self.delta_b = np.zeros(self.bias.shape)

    def connect(self, layer):
        super(DenseLayer, self).connect(layer)
        self.weight = np.random.randn(self.dim, layer.dim)
        self.bias = np.random.randn(self.dim, 1)

        self.params = [self.weight, self.bias]

        self.delta_w = np.zeros(self.weight.shape)
        self.delta_b = np.zeros(self.bias.shape)


class Conv2DLayer (Layer):
    def __init__(self, filter_count=1, filter_size=(0, 0), pad=0, activations='none', stride=1, input_size=(1, 1, 1)):
        super(Conv2DLayer, self).__init__(activations)
        self.stride = stride

        self.filter_height = filter_size[0]
        self.filter_width = filter_size[1]
        self.filter_count = filter_count

        self.pad = pad
        self.input_height = input_size[0]
        self.input_width = input_size[1]
        self.output_height = self.input_height
        self.output_width = self.input_width

        self.channel = input_size[2] * filter_count

        self.filter = np.array([np.random.randn(self.filter_height, self.filter_width) for _ in range(self.filter_count)], dtype=np.float128)
        self.bias = np.array([np.random.randn(1, 1) for _ in range(self.filter_count)], dtype=np.float128)
        self.params = [self.filter, self.bias]

        self.delta_filter = np.zeros(self.filter.shape, dtype=np.float128)

        self.input = None
        self.output = None

    def get_input(self):
        if self.previous_layer is not None:
            img = [np.pad(output, ((self.pad, self.pad), (self.pad, self.pad)), 'constant',constant_values = (0)) for output in self.previous_layer.output]
            result = []
            for src in img:
                for kernel in self.filter:
                    result.append(conv2D(src, kernel, self.stride))
            return result
        else:
            return np.array(self.input, dtype=np.float128)

    def feed_forward(self):
        self.input = self.get_input()
        self.output = self.activation_function.get_activate(self.input)
        self.output_height, self.output_width = self.output[0].shape

    def backpropagation(self):
        accumulated_delta = self.get_accumulated_delta()
        _input = self.previous_layer.output

        for out_ch in range(0, self.channel, self.filter_count):
            for fc in range(self.filter_count):
                self.delta_filter += conv2D(_input[out_ch // self.filter_count], accumulated_delta[out_ch + fc])

        delta_input = np.zeros(self.previous_layer.output.shape, dtype=np.float128)
        for fc in range(self.filter_count):
            for pre_ch in range(self.channel // self.filter_count):
                pad_h = self.filter_height - 1
                pad_w = self.filter_width - 1
                img = np.pad(accumulated_delta[pre_ch + fc], ((pad_h, pad_h), (pad_w, pad_w)), 'constant',constant_values = (0))
                temp = conv2D(img, np.rot90(self.filter[fc], 2))
                delta_input[pre_ch] += temp
        self.accumulated_delta = delta_input

    def update(self, learning_rate):
        self.filter -= learning_rate * self.delta_filter

    def clear_deltas(self):
        self.delta_filter = np.zeros(self.filter.shape, dtype=np.float128)

    def connect(self, layer):
        super(Conv2DLayer, self).connect(layer)
        self.input_height = (self.previous_layer.output_height + 2 * self.pad - self.filter_height ) // self.stride + 1
        self.input_width = (self.previous_layer.output_width + 2 * self.pad - self.filter_width) // self.stride + 1
        self.output_height = self.input_height
        self.output_width = self.input_width
        self.channel = layer.channel * self.filter_count


# Conv -> Fully Connected Layer

class MaxPooling2DLayer(Layer):
    def __init__(self, pool_size=(2, 2), activations='none'):

        super(MaxPooling2DLayer, self).__init__(activations)
        self.pool_size = pool_size
        self.pool_height = pool_size[0]
        self.pool_width = pool_size[1]

        self.input_height = None
        self.input_width = None
        self.output_height = None
        self.output_width = None

        self.channel = None

    def get_input(self):
        if self.previous_layer is not None:
            temp = self.previous_layer.output
            result = np.zeros((self.channel, self.input_height, self.input_width),dtype = np.float128)
            for pre_output_ch in range(self.previous_layer.channel):
                for m in range(self.input_width):
                    for n in range(self.input_height):
                        result[pre_output_ch][n][m] += max([temp[pre_output_ch][n + i][m + j] for i in range(self.pool_height) for j in range(self.pool_width)])
            return result
        else:
            return self.input

    def feed_forward(self):
        self.input = self.get_input()
        self.output = self.input
        self.output_height, self.output_width = self.output[0].shape

    def backpropagation(self):
        accumulated_delta = self.get_accumulated_delta()
        delta = np.zeros(self.previous_layer.output.shape,dtype = np.float128)
        temp = self.previous_layer.output

        for ch in range(self.channel):
            for n in range(0, self.input_height - self.pool_height):
                for m in range(0, self.input_width - self.pool_width):
                    M = max([temp[ch][n + i][m + j] for i in range(self.pool_height) for j in range(self.pool_width)])
                    for i in range(self.pool_height):
                        for j in range(self.pool_width):
                            if self.input[ch][n + i][m + j] == M:
                                delta[ch][n + i][m + j] += accumulated_delta[ch][n][m]

        self.accumulated_delta = delta

    def update(self, learning_rate):
        pass

    def clear_deltas(self):
        pass

    def connect(self, layer):
        super(MaxPooling2DLayer, self).connect(layer)
        self.input_height = (self.previous_layer.output_height - self.pool_height) + 1
        self.input_width = (self.previous_layer.output_width - self.pool_width) + 1
        self.output_height = self.input_height
        self.output_width = self.input_width
        self.channel = self.previous_layer.channel


class AvgPooling2DLayer(Layer):
    def __init__(self, pool_size=(2, 2), activations='none'):

        super(AvgPooling2DLayer, self).__init__(activations)
        self.pool_size = pool_size
        self.pool_height = pool_size[0]
        self.pool_width = pool_size[1]

        self.input_height = None
        self.input_width = None
        self.output_height = None
        self.output_width = None

        self.channel = None

    def get_input(self):
        if self.previous_layer is not None:
            temp = self.previous_layer.output
            result = np.zeros((self.channel, self.input_height, self.input_width), dtype = np.float128)
            for pre_output_ch in range(self.previous_layer.channel):
                for m in range(self.input_width):
                    for n in range(self.input_height):
                        result[pre_output_ch][n][m] += sum([temp[pre_output_ch][n + i][m + j] for i in range(self.pool_height) for j in range(self.pool_width)]) / (self.pool_height * self.pool_width)
            return result
        else:
            return None

    def feed_forward(self):
        self.input = self.get_input()
        self.output = self.input

    def backpropagation(self):
        accumulated_delta = self.get_accumulated_delta()
        delta = np.zeros(self.previous_layer.output.shape,dtype = np.float128)

        temp = self.previous_layer.output
        for ch in range(self.channel):
            for n in range(self.input_height):
                for m in range(self.input_width):
                    d = accumulated_delta[ch][n][m] * sum([temp[ch][n + i][m + j] for i in range(self.pool_height) for j in range(self.pool_width)]) / (self.pool_height * self.pool_width)
                    delta[ch][n][m] += d
                    delta[ch][n+1][m] += d
                    delta[ch][n][m+1] += d
                    delta[ch][n+1][m+1] += d

        self.accumulated_delta = delta

    def update(self, learning_rate):
        pass

    def clear_deltas(self):
        pass

    def connect(self, layer):
        super(AvgPooling2DLayer, self).connect(layer)
        self.input_height = (self.previous_layer.output_height - self.pool_height) + 1
        self.input_width = (self.previous_layer.output_width - self.pool_width) + 1
        self.output_height = self.input_height
        self.output_width = self.input_width
        self.channel = self.previous_layer.channel


class FlattenLayer(Layer):
    def __init__(self, dim=None, activations='none'):
        super(FlattenLayer, self).__init__(activations)
        self.dim = dim


    def get_input(self):
        if self.previous_layer is not None:
            return self.previous_layer.output
        else:
            return None

    def feed_forward(self):
        self.input = self.get_input()
        self.output = self.input.reshape(self.previous_layer.output_width * self.previous_layer.output_height * self.previous_layer.channel, 1)

    def backpropagation(self):
        accumulated_delta = self.get_accumulated_delta()
        self.accumulated_delta = accumulated_delta.reshape(self.input.shape)

    def update(self, learning_rate):
        pass

    def clear_deltas(self):
        pass

    def connect(self, layer):
        super(FlattenLayer, self).connect(layer)
        self.dim = self.previous_layer.output_width * self.previous_layer.output_height * self.previous_layer.channel

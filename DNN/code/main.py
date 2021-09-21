from nn import load_mnist
from nn import network
#from nn import Optimizer
from nn.layers import DenseLayer


def main():
    training_data, test_data = load_mnist.load_data()

    model = network.SequentialNetwork()

    model.add_layer(DenseLayer(784,'none'))
    model.add_layer(DenseLayer(392,'sigmoid'))
    model.add_layer(DenseLayer(196,'sigmoid'))
    model.add_layer(DenseLayer(10,'sigmoid'))
    #optimizer = Optimizer.Momentum();

    model.train(training_data, epochs=3, mini_batch_size=10, learning_rate=3.0, test_data=test_data)


if __name__ == '__main__':
    main()
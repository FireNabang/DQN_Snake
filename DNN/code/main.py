from nn import load_mnist
from nn import network
from nn.layers import DenseLayer, Conv2DLayer, MaxPooling2DLayer, FlattenLayer,DropoutLayer


def main():
    training_data, test_data = load_mnist.load_data(train_data_cnt=500, test_data_cnt=1000, shape='matrix')
    model = network.SequentialNetwork()
    model.add_layer(Conv2DLayer(input_size=(28, 28, 1)))
    model.add_layer(Conv2DLayer(filter_count=10, filter_size=(2, 2)))
    model.add_layer(MaxPooling2DLayer(pool_size=(2, 2)))
    model.add_layer(DropoutLayer(0.5))
    model.add_layer(FlattenLayer())
    model.add_layer(DenseLayer(32, 'sigmoid'))
    model.add_layer(DropoutLayer(0.5))
    model.add_layer(DenseLayer(10, 'sigmoid'))

    # model.load_model()

    model.summary()
    model.train(training_data, epochs=3, mini_batch_size=10, learning_rate=0.9, test_data=test_data)

    model.save_model()


if __name__ == '__main__':
    main()
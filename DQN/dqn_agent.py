import queue
import random
import numpy as np
from DNN.code.nn import network
from DNN.code.nn.layers import DenseLayer, Conv2DLayer, MaxPooling2DLayer, FlattenLayer,DropoutLayer


class DQNAgent:
    def __init__(self, field_size, batch_size, learning_rate, discount_factor):
        self.field_height, self.field_width = field_size
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor

        self.model = self.create_model()
        self.model.summary()

        self.game_data = queue(maxsize=65536)
        self.target_update_counter = 0

    def create_model(self):
        model = network.SequentialNetwork('categorical_crossentropy')
        model.add_layer(Conv2DLayer(input_size=(self.field_height,self.field_width,1)))
        model.add_layer(Conv2DLayer(filter_count=8,filter_size=(3,3),activations='relu'))
        model.add_layer(DropoutLayer(0.1))
        model.add_layer(Conv2DLayer(filter_count=8, filter_size=(3, 3), activations='relu'))
        model.add_layer(FlattenLayer())
        model.add_layer(DenseLayer(32, activations='relu'))
        model.add_layer(DropoutLayer(0.1))
        model.add_layer(DenseLayer(4, activations='soft_max'))
        return model

    def update_game_data(self, current_state, action, reward, next_state, live):
        self.game_data.append((current_state, action, reward, next_state, live))

    def get_q_values(self, x):
        return self.model.predict(x)

    def train(self):
        if len(self.game_data) < 2048:
            return

        samples = random.sample(self.game_data, 2048)
        current_input = np.stack([sample[0] for sample in samples])
        current_q_values = self.model.predict(current_input)
        next_input = np.stack([sample[3] for sample in samples])
        next_q_values = self.model.predict(next_input)

        # update q values
        for i, (current_state, action, reward, _, done) in enumerate(samples):
            if done:
                next_q_value = reward
            else:
                next_q_value = (1-self.learning_rate)*current_q_values[i, action] + self.learning_rate*(reward + self.discount_factor * np.max(next_q_values[i]))
            current_q_values[i, action] = next_q_value

        #  model train
        self.model.train(zip(current_input, current_q_values), epochs=3, mini_batch_size=self.batch_size, learning_rate=0.9)


    def save(self, model_filepath):
        self.model.save_model(model_filepath)

    def load(self, model_filepath):
        self.model.load_model(model_filepath)
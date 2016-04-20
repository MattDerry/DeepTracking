from keras.models import Model
from keras.layers import Input, Convolution2D, merge
from keras.optimizers import Adagrad
import keras.backend as K


class FeedForwardRNN:
    def __init__(self, h_input_shape, x_input_shape, sequence_length):
        self.h_input_dims = h_input_shape
        self.x_input_dims = x_input_shape
        self.sequence_length = sequence_length
        self.build_model()
        self.h_prev = K.zeros(shape=h_input_shape, name='prev_h')
        self.model = self.build_model()
        return

    def __call__(self, state):
        result = self.model({'h0': self.h_prev, 'x1': state})
        self.h_prev = result['h1']
        return result['y1']

    def build_model(self):
        h0 = Input(shape=self.h_input_dims)
        x1 = Input(shape=self.x_input_dims)
        e = Convolution2D(16, 7, 7, activation='sigmoid', dim_ordering='tf')(x1)
        j = merge([e, h0], mode='concat', concat_axis=1)
        h1 = Convolution2D(32, 7, 7, activation='sigmoid', dim_ordering='tf')(j)
        y1 = Convolution2D(1, 7, 7, activation='sigmoid', dim_ordering='tf')(h1)
        model = Model(input=[h0, x1], output=[h1, y1])
        opt = Adagrad(lr=0.01, epsilon=1e-6)
        model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def reset_state(self):
        self.h_prev = K.zeros(shape=self.h_input_dims, name='prev_h')
        return

    # TODO (Matt): Add check for state measurements and incorporate them when available
    # Possible use cases:
    #  1) one initial state measurement and arbitrary prediction...DONE
    #  2) sparse state measurements, blank states otherwise
    #  3) 1-to-1 correspondance between number of state measurements and number of time_steps
    def predict_future(self, time_steps, initial_state):
        predictions = []
        h_temp = self.h_prev
        result = self.model({'h0': h_temp, 'x1': initial_state})
        h_temp = result['h1']
        predictions.append(result['y1'])
        masked_input = K.zeros(shape=self.x_input_dims)
        for i in range(1, time_steps):
            result = self.model({'h0': h_temp, 'x1': masked_input})
            h_temp = result['h1']
            predictions.append(result['y1'])
        return predictions

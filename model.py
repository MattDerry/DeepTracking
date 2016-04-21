import json

from keras.models import Model, model_from_json
from keras.layers import Input, Convolution2D, merge
from keras.optimizers import Adagrad
import keras.backend as K


class FeedForwardRNN:
    def __init__(self, h_input_shape, x_input_shape):
        self.h_input_dims = h_input_shape
        self.x_input_dims = x_input_shape
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
    #  3) 1-to-1 correspondence between number of state measurements and number of time_steps
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

    # The file_name parameter should not include a file extension
    # since it uses the same name for the json model file and the h5py weights file
    def save_model_and_weights(self, save_folder_path, file_name):
        json_model_string = self.model.to_json()
        config_filename = '%s%s.json' % (save_folder_path, file_name)
        with open(config_filename, 'w') as model_out:
            json.dump(json_model_string, model_out)
        weights_filename = '%s%s.h5' % (save_folder_path, file_name)
        self.model.save_weights(weights_filename)
        return

    # The file_name parameter should not include a file extension
    # since it uses the same name for the json model file and the h5py weights file
    def load_model_and_weights(self, save_folder_path, file_name):
        config_filename = '%s%s.json' % (save_folder_path, file_name)
        with open(config_filename, 'r') as model_in:
            model_json = json.load(model_in)
        self.model = model_from_json(model_json)
        weights_filename = '%s%s.h5' % (save_folder_path, file_name)
        self.model.load_weights(weights_filename)
        return

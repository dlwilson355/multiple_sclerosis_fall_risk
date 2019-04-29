import numpy as np
import tensorflow as tf
import keras

class RNNImp(object):
    def __init__(self, hiddenunits):
        self.hiddenunits = hiddenunits

    def SimpleRNN(self, input_shape, classes):
        model= keras.models.Sequential()
        model.add(keras.layers.SimpleRNN(units=self.hiddenunits,input_shape=input_shape, activation='tanh', return_sequences = False))
        ol = keras.layers.Dense(classes, activation = 'sigmoid')
        model.add(ol)
        return model

    def GRU(self, input_shape, classes):
        model= keras.models.Sequential()  
        model.add(keras.layers.GRU(units=self.hiddenunits,input_shape=input_shape, activation='tanh'))
        ol = keras.layers.Dense(classes, activation = 'sigmoid')
        model.add(ol)
        return model

    def LSTM(self, input_shape, classes):
        model= keras.models.Sequential()
        model.add(keras.layers.LSTM(units=self.hiddenunits,input_shape=input_shape, activation='tanh', return_sequences = False))
        ol = keras.layers.Dense(classes, activation = 'sigmoid')
        model.add(ol)
        return model


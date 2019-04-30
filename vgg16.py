import numpy as np
import tensorflow as tf
import keras

class VGG16Imp(object):
    def __init__(self):
        return

    def ConvBlock(self, units, dropout=0.2, activation='relu', input=None):
        if input==None:
            self.model.add(keras.layers.Conv2D(units, (3, 3), padding='same'))
        else:
            self.model.add(keras.layers.Conv2D(units, (3, 3),input_shape=input, padding='same'))
        self.model.add(keras.layers.BatchNormalization())
        self.model.add(keras.layers.Activation(activation))
        self.model.add(keras.layers.Dropout(dropout))

    def DenseBlock(self,units, dropout=0.2, activation='relu'):
        self.model.add(keras.layers.Dense(units))
        self.model.add(keras.layers.BatchNormalization())
        self.model.add(keras.layers.Activation(activation))
        self.model.add(keras.layers.Dropout(dropout))
        

    def VGG16WithBN(self, input_shape=None, classes=1000, conv_dropout=0.1, dropout=0.3, activation='relu'):
        self.model = keras.models.Sequential()

        # Block 1
        self.ConvBlock(64, dropout=conv_dropout, activation=activation, input=input_shape)
        self.ConvBlock(64, dropout=conv_dropout, activation=activation)
        self.model.add(keras.layers.MaxPooling2D((2, 2), strides=(2, 2)))

        # Block 2
        self.ConvBlock(128, dropout=conv_dropout, activation=activation)
        self.ConvBlock(128, dropout=conv_dropout, activation=activation)
        self.model.add(keras.layers.MaxPooling2D((2, 2), strides=(2, 2)))

        # Block 3
        self.ConvBlock(256, dropout=conv_dropout, activation=activation)
        self.ConvBlock(256, dropout=conv_dropout, activation=activation)
        self.model.add(keras.layers.MaxPooling2D((2, 2), strides=(2, 2)))

        # Block 4
        self.ConvBlock(512, dropout=conv_dropout, activation=activation)
        self.ConvBlock(512, dropout=conv_dropout, activation=activation)
        self.ConvBlock(512, dropout=conv_dropout, activation=activation)
        self.model.add(keras.layers.MaxPooling2D((2, 2), strides=(2, 2)))

        # Block 5
        self.ConvBlock(512, dropout=conv_dropout, activation=activation)
        self.ConvBlock(512, dropout=conv_dropout, activation=activation)
        self.ConvBlock(512, dropout=conv_dropout, activation=activation)
        self.model.add(keras.layers.MaxPooling2D((2, 2), strides=(2, 2)))

        # Flatten
        self.model.add(keras.layers.GlobalAveragePooling2D())

        # FC Layers
        self.DenseBlock(4096, dropout=dropout, activation=activation)
        self.DenseBlock(4096, dropout=dropout, activation=activation)
    
        # Output layer
        self.model.add(keras.layers.Dense(classes, activation='softmax', name='predictions'))
        return self.model


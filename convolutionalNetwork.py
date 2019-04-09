import keras
from data_reader import DataReader
import numpy as np
from pathlib import Path
import tensorflow as tf
from keras.applications.resnet50 import ResNet50

class ConvNetwork():
    def __init__(self, filepath, segment_size, segments_per_patient, epochs):
        self.master_filepath = filepath # replace this with your filepath to the "MS Fall Study" directory
        self.segment_size = segment_size # this variable represents the number of sequential data measurements that are part of each "segment" of x data
        self.segments_per_patient = segments_per_patient # this variable represents the number of segments to get from each patient
        self.batch_size = 20
        self.epochs = epochs
        self.validation_split=0.25
        return

    def create_model(self, x_data, y_data):
        model = keras.Sequential()
        model.add(keras.layers.Conv2D(100, input_shape=(x_data.shape[1], x_data.shape[2], 1), kernel_size=(10, 10)))
        model.add(keras.layers.Activation('relu'))
        model.add(keras.layers.Conv2D(100, kernel_size=(3, 3)))
        model.add(keras.layers.Activation('relu'))
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(y_data.shape[1]))
        model.add(keras.layers.Activation('softmax'))
        opt = keras.optimizers.SGD(lr=0.001)
        model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
        print(model.summary())
        return (model)

    def train_model(self, model, x_data, y_data):
        model.fit(x=x_data, y=y_data, batch_size=self.batch_size, epochs=self.epochs, validation_split=self.validation_split)
        model.save_weights("cnn_weights.h5")

    def load_data(self):
        reader = DataReader(self.master_filepath)
        data = reader.get_segmented_data(self.segment_size, self.segments_per_patient)
        return (data)

    def run(self):
        (x_data, y_data) = self.load_data()
        x_data = x_data.reshape((x_data.shape[0], x_data.shape[1], x_data.shape[2], 1))
        model = self.create_model(x_data, y_data)
        self.train_model(model, x_data, y_data)

class MLPNetwork():
    def __init__(self, filepath, segment_size, segments_per_patient, epochs):
        self.master_filepath = filepath # replace this with your filepath to the "MS Fall Study" directory
        return

    def create_model(self, data):
        model = keras.Sequential()
        # this is a simple MLP I made
        # we will use something more sophisticated (and better suited) like an RNN for the final project
        model.add(keras.layers.Dense(2000, input_shape=(data[0].shape[1], data[0].shape[2])))
        model.add(keras.layers.Activation('relu'))
        model.add(keras.layers.Dropout(0.15))
        model.add(keras.layers.Dense(1000))
        model.add(keras.layers.Activation('relu'))
        model.add(keras.layers.Dropout(0.15))
        model.add(keras.layers.Dense(500))
        model.add(keras.layers.Activation('relu'))
        model.add(keras.layers.Dropout(0.15))
        model.add(keras.layers.Dense(200))
        model.add(keras.layers.Activation('relu'))
        model.add(keras.layers.Dropout(0.15))
        model.add(keras.layers.Dense(100))
        model.add(keras.layers.Activation('relu'))
        model.add(keras.layers.Dropout(0.15))
        model.add(keras.layers.Dense(50))
        model.add(keras.layers.Activation('relu'))
        model.add(keras.layers.Dropout(0.15))
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(data[1].shape[1]))
        model.add(keras.layers.Activation('softmax'))
        opt = keras.optimizers.SGD(lr=0.000000000001)
        model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
        print(model.summary())
        return (model)

    def train_model(self, model, data):
        print(model.summary())
        model.fit(x=data[0], y=data[1], batch_size=16, epochs=20, validation_split=0.5)
        model.save_weights("weights.h5")

    def load_data(self, folder, onehot=True, samplesfromz=False):
        reader = DataReader(folder)
        sample_data = reader.get_data(onehot,samplesfromz)
        return (sample_data)

    def run(self):
        data = self.load_data(master_filepath)
        print(data[0].shape)
        print(data[1].shape)
        model = self.create_model(data)
        self.train_model(model, data)

class Res50Network():
    def __init__(self, filepath, segment_size, segments_per_patient, epochs):
        self.master_filepath = filepath # replace this with your filepath to the "MS Fall Study" directory
        self.segment_size = segment_size # this variable represents the number of sequential data measurements that are part of each "segment" of x data
        self.segments_per_patient = segments_per_patient # this variable represents the number of segments to get from each patient
        self.batch_size = 64
        self.epochs = epochs
        self.validation_split=0.4
        return

    def create_model(self, x_data, y_data):
        print('create ResNet50 model')
        base_model = ResNet50(include_top=False, weights=None,  pooling=None, classes=y_data.shape[1],input_shape=(x_data.shape[1],x_data.shape[2],x_data.shape[3]))
        # add a global spatial average pooling layer
        x = base_model.output
        x = keras.layers.GlobalAveragePooling2D()(x)
        # add a fully-connected layer
        x = keras.layers.Dense(1024, activation='relu')(x)
        # and a logistic layer -- with numc classes
        predictions = keras.layers.Dense(y_data.shape[1], activation='softmax')(x)
        # this is the model to train
        model = keras.models.Model(inputs=base_model.input, outputs=predictions)
        model.compile(optimizer=tf.train.AdamOptimizer(learning_rate=0.001), loss='categorical_crossentropy',metrics=['accuracy'])
        print(model.summary())
        return (model)

    def train_model(self, model, x_data, y_data):
        print('train ResNet50')
        model.fit(x=x_data, y=y_data, batch_size=self.batch_size, epochs=self.epochs, validation_split=self.validation_split)
        model.save_weights("r50_weights.h5")

    def load_data(self):
        reader = DataReader(self.master_filepath)
        data = reader.get_segmented_data(self.segment_size, self.segments_per_patient)
        return (data)

    def run(self):
        (x_data, y_data) = self.load_data()
        x_data = x_data.reshape((x_data.shape[0], x_data.shape[1], x_data.shape[2], 1))
        model = self.create_model(x_data, y_data)
        self.train_model(model, x_data, y_data)

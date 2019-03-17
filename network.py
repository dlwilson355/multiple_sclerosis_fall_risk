import keras
from data_reader import DataReader
import os
import pickle

def create_model():
    model = keras.Sequential()
    # this is a simple MLP I made
    # we will use something more sophisticated (and better suited) like an RNN for the final project
    model.add(keras.layers.Dense(2000, input_shape=(2000, 4)))
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
    model.add(keras.layers.Dense(2))
    model.add(keras.layers.Activation('softmax'))
    opt = keras.optimizers.SGD(lr=0.000000000001)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    print(model.summary())
    return (model)

def train_model(model, data):
    print(model.summary())
    model.fit(x=data[0], y=data[1], batch_size=16, epochs=20, validation_split=0.5)
    model.save_weights("weights.h5")

def load_data():
    reader = DataReader("D:\\deep learning dataset\\MS Fall Study")
    sample_data = reader.get_pickled_data()
    return (sample_data)

def main():
    model = create_model()
    data = load_data()
    train_model(model, data)

main()
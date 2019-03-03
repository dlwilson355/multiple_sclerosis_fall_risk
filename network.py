import keras
from data_reader import DataReader
import os

def create_model():
    model = keras.Sequential()
    # this is a simple MLP I found online, I am temporarily using it to start testing code
    # we will use something more sophisticated (and better suited) like an RNN for the final project
    model.add(keras.layers.Dense(128, input_shape=(2000, 4)))
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.Dropout(0.15))
    model.add(keras.layers.Dense(128))
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.Dropout(0.15))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(1))
    model.add(keras.layers.Activation('softmax'))
    model.compile(loss='binary_crossentropy', optimizer='rmsprop')
    print(model.summary())
    return (model)

def train_model(model, data):
    print(type(model))
    print(type(data))
    print(model.summary())
    model.fit(x=data[0], y=data[1])

def load_data():
    sample_data = DataReader("MS Fall Study").get_data()
    return (sample_data)

def main():
    model = create_model()
    data = load_data()
    print(data[0].shape)
    print(data[1].shape)
    train_model(model, data)

main()
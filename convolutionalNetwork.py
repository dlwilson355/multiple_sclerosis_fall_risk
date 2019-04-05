import keras
from data_reader import DataReader
import numpy as np

def create_model(x_data, y_data):
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

def train_model(model, x_data, y_data):
    model.fit(x=x_data, y=y_data, batch_size=20, epochs=20, validation_split=0.25)
    model.save_weights("weights.h5")

def load_data():
    MASTER_FILEPATH = "D:\\deep learning dataset\\MS Fall Study" # replace this with your filepath to the "MS Fall Study" directory
    SEGMENT_SIZE = 40 # this variable represents the number of sequential data measurements that are part of each "segment" of x data
    SEGMENTS_PER_PATIENT = 50 # this variable represents the number of segments to get from each patient

    reader = DataReader(MASTER_FILEPATH)
    #data = reader.get_segmented_data(SEGMENT_SIZE, SEGMENTS_PER_PATIENT)
    #reader.save_pickle(data)
    data = reader.get_pickled_data()
    return (data)

def main():
    (x_data, y_data) = load_data()
    x_data = x_data.reshape((x_data.shape[0], x_data.shape[1], x_data.shape[2], 1))
    model = create_model(x_data, y_data)
    train_model(model, x_data, y_data)

main()
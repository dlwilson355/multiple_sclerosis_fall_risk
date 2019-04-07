import keras
from data_reader import DataReader
import numpy as np

def create_model(x_data, y_data):
    model = keras.Sequential()

    # first convolutional layer
    model.add(keras.layers.Conv2D(100, input_shape=(x_data.shape[1], x_data.shape[2], 1), kernel_size=(10, 1), use_bias=False))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation('relu'))

    # second convolutional layer
    model.add(keras.layers.Conv2D(100, kernel_size=(3, 3)))
    model.add(keras.layers.Activation('relu'))

    # softmax layer
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(y_data.shape[1]))
    model.add(keras.layers.Activation('softmax'))


    opt = keras.optimizers.SGD(lr=0.001)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    print(model.summary())
    return (model)

def create_alexnet(x_data, y_data):
    model = keras.Sequential()
    model.add(keras.layers.Conv2D(filters=96, input_shape=(x_data.shape[1], x_data.shape[2], 1), kernel_size=(3,3), strides=(1,1), padding='valid'))
    model.add(keras.layers.Activation('relu'))
    # Max Pooling
    model.add(keras.layers.MaxPooling2D(pool_size=(2,2), strides=(1,1), padding='valid'))

    # 2nd Convolutional Layer
    model.add(keras.layers.Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='valid'))
    model.add(keras.layers.Activation('relu'))
    # Max Pooling
    model.add(keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))

    # 3rd Convolutional Layer
    model.add(keras.layers.Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid'))
    model.add(keras.layers.Activation('relu'))

    # 4th Convolutional Layer
    model.add(keras.layers.Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid'))
    model.add(keras.layers.Activation('relu'))

    # 5th Convolutional Layer
    model.add(keras.layers.Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='valid'))
    model.add(keras.layers.Activation('relu'))
    # Max Pooling
    model.add(keras.layers.MaxPooling2D(pool_size=(2,2), strides=(1,1), padding='valid'))

    # Passing it to a Fully Connected layer
    model.add(keras.layers.Flatten())
    # 1st Fully Connected Layer
    model.add(keras.layers.Dense(4096, input_shape=(224*224*3,)))
    model.add(keras.layers.Activation('relu'))
    # Add Dropout to prevent overfitting
    model.add(keras.layers.Dropout(0.4))

    # 2nd Fully Connected Layer
    model.add(keras.layers.Dense(4096))
    model.add(keras.layers.Activation('relu'))
    # Add Dropout
    model.add(keras.layers.Dropout(0.4))

    # 3rd Fully Connected Layer
    model.add(keras.layers.Dense(1000))
    model.add(keras.layers.Activation('relu'))
    # Add Dropout
    model.add(keras.layers.Dropout(0.4))

    # Output Layer
    model.add(keras.layers.Dense(y_data.shape[1]))
    model.add(keras.layers.Activation('softmax'))
    opt = keras.optimizers.Adam(lr=0.001)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    print(model.summary())
    return (model)

def train_model(model, x_data, y_data):
    model.fit(x=x_data, y=y_data, batch_size=200, epochs=10, validation_split=0.25)
    model.save_weights("weights.h5")

def load_data():
    MASTER_FILEPATH = "D:\\deep learning dataset\\MS Fall Study" # replace this with your filepath to the "MS Fall Study" directory
    SEGMENT_SIZE = 50 # this variable represents the number of sequential data measurements that are part of each "segment" of x data
    SEGMENTS_PER_PATIENT = 300 # this variable represents the number of segments to get from each patient

    reader = DataReader(MASTER_FILEPATH)
    #data = reader.get_segmented_data(SEGMENT_SIZE, SEGMENTS_PER_PATIENT)
    #reader.save_pickle(data)
    data = reader.get_pickled_data()
    return (data)

def main():
    (x_data, y_data) = load_data()
    print("Shapes")
    print(x_data.shape)
    print(y_data.shape)
    x_data = x_data.reshape((x_data.shape[0], x_data.shape[1], x_data.shape[2], 1))
    model = create_alexnet(x_data, y_data)
    train_model(model, x_data, y_data)

main()
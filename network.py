import keras
from data_reader import DataReader
import os
import sys, getopt
from recurrentnetwork import RNNRnunner
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

def Help():
    print('''network.py 
             -g <use test data generator, def: 0>
             -t <RNN type: simple, GRU, LSTM, def: simple> 
             -d <training data size, def: 320> 
             -e <number of epochs, def: 30> 
             -s <steps in the sequence, def: 6> 
             -h <number of hidden units, def:75>
             -v <verbose,def:1>
             -m <multi threaded, def:0''')

def main(argv):
    # setup options from command line
    testDataGenerator = False
    rnnType = 'simple'
    trainingDataSize = 320
    numberOfEpochs = 30
    stepsInSequence = 6 
    hiddenUnits = 75
    verbose = 1
    multiThreaded = 0
    try:
        opts, args = getopt.getopt(argv,"?g:t:d:e:s:h:v:m:")
    except getopt.GetoptError:
        Help()
        return
    for opt, arg in opts:
        if opt == '-?':
            Help()
            return
        elif opt == '-g':
            testDataGenerator = int(arg)
        elif opt == '-t':
            rnnType = arg
        elif opt == '-d':
            trainingDataSize = int(arg)
        elif opt == '-e':
            numberOfEpochs = int(arg)
        elif opt == '-s':
            stepsInSequence = int(arg)
            # steps has to be an even number
            stepsInSequence = int(stepsInSequence) * 2
        elif opt == '-h':
            hiddenUnits = int(arg)
        elif opt == '-v':
            verbose = int(arg)
        elif opt == '-m':
            multiThreaded = int(arg)

    if testDataGenerator:
        run = RNNRnunner(verbose,multiThreaded)
        run.RunRNN(rnnType,trainingDataSize,numberOfEpochs, stepsInSequence,hiddenUnits)
    else:
        model = create_model()
        data = load_data()
        print(data[0].shape)
        print(data[1].shape)
        train_model(model, data)

if __name__ == "__main__":  
    main(sys.argv[1:])

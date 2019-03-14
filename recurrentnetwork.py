import numpy as np
import tensorflow as tf
import keras
from keras.models import Model
from testdatagenerator import TestDataGenerator

class RNNRnunner(object):
    def __init__(self, verb, multi):
        self.num_predict=32
        self.num_features = 1
        self.batch_size = 32
        self.be_verbose = verb
        if multi:
            self.num_workers = 5
            self.do_multiprocess = True
        else:
            self.num_workers = 0
            self.do_multiprocess = False

    def Compile(self, model, lr=0.001,lossFct='mean_squared_error'):
        #model.compile(optimizer = keras.optimizers.Adam(clipnorm=1.), loss=lossFct,metrics=['accuracy'])
        model.compile(optimizer = keras.optimizers.RMSprop(clipnorm=1.), loss=lossFct,metrics=['accuracy'])
        return

    def Fit(self, model, numTrain, numepoch):
        trainGen = TestDataGenerator(numTrain, self.batch_size, self.length_of_sequence)
        valGen = TestDataGenerator(self.num_val, self.batch_size, self.length_of_sequence)
        hist = model.fit_generator(generator=trainGen, validation_data=valGen, 
                            epochs=numepoch, verbose=self.be_verbose, workers=self.num_workers,
                            use_multiprocessing=self.do_multiprocess)
        print(hist.history)
        return

    def Predict(self, model):
        print('Predict')
        preGen = TestDataGenerator(1, 1, self.length_of_sequence)
        errCount = 0
        for i in range(self.num_predict):
            x,y = preGen.GetData()
            p = model.predict(x)
            diff = abs(p[0][0]-y[0][0])
            if diff >= 0.01:
                print('count {0:d} diff {1:f} expect {2:f} predict {3:f}'.format(errCount,diff,y[0][0],p[0][0]))
                errCount += 1
        if errCount == 0:
            print('no error')
        return

    def SimpleRNN(self, numTrain, numEpoch):
        model=keras.models.Sequential()
        model.add(keras.layers.SimpleRNN(units=self.hiddenunits,input_shape=(self.length_of_sequence,1), activation='relu', return_sequences = False))
        ol = keras.layers.Dense(self.num_features, activation = "sigmoid")
        model.add(ol)
        return model

    def GRU(self, numTrain, numEpoch):
        model=keras.models.Sequential()  
        model.add(keras.layers.GRU(units=self.hiddenunits,input_shape=(self.length_of_sequence,1), activation='relu', return_sequences = False))
        ol = keras.layers.Dense(self.num_features, activation = "sigmoid")
        model.add(ol)
        return model

    def LSTM(self, numTrain, numEpoch):
        model=keras.models.Sequential()
        model.add(keras.layers.LSTM(units=self.hiddenunits,input_shape=(self.length_of_sequence,1), activation='tanh', return_sequences = False))
        ol = keras.layers.Dense(self.num_features, activation = "sigmoid")
        model.add(ol)
        return model

    def RunRNN(self, type,numTrain, numEpoch, steps, hiddenunits):
        print('##################################################################################')
        print('# RunRNN({0:s}, {1:d}, {2:d}, {3:d}, {4:d})                                      #'.format(type,numTrain, numEpoch, steps, hiddenunits))
        print('##################################################################################')
        self.length_of_sequence = steps
        self.hiddenunits = hiddenunits
        self.num_val = int(numTrain / 10)
        if self.num_val < 1:
            self.num_val = 1
        elif self.num_val > 128:
            self.num_val = 128
        if 'LSTM'== type:
            model = self.LSTM(numTrain, numEpoch)
        elif 'GRU' == type:
            model = self.GRU(numTrain, numEpoch)
        else:
            model = self.SimpleRNN(numTrain, numEpoch)
        self.Compile(model)
        #model.summary()
        self.Fit(model,numTrain, numEpoch)
        self.Predict(model)
        return


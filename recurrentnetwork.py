import numpy as np
import tensorflow as tf
import keras
from keras.models import Model
from keras import backend as K
#from testdatagenerator import DataGenerator
#from datagenerator import DataGenerator
#from segmentdatagenerator import DataGenerator
from tsdatagenerator import DataGenerator

def recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    all_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    return true_positives / (all_positives + K.epsilon())

class RNNRnunner(object):
    def __init__(self, verb, multi,folder,numFeatures):
        self.folder = folder
        self.num_predict=100
        self.batch_size = 32
        self.be_verbose = verb
        self.weightsfile = 'weights.h5'
        self.num_features = numFeatures
        if multi:
            self.num_workers = 5
            self.do_multiprocess = True
        else:
            self.num_workers = 0
            self.do_multiprocess = False


    def Compile(self, model,lossFct='mean_squared_error', lr=0.005):
        #opt = keras.optimizers.Adam(clipnorm=1., lr=lr)
        #opt = keras.optimizers.Adam(lr=lr)
        #opt = keras.optimizers.RMSprop(clipnorm=1., lr=lr)
        opt = keras.optimizers.RMSprop(lr=lr,decay=0.1)
        met = ['accuracy', recall]
        model.compile(optimizer = opt, loss=lossFct,metrics=met)
        return

    def Fit(self, model, numTrain, numepoch,data):
        if None == data:
            trainGen = DataGenerator(self.folder,'train',numTrain, self.batch_size, self.length_of_sequence,self.num_features)
            valGen = DataGenerator(self.folder,'validate',self.num_val, self.batch_size, self.length_of_sequence,self.num_features)
            hist = model.fit_generator(generator=trainGen, validation_data=valGen, 
                            epochs=numepoch, verbose=self.be_verbose, workers=self.num_workers,
                            use_multiprocessing=self.do_multiprocess)
        else:
            hist = model.fit(x=data[0], y=data[1], batch_size=self.batch_size, epochs=numepoch, validation_split=0.5)
        #print(hist.history)
        model.save_weights(self.weightsfile)
        return

    def getY(self, y):
        i = 0
        max = 0
        index = 0
        for n in y:
            if n > max:
                index = i
                max = n
            #print(n)
            i += 1
        return index

    def Predict(self, model,data):
        if None == data:
            print('Predict')
            preGen = DataGenerator(self.folder,'predict', 1, 1, self.length_of_sequence,self.num_features)
            errCount = 0
            for i in range(self.num_predict):
                x,y1h = preGen.GetData(i)
                p1h = model.predict(x)
                y = int(self.getY(y1h[0]))
                p = int(self.getY(p1h[0]))
                if p != y:
                    print('count {0:d} expect {1:d} predict {2:d}'.format(errCount,y,p))
                    errCount += 1
            if errCount == 0:
                print('no error')
        else:
            print('Predict')
            #samp = 0
            #p = model.predict(data[0][samp])
            #diff = abs(p[0][0]-data[1][samp])
            #if diff >= 0.01:
            #    print('count {0:d} diff {1:f} expect {2:f} predict {3:f}'.format(errCount,diff,data[1][samp],p[0][0]))            
        return

    def SimpleRNN(self, numTrain, numEpoch,actFct):
        model=keras.models.Sequential()
        model.add(keras.layers.SimpleRNN(units=self.hiddenunits,input_shape=(self.length_of_sequence,self.num_sequences), activation='tanh', return_sequences = False))
        ol = keras.layers.Dense(self.num_features, activation = actFct)
        model.add(ol)
        return model

    def GRU(self, numTrain, numEpoch,actFct):
        model=keras.models.Sequential()  
        model.add(keras.layers.GRU(units=self.hiddenunits,input_shape=(self.length_of_sequence,self.num_sequences), activation='tanh'))
        ol = keras.layers.Dense(self.num_features, activation = actFct)
        model.add(ol)
        return model

    def LSTM(self, numTrain, numEpoch,actFct):
        model=keras.models.Sequential()
        model.add(keras.layers.LSTM(units=self.hiddenunits,input_shape=(self.length_of_sequence,self.num_sequences), activation='tanh', return_sequences = False))
        ol = keras.layers.Dense(self.num_features, activation = actFct)
        model.add(ol)
        return model

    def RunRNN(self, type,numTrain, numEpoch, steps, hiddenunits, data):
        print('##################################################################################')
        print('# RunRNN({0:s}, {1:d}, {2:d}, {3:d}, {4:d})                                      #'.format(type,numTrain, numEpoch, steps, hiddenunits))
        print('##################################################################################')
        if None == data:
            self.length_of_sequence = steps
            self.num_sequences = 3
            lossFct='mean_squared_error'
            actFct = 'sigmoid'
        else:
            self.length_of_sequence = data[0].shape[1]
            self.num_sequences = data[0].shape[2]
            #lossFct='categorical_crossentropy'
            #lossFct='binary_crossentropy'
            lossFct='mean_squared_error'
            actFct = 'sigmoid'
            #actFct = 'softmax'
        self.hiddenunits = hiddenunits
        self.num_val = int(numTrain * 0.6)
        if self.num_val < 1:
            self.num_val = 1
        self.weightsfile = 'weights_' + type + '.h5'
        if 'LSTM'== type:
            model = self.LSTM(numTrain, numEpoch,actFct)
        elif 'GRU' == type:
            model = self.GRU(numTrain, numEpoch,actFct)
        else:
            model = self.SimpleRNN(numTrain, numEpoch,actFct)
        self.Compile(model,lossFct)
        #model.summary()
        self.Fit(model,numTrain, numEpoch,data)
        self.Predict(model,data)
        return


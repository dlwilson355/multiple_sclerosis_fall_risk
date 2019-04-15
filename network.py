import keras
from data_reader import DataReader
import os
import sys, getopt
from recurrentnetwork import RNNRnunner
import pickle
from convolutionalNetwork import ConvNetwork, MLPNetwork
from matrixDataGenerator import MatrixDataGenerator

def Help():
    print('''network.py 
             -f <"folder", def: "D:\\deep learning dataset\\MS Fall Study">
             -t <Network type: MLP, CNN, ResNet, NASNet, simple, GRU, LSTM, def: MLP> 
             -g <use data generator, def: 0>
             -d <training data size, def: 320> 
             -e <number of epochs, def: 30> 
             -s <steps in the sequence, def: 6> 
             -h <number of hidden units, def:75>
             -v <verbose,def:1>
             -m <multi threaded, def:0''')

def main(argv):
    # setup options from command line
    useDataGenerator = False
    netType = 'MLP'
    trainingDataSize = 320
    numberOfEpochs = 30
    stepsInSequence = 6 
    hiddenUnits = 75
    verbose = 1
    multiThreaded = 0
    folder = "D:\\deep learning dataset\\MS Fall Study"
    try:
        opts, args = getopt.getopt(argv,"?f:g:t:d:e:s:h:v:m:")
    except getopt.GetoptError:
        Help()
        return
    for opt, arg in opts:
        if opt == '-?':
            Help()
            return
        elif opt == '-f':
            folder = arg
        elif opt == '-g':
            useDataGenerator = int(arg)
        elif opt == '-t':
            netType = arg
        elif opt == '-d':
            trainingDataSize = int(arg)
        elif opt == '-e':
            numberOfEpochs = int(arg)
        elif opt == '-s':
            stepsInSequence = int(arg)
            # steps has to be an even number
            #stepsInSequence = int(stepsInSequence) * 2
        elif opt == '-h':
            hiddenUnits = int(arg)
        elif opt == '-v':
            verbose = int(arg)
        elif opt == '-m':
            multiThreaded = int(arg)

    if 'MLP'== netType:
        mlpNet = MLPNetwork(folder)
        mlpNet.run()
    elif 'CNN'== netType:
        convNet = ConvNetwork(folder, stepsInSequence,trainingDataSize, numberOfEpochs)
        convNet.run()
    elif useDataGenerator:
        run = RNNRnunner(verbose,multiThreaded,folder,0)
        run.RunRNN(netType,trainingDataSize,numberOfEpochs, stepsInSequence,hiddenUnits, None)
    else:
        reader = DataReader(folder)
        data = reader.get_data(True,True)
        numFeatures = reader.NumberOfFeatures()
        print(data[0].shape)
        print(data[1].shape)
        run = RNNRnunner(verbose,multiThreaded,folder,numFeatures)
        run.RunRNN(netType,trainingDataSize,numberOfEpochs, stepsInSequence,hiddenUnits, data)

if __name__ == "__main__":  
    main(sys.argv[1:])

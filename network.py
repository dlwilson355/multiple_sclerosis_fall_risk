from keras.applications.vgg16 import VGG16
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
import sys, getopt
from recurrentnetwork import RNNImp
from visualize import Visualize
from table import DataTable
from vgg16 import VGG16Imp
from resnet import ResNetImp
from matrixDataGenerator import MatrixDataGenerator, MatrixPreLoader


def create_generators(preLoader, input_shape, rgb, twoD, gaus, batchSize):
        training_generator = MatrixDataGenerator(preLoader, matrix_dimensions = input_shape, rgb = rgb, twoD = twoD, add_gaussian_noise = gaus, zero_sensors = 0, batch_size = batchSize, grab_data_from = (0, .7), overflow = "BEFORE", print_loading_progress = False)
        validation_generator = MatrixDataGenerator(preLoader, matrix_dimensions = input_shape, rgb = rgb, twoD = twoD, add_gaussian_noise = 0, zero_sensors = 0, batch_size = batchSize, grab_data_from = (.7, 1), overflow = "AFTER", print_loading_progress = False)
        return training_generator, validation_generator

def train_model_with_generator(model, training_generator, validation_generator,numberOfEpochs,netType):
        part_file = "weights({0:s})".format(netType)
        weights_filepath = part_file+"-{epoch:02d}-{val_acc:.2f}.hdf5"
        checkpoint = ModelCheckpoint(weights_filepath, monitor="val_acc", verbose=1, save_best_only=True, mode="max")
        optimizer = Adam(lr=0.0001)
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        model.fit_generator(training_generator, epochs=numberOfEpochs, steps_per_epoch=10, validation_steps=10, validation_data=validation_generator, callbacks=[checkpoint])

NETTYPE_INVALID = 0
NETTYPE_VGGBN = 1
NETTYPE_VGG16 = 2
NETTYPE_RESNET = 3
NETTYPE_SIMPLE = 4
NETTYPE_GRU = 5
NETTYPE_LSTM = 6
NETTYPE_RESNETP = 7

def GetType(netType):
    if 'VGGBN'== netType:
        return NETTYPE_VGGBN
    elif 'VGG16'== netType:
        return NETTYPE_VGG16
    elif 'ResNet'== netType:
        return NETTYPE_RESNET
    elif 'Simple' == netType:
        return NETTYPE_SIMPLE
    elif 'GRU' == netType:
        return NETTYPE_GRU
    elif 'LSTM' == netType:
        return NETTYPE_LSTM
    elif 'ResNetP'== netType:
        return NETTYPE_RESNETP
    return NETTYPE_INVALID


def Help():
    print('''network.py 
             -f <folder, def: "D:\\deep learning dataset\\MS Fall Study">
             -t <Network type: ResNet, ResNetP, VGGBN, VGG16, Simple, GRU, LSTM, Plt,Img,Table def: ResNet> 
                Plt - generate plots for all patients and all activities over all sensors
                Img - generate images for all patients and all activities over all sensors
                Table - generate csv file for all patients, all activities, all sensors with indices, frequencies, min, max, mean and stdev
                ResNetP - pretrained ResNet
             -1 <first part of input shape argument, def: 224> 
             -2 <second part of input shape argument, def: 224> 
             -w <weights file, def: ''>
             -g <gaussian noise for training, def: 0>
             -e <number of epochs, def: 30> 
             -h <number of hidden units, def:75>
             -p <number of patients to use, def:ALL>''')

def main(argv):
    # setup options from command line
    netType = 'ResNet'
    input_shape1 = 224
    input_shape2 = 224
    gaus = 0
    numberOfEpochs = 30
    hiddenUnits = 75
    patients_to_use = 'ALL'
    weights = ''
    folder = "D:\\deep learning dataset\\MS Fall Study"
    try:
        opts, args = getopt.getopt(argv,"?f:t:1:2:g:e:h:p:w:")
    except getopt.GetoptError:
        Help()
        return
    for opt, arg in opts:
        if opt == '-?':
            Help()
            return
        elif opt == '-f':
            folder = arg
        elif opt == '-w':
            weights = arg
        elif opt == '-t':
            netType = arg
        elif opt == '-1':
            input_shape1 = int(arg)
        elif opt == '-2':
            input_shape2 = int(arg)
        elif opt == '-g':
            gaus = float(arg)
        elif opt == '-e':
            numberOfEpochs = int(arg)
        elif opt == '-h':
            hiddenUnits = int(arg)
        elif opt == '-p':
            try:
                patients_to_use = int(arg)
            except ValueError:
                patients_to_use = arg



    if 'Plt' == netType:
        vis = Visualize(folder, patients_to_use, True)
        vis.run()
    elif 'Img' == netType:
        vis = Visualize(folder, patients_to_use, False)
        vis.run()
    elif 'Table'== netType:
        table = DataTable(folder)
        table.run()
    else:
        netTypeVal = GetType(netType)
        if NETTYPE_INVALID == netTypeVal:
            print('unknown tpye:', netType)
            return
        print('##################################################################################')
        print('# {0:s} shape ({1:d}, {2:d}) epochs {3:d} gaus {4:f} hidden units {5:d}          #'.format(netType,input_shape1, input_shape2, numberOfEpochs, gaus, hiddenUnits))
        print('##################################################################################')
        rgb = True
        twoD = False
        batchSize = 32
        input_shape = (input_shape1, input_shape2)
        activities_to_load = ["30s Chair Stand Test", "Tandem Balance Assessment", "Standing Balance Assessment", "Standing Balance Eyes Closed", "ADL: Normal Walking", "ADL: Normal Standing", "ADL: Normal Sitting", "ADL: Slouch sitting", "ADL: Lying on back", "ADL: Lying on left side", "ADL: Lying on right side"]
        preLoader = MatrixPreLoader(dataset_directory = folder, patients_to_use = patients_to_use, activity_types = activities_to_load, print_loading_progress = True)
        num_features = preLoader.get_number_of_patients()
        if NETTYPE_VGGBN == netTypeVal:
            vgg = VGG16Imp()
            model = vgg.VGG16WithBN(input_shape=(input_shape1,input_shape2,3), classes=num_features)
        elif NETTYPE_VGG16 == netTypeVal:
            model = VGG16(weights=None, classes = num_features,input_shape=(input_shape1,input_shape2,3))
        elif NETTYPE_RESNET == netTypeVal:
            resnet = ResNetImp()
            model = resnet.ResNet((input_shape1,input_shape2,3),num_features)
        elif NETTYPE_RESNETP == netTypeVal:
            resnet = ResNetImp()
            model = resnet.ResNetP(weights,(input_shape1,input_shape2,3),num_features)
        elif NETTYPE_SIMPLE == netTypeVal:
            rgb = False
            twoD = True
            rnn = RNNImp(hiddenUnits)
            model = rnn.SimpleRNN(input_shape, num_features)
        elif NETTYPE_GRU == netTypeVal:
            rgb = False
            twoD = True
            rnn = RNNImp(hiddenUnits)
            model = rnn.GRU(input_shape, num_features)
        elif NETTYPE_LSTM == netTypeVal:
            rgb = False
            twoD = True
            rnn = RNNImp(hiddenUnits)
            model = rnn.LSTM(input_shape, num_features)
        else:
            return

        print("create_generators")
        training_gen, validation_gen = create_generators(preLoader, input_shape, rgb, twoD, gaus, batchSize)

        print("train_model_with_generator")
        train_model_with_generator(model, training_gen, validation_gen, numberOfEpochs,netType)

        training_dist = training_gen.GetDistribution()
        validation_dist = validation_gen.GetDistribution()
        print('training distribution', training_dist)
        print('validation distribution', validation_dist)


if __name__ == "__main__":  
    main(sys.argv[1:])
#-f "E:\MS Fall Study"  -t ResNet -1 224 -2 224 -g 0 -e 4 -p "S0002 S0003 S0005 S0008 S0015 S0017 S0019 S0020 S0021 S0022"
#-f "E:\MS Fall Study"  -t ResNetP -1 224 -2 224 -g 0 -e 4 -w "weights(ResNet)-11-1.00.hdf5" -p "S0002 S0003 S0005 S0008 S0015 S0017 S0019 S0020 S0021 S0022 S0012 S0013 S0014 S0016"
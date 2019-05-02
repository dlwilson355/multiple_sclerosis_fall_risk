"""
A test file for the feature extractor.
"""
import keras
import tensorflow as tf
from keras import Model
from keras.applications.vgg16 import VGG16
from matrixDataGenerator import MatrixDataGenerator, MatrixPreLoader
from featureExtractor import FeatureExtractor

patient_fall_filepath = "D:\\deep learning dataset\\MS Fall Study\\SubjectInfo.csv"
training_filepath = "D:\\deep learning dataset\\training features" # the filepath to the sensor data for the patients from which features will be extracted when training
testing_filepath = "D:\\deep learning dataset\\testing features" # the filepath to the sensor data for the patients from which features will be extracted when testing
weigths_filepath = "D:\\deep learning dataset\\weights(ResNet)-27-1.00.hdf5" # the filepath to the weights of the classification model
activities_to_load = ["30s Chair Stand Test", "Tandem Balance Assessment", "Standing Balance Assessment", "Standing Balance Eyes Closed", "ADL: Normal Walking", "ADL: Normal Standing", "ADL: Normal Sitting", "ADL: Slouch sitting", "ADL: Lying on back", "ADL: Lying on left side", "ADL: Lying on right side"]

def create_generators():
    # creating training generator
    preloader = MatrixPreLoader(dataset_directory = training_filepath, num_patients_to_use = 1, activity_types = activities_to_load, print_loading_progress = False)
    matrix_data_generator = MatrixDataGenerator(preloader, matrix_dimensions = (224, 224), rgb = True, twoD = False, add_gaussian_noise = 0, zero_sensors = 0, batch_size = 32, grab_data_from = (0, 1), overflow = "BEFORE", print_loading_progress = False)
    training_generator = FeatureExtractor(matrix_data_generator, patient_fall_filepath, weigths_filepath, test=True)

    # create testing generator
    preloader = MatrixPreLoader(dataset_directory = testing_filepath, num_patients_to_use = 1, activity_types = activities_to_load, print_loading_progress = False)
    matrix_data_generator = MatrixDataGenerator(preloader, matrix_dimensions = (224, 224), rgb = True, twoD = False, add_gaussian_noise = 0, zero_sensors = 3, batch_size = 32, grab_data_from = (0, 1), overflow = "BEFORE", print_loading_progress = False)
    testing_generator = FeatureExtractor(matrix_data_generator, patient_fall_filepath, weigths_filepath, test=False)

    return training_generator, testing_generator

def create_model():
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(512, input_dim = 1024))
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.Dense(128))
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.Dense(32))
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.Dense(2))
    model.add(keras.layers.Activation('softmax'))
    optimzer = keras.optimizers.Adam(lr=0.001)
    model.compile(loss = 'categorical_crossentropy', optimizer = optimzer, metrics=['accuracy'])
    return (model)

def train_model(model, training_generator, validation_generator):
    weights_filepath = "weights(fall risk prediction model)-{epoch:02d}-{val_acc:.2f}.hdf5"
    checkpoint = keras.callbacks.ModelCheckpoint(weights_filepath, monitor="val_acc", verbose=1, save_best_only=True, mode="max")
    model.fit_generator(training_generator, validation_data=validation_generator, epochs=100, steps_per_epoch=10, validation_steps=10, callbacks=[checkpoint])

def main():
    training_gen, validation_gen = create_generators()
    model = create_model()
    train_model(model, training_gen, validation_gen)

if __name__ == "__main__":
    main()
from keras.applications.resnet50 import ResNet50
from matrixDataGenerator import MatrixDataGenerator

data_filepath = "D:\\deep learning dataset\\MS Fall Study"

def create_generators():
        training_generator = MatrixDataGenerator(directory = data_filepath, matrix_dimensions = (224, 224), rgb = True, batch_size = 3, grab_data_from = (0, .75), debug = False)
        validation_generator = MatrixDataGenerator(directory = data_filepath, matrix_dimensions = (224, 224), rgb = True, batch_size = 3, grab_data_from = (.75, 1), debug = False)
        return training_generator, validation_generator

def train_model_with_generator(model, training_generator, validation_generator):
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
    model.fit_generator(training_generator, epochs=10, validation_data = validation_generator)
    model.save_weights("resNet_weights.h5")

def main():
    training_gen, validation_gen = create_generators()
    sample_X, sample_Y = training_gen.__getitem__() # we get sample data because we use it to determine the data dimensions when we create the model
    model = ResNet50(weights=None, classes = sample_Y.shape[1])
    train_model_with_generator(model, training_gen, validation_gen)

if __name__ == "__main__":
    main()
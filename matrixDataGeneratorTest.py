from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from matrixDataGenerator import MatrixDataGenerator, MatrixPreLoader

data_filepath = "D:\\deep learning dataset\\MS Fall Study" # you will obviously need to change this to the appropriate filepath based on where you placed the data folder
data_filepath = "data"

def create_generators():
        preLoader = MatrixPreLoader(data_filepath, num_patients_to_use = 15, print_loading_progress = False, debug = False)
        training_generator = MatrixDataGenerator(preLoader, matrix_dimensions = (224, 224), rgb = True, twoD = False, add_gaussian_noise = .01, batch_size = 32, grab_data_from = (0, .75), print_loading_progress = False, debug = False)
        validation_generator = MatrixDataGenerator(preLoader, matrix_dimensions = (224, 224), rgb = True, twoD = False, add_gaussian_noise = None, batch_size = 32, grab_data_from = (.75, 1), print_loading_progress = False, debug = False)
        return training_generator, validation_generator

def train_model_with_generator(model, training_generator, validation_generator):
        weights_filepath = "weights(first 15 patients training)-{epoch:02d}-{val_acc:.2f}.hdf5"
        checkpoint = ModelCheckpoint(weights_filepath, monitor="val_acc", verbose=1, save_best_only=True, mode="max")
        optimizer = Adam(lr=0.0001)
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        model.fit_generator(training_generator, epochs=10, steps_per_epoch=10, validation_steps=10, validation_data=validation_generator, callbacks=[checkpoint])
        model.save_weights("resNet_weights.h5")

def main():
        training_gen, validation_gen = create_generators()
        sample_X, sample_Y = training_gen.__getitem__() # we get sample data because we use it to determine the data dimensions when we create the model
        model = VGG16(weights=None, classes = sample_Y.shape[1])
        train_model_with_generator(model, training_gen, validation_gen)

if __name__ == "__main__":
        main()

from keras.applications.vgg16 import VGG16
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from matrixDataGenerator import MatrixDataGenerator, MatrixPreLoader

data_filepath = "D:\\deep learning dataset\\MS Fall Study" # you will obviously need to change this to the appropriate filepath based on where you placed the data folder

def create_generators():
        activities_to_load = ["30s Chair Stand Test", "Tandem Balance Assessment", "Standing Balance Assessment", "Standing Balance Eyes Closed", "ADL: Normal Walking", "ADL: Normal Standing", "ADL: Normal Sitting", "ADL: Slouch sitting", "ADL: Lying on back", "ADL: Lying on left side", "ADL: Lying on right side"]
        preLoader = MatrixPreLoader(dataset_directory = data_filepath, patients_to_use = "ALL", activity_types = activities_to_load, print_loading_progress = False)
        training_generator = MatrixDataGenerator(preLoader, matrix_dimensions = (224, 224), rgb = True, twoD = False, add_gaussian_noise = .01, zero_sensors = 3, batch_size = 32, grab_data_from = (0, .75), overflow = "BEFORE", print_loading_progress = False)
        validation_generator = MatrixDataGenerator(preLoader, matrix_dimensions = (224, 224), rgb = True, twoD = False, add_gaussian_noise = 0, zero_sensors = 0, batch_size = 100, grab_data_from = (.75, 1), overflow = "AFTER", print_loading_progress = False)
        return training_generator, validation_generator

def train_model_with_generator(model, training_generator, validation_generator):
        weights_filepath = "weights(first 15 patients training)-{epoch:02d}-{val_acc:.2f}.hdf5"
        checkpoint = ModelCheckpoint(weights_filepath, monitor="val_acc", verbose=1, save_best_only=True, mode="max")
        optimizer = Adam(lr=0.0001)
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        model.fit_generator(training_generator, epochs=10, steps_per_epoch=10, validation_steps=10, validation_data=validation_generator, callbacks=[checkpoint])

def main():
        training_gen, validation_gen = create_generators()
        sample_X, sample_Y = training_gen.__getitem__() # we get sample data because we use it to determine the data dimensions when we create the model
        model = VGG16(weights=None, classes = sample_Y.shape[1])
        train_model_with_generator(model, training_gen, validation_gen)

if __name__ == "__main__":
        main()

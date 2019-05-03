"""
The feature extractor class.
This is used to extract features from a patient classification model.  These features are then used to determine fall risk.
"""
import keras
from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50
import tensorflow as tf
import numpy as np
import pandas as pd
from resnet import ResNetImp

class FeatureExtractor(keras.utils.Sequence):
    def __init__(self, matrix_data_generator, fall_filepath, weigths_filepath, test=False):
        self.matrix_data_generator = matrix_data_generator
        self.fall_values = self.get_fall_dataframe(fall_filepath)
        self.test = test
        model = ResNetImp().ResNet(input_shape = (224, 224, 3), classes = 16)
        model.load_weights(weigths_filepath)
        model.layers.pop() # remove the last layer to extract features from it
        self.feature_extracting_model = keras.Model(model.input, model.layers[-1].output)
        self.feature_extracting_model._make_predict_function()
        self.graph = tf.get_default_graph()

    # returns the number of batches per epoch
    def __len__(self):
        return len(self.matrix_data_generator)

    # generates a batch of data, the index argument doesn't do anything
    def __getitem__(self, index=0):
        matricies, one_hot = self.matrix_data_generator.__getitem__()
        X = self.get_feature_batch(matricies)
        X = tf.keras.utils.normalize(X, axis=-1)
        y = self.get_fall_values(one_hot)
        return X, y

    # converts the batch of patient data matricies to a batch of features using the feature extractor
    def get_feature_batch(self, patient_data_batch):
        with self.graph.as_default():
            features = self.feature_extracting_model.predict(patient_data_batch)
        return (features)

    # returns a numpy array of shape (batch_size, 1) containing integers representing whether the patient fell: 1 indicates they fell, 0 indicates they did not fall
    def get_fall_values(self, one_hot_array):
        fall_values = []
        for i in range(one_hot_array.shape[0]):
            filepath = self.get_corresponding_patient_filepath(one_hot_array[i, ])
            patient_id = self.get_patient_ID(filepath)
            fell = self.get_patient_fell(patient_id)
            fall_values.append(fell)
        yData = np.array(fall_values)
        return (yData)

    # returns the patient filepath corresponding to each patient encoded as one-hot
    def get_corresponding_patient_filepath(self, one_hot):
        return (self.matrix_data_generator.preLoader.Get_patients()[np.where(one_hot == 1)[0][0]])

    # returns an integer representing whether the patient fell: 1 indicates they fell, 0 indicates they did not fall using patient ID as an arugment (eg: S0002)
    def get_patient_fell(self, patient_ID):
        row = self.fall_values.loc[patient_ID]
        fell = row['Fall']
        if (fell == 'y'):
            fall_value = [1, 0]#[1]
        else:
            fall_value = [0, 1]#[0]
        return (fall_value)

    # returns the ID of a patient from its filepath (eg: S0002)
    def get_patient_ID(self, patient_filepath):
        index = patient_filepath.index("S0")
        ID = patient_filepath[index: index+5]
        return (ID)

    # returns the pandas dataframe containg the information representing which patient's fell
    def get_fall_dataframe(self, filepath):
        df = pd.read_csv(filepath)
        df = df.set_index('patient')
        return (df)
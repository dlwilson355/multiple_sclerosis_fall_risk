import numpy as np
import os
import keras
import glob
import tensorflow as tf
from pathlib import Path
from data_reader import DataReader

class DataGenerator(keras.utils.Sequence):
    def __init__(self, folder, type, num, batchSize, steps, numFeatures):
        self.master_filepath = folder # the master filepath in which all of the data is located
        self.len = int(np.ceil(num / float(batchSize)))
        self.num = num
        # ignore and calc self
        #self.numFeatures = numFeatures 
        self.batchSize = batchSize
        self.steps = steps

        reader = DataReader(self.master_filepath)

        self.data  = reader.get_pickled_data("data.seg")
        if self.data != None:
            self.numFeatures = self.data[1].shape[0]
            print(self.data[0].shape)
            print(self.data[1].shape)
            return

        self.patients = glob.glob(self.master_filepath + "\\*\\")
        self.patients = reader.get_session1_lab_data_directory(self.patients)
        self.patients = reader.remove_patients_with_incomplete_data(self.patients)
        self.numFeatures = len(self.patients)
        xValues = []
        yValues = []
        for patient in self.patients:
            patient_data = reader.get_concatenated_patient_data(patient)
            one_hot = reader.convert_patient_to_one_hot(patient, self.patients)
            xValues.append(patient_data[:self.num*self.steps].values)
            yValues.append(one_hot)
        xData1 = np.array(xValues)
        xData = tf.keras.utils.normalize(xData1, axis=-1)
        yData = np.array(yValues)
        self.data = ((xData, yData))
        print(self.data[0].shape)
        print(self.data[1].shape)
        reader.save_pickle(self.data, "data.seg")
        return

    def GetData(self,index):
        return self.__getitem__(index)

    def __getitem__(self, index):
        x = np.ndarray((self.num,self.steps,1))
        y = np.ndarray((self.num,self.numFeatures))
        for i in range(self.num):
            l = self.data[1].shape[0]
            c = np.random.randint(0,l)
            p = index * self.num + i
            if p > self.data[0].shape[1]-self.steps:
                p = 0
            x[i] = self.data[0][c][p:p+self.steps]
            y[i] = self.data[1][c]
        return x, y

    def __len__(self):
        return self.len

    def on_epoch_end(self):
        return


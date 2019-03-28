import pandas as pd
import numpy as np
import os
import glob
import pickle
import keras
from pathlib import Path

class DataReader():
    def __init__(self, filepath):
        self.master_filepath = filepath # the master filepath in which all of the data is located
        file = Path(os.path.join(self.master_filepath, "data.txt"))
        if file.exists() == True:
            data = self.get_pickled_data()
            self.numFeatures = data[1].shape[1]
        else:
            self.numFeatures = self.calculateNumberOfFeatures()

    def calculateNumberOfFeatures(self):
        patients = next(os.walk(self.master_filepath))[1]
        index = 0 
        for patient in patients:
            test_directory = os.path.join(self.master_filepath, patient, "Session_1", "Home", "MC10", "anterior_thigh_right")
            csv_filepaths = [y for x in os.walk(test_directory) for y in glob.glob(os.path.join(x[0], '*.csv'))]
            if (len(csv_filepaths) > 0):
                index += 1
        return index

    def NumberOfFeatures(self):
        return self.numFeatures

    def get_y_values(self, onehot=True):
        data = []
        for index in range(self.numFeatures):
            data.append(index)
        if onehot:
            data = keras.utils.np_utils.to_categorical(data)
        else:
            data = np.array(data)
        return (data)

    # a function where you can pass the patient test and sensor data you want it will return the corresponding data
    def getData(self, patientNumber, testType, sensorType):
        pass

    # for now I am just loading a small sample of x values from one particular test
    def get_x_values(self):
        data = []
        patients = next(os.walk(self.master_filepath))[1]
        for patient in patients:
            # for now we just load sensor readings from one test for each patient
            test_directory = os.path.join(self.master_filepath, patient, "Session_1", "Home", "MC10", "anterior_thigh_right")
            csv_filepaths = [y for x in os.walk(test_directory) for y in glob.glob(os.path.join(x[0], '*.csv'))]
            if (len(csv_filepaths) > 0):
                df = pd.read_csv(csv_filepaths[0])
                # drop timestamp
                df = df.drop(df.columns[0], axis=1)
                data.append(np.resize(df.values, (2000, 3))) # cut the values to make them all the same dimensions, this also gives us less data to work with (for now)
                print("Loaded data from %s." % (csv_filepaths[0]))
            else:
                print("No csv files found in %s." % (test_directory))
        data = np.array(data)
        return (data)

    def get_x_samplesfromz(self):
        data = []
        patients = next(os.walk(self.master_filepath))[1]
        i = 0
        for patient in patients:
            # for now we just load sensor readings from one test for each patient
            test_directory = os.path.join(self.master_filepath, patient, "Session_1", "Home", "MC10", "anterior_thigh_right")
            csv_filepaths = [y for x in os.walk(test_directory) for y in glob.glob(os.path.join(x[0], '*.csv'))]
            if (len(csv_filepaths) > 0):
                df = pd.read_csv(csv_filepaths[0])
                # drop timestamp
                df = df.drop(df.columns[0], axis=1)
                df = df.drop(df.columns[0], axis=1)#x
                df = df.drop(df.columns[0], axis=1)#y
                # drop 9 out of 10 rows
                df = df.iloc[::100]
                data.append(np.resize(df, (10000, 1))) # cut the values to make them all the same dimensions, this also gives us less data to work with (for now)
                print("Loaded data from %s." % (csv_filepaths[0]))
            else:
                print("No csv files found in %s." % (test_directory))
        data = np.array(data)
        return (data)

    def get_data(self,onehot=True, samplesfromz=False):
        file = Path(os.path.join(self.master_filepath, "data.txt"))
        if file.exists() == True:
            return self.get_pickled_data()
        y = self.get_y_values(onehot)
        if samplesfromz:
            x = self.get_x_samplesfromz()
        else:
            x = self.get_x_values()
        data = ((x, y))
        self.save_pickle(data)
        return data

    def save_pickle(self, data):
        pickle.dump(data, open(os.path.join(self.master_filepath, "data.txt"), 'wb'))

    def get_pickled_data(self):
        return (pickle.load(open(os.path.join(self.master_filepath, "data.txt"), 'rb')))
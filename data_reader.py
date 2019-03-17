import pandas as pd
import numpy as np
import os
import glob
import pickle
import keras

class DataReader():
    def __init__(self, filepath):
        self.master_filepath = filepath # the master filepath in which all of the data is located

    def get_y_values(self):
        df = pd.read_csv(os.path.join(self.master_filepath, "SubjectInfo.csv"))
        df = df['Fall']
        # set values that don't make sense to not falling
        for i in range(len(df)):
            if (df[i] == 'y'):
                df[i] = 1
            else:
                df[i] = 0
        df = df.values
        df = keras.utils.np_utils.to_categorical(df)
        return (df)

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
                data.append(np.resize(df.values, (2000, 4))) # cut the values to make them all the same dimensions, this also gives us less data to work with (for now)
                print("Loaded data from %s." % (csv_filepaths[0]))
            else:
                print("No csv files found in %s." % (test_directory))
        data = np.array(data)
        return (data)

    def get_data(self):
        x = self.get_x_values()
        y = self.get_y_values()
        return ((x, y))

    def save_pickle(self, data):
        pickle.dump(data, open(os.path.join(self.master_filepath, "data.txt"), 'wb'))

    def get_pickled_data(self):
        return (pickle.load(open(os.path.join(self.master_filepath, "data.txt"), 'rb')))
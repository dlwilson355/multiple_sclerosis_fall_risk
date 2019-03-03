import pandas as pd
import numpy as np
import os

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
        df = df[:3]
        return (df)

    # for now I am just loading a small sample of x values from one particular test
    def get_x_values(self):
        data = []
        patients = next(os.walk(self.master_filepath))[1]
        for patient in patients[:3]:
            # for now we just load sensor readings from one test for each patient
            test_directory = os.path.join(self.master_filepath, patient, "Session_1", "Home", "MC10", "anterior_thigh_right")
            test_directory = os.path.join(test_directory, os.listdir(test_directory)[0])
            test_directory = os.path.join(test_directory, os.listdir(test_directory)[0], "accel.csv")
            df = pd.read_csv(test_directory)
            data.append(np.resize(df.values, (2000, 4))) # cut the values to make them all the same dimensions, this also gives us less data to work with (for now)
        data = np.array(data)
        return (data)

    def get_data(self):
        x = self.get_x_values()
        y = self.get_y_values()
        return ((x, y))
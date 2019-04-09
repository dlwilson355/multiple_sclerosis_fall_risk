import pandas as pd
import numpy as np
import os
import glob
import pickle
import keras
from pathlib import Path
import tensorflow as tf

class DataReader():
    def __init__(self, filepath):
        self.master_filepath = filepath # the master filepath in which all of the data is located
        data = self.get_pickled_data("data.txt")
        if data != None:
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

    # returns a full matrix of data for a patient corresponding to the timestamps
    def get_segmented_data(self, segment_size, segments_per_patient):
        data = self.get_pickled_data("data.seg")
        if data != None:
            return data
        xValues = []
        yValues = []
        patients = glob.glob(self.master_filepath + "\\*\\")
        print("all patients")
        print(patients)
        patients = self.get_session1_lab_data_directory(patients)
        print("lab data directories")
        print(patients)
        patients = self.remove_patients_with_incomplete_data(patients)
        print("patients with no missing data")
        print(patients)
        segment_window = int(segment_size /100)
        if segment_window == 0:
            segment_window = 1
        for patient in patients:
            patient_data = self.get_concatenated_patient_data(patient)
            one_hot = self.convert_patient_to_one_hot(patient, patients)
            for i in range(segments_per_patient):
                start = i*segment_window
                end = start + segment_size
                if end < len(patient_data):
                    xData1 = patient_data.iloc[start: end, ].values
                    xData = tf.keras.utils.normalize(xData1, axis=-1)
                    xValues.append(xData)
                    yValues.append(one_hot)
                else:
                    print(patient, 'out of range', end)
        xData = np.array(xValues)
        yData = np.array(yValues)
        data = ((xData, yData))
        self.save_pickle(data,"data.seg")
        return data

    def get_session1_lab_data_directory(self, patient_directories):
        directories = []
        for patient_directory in patient_directories:
            directories.append(os.path.join(patient_directory, "Session_1", "Lab", "MC10"))
        return (directories)

    def convert_patient_to_one_hot(self, patient, patients):
        one_hot = [0 for i in patients]
        for i in range(len(patients)):
            if (patient == patients[i]):
                one_hot[i] = 1
        return (one_hot)
    
    # now I realize that we probabaly don't need this but I will leave it for now
    def determine_if_patient_fell(self, patient_filepath):
        subject_info = pd.read_csv(os.path.join(self.master_filepath, "SubjectInfo.csv"))
        patient = os.path.basename(os.path.dirname(patient_filepath))
        print(subject_info.loc[subject_info['patient'] == patient])

    # returns a list of patients that no longer contains patients with incomplete data
    def remove_patients_with_incomplete_data(self, patient_list):
        trimmed_patient_list = []
        for patient in patient_list:
            if (self.has_complete_lab_data(patient)):
                trimmed_patient_list.append(patient)
        return (trimmed_patient_list)

    def has_complete_lab_data(self, patient_filepath):
        return (len([y for x in os.walk(patient_filepath) for y in glob.glob(os.path.join(x[0], '*.csv'))]) == 21)

    def get_concatenated_patient_data(self, patient_filepath):
        csv_filepaths = [y for x in os.walk(patient_filepath) for y in glob.glob(os.path.join(x[0], '*.csv'))]
        print("Loading data for %s" % (patient_filepath))
        dataframes = []
        for filepath in csv_filepaths:
            # make sure the file type is correct
            if ('accel' in filepath or 'gyro' in filepath or 'elec' in filepath):
                df = pd.read_csv(filepath)
                df = df.drop(df.columns[0], axis=1)
                dataframes.append(df)
        result = pd.concat(dataframes, axis=1)
        return (result)

    def get_x_samplesfromz(self):
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
        data = self.get_pickled_data("data.txt")
        if data != None:
            return data
        y = self.get_y_values(onehot)
        if samplesfromz:
            x = self.get_x_samplesfromz()
        else:
            x = self.get_x_values()
        data = ((x, y))
        self.save_pickle(data,"data.txt")
        return data

    def get_tsdata(self):
        data = self.get_pickled_data("data.ts")
        if data != None:
            return data
        tsfile = Path(os.path.join(self.master_filepath, "timestamps.csv"))
        if tsfile.exists() == False:
            print('Missing', tsfile)
            # force an exception
            return 0
        print('reading patient data using timestamps')
        tslist = pd.read_csv(tsfile)
        tsData = np.array(tslist)
        xValues = []
        yValues = []
        patients = next(os.walk(self.master_filepath))[1]
        for patient in patients:
            print(patient)
            test_directory = os.path.join(self.master_filepath, patient, "Session_1", "Home", "MC10", "anterior_thigh_right")
            csv_filepaths = [y for x in os.walk(test_directory) for y in glob.glob(os.path.join(x[0], '*.csv'))]
            if (len(csv_filepaths) > 0):
                one_hot = self.convert_patient_to_one_hot(patient, patients)
                df = pd.read_csv(csv_filepaths[0])
                first = df.iloc[0][0]
                stepsize = df.iloc[1][0] - first
                if stepsize > 1000:
                    factor = 1000000 / stepsize
                else:
                    factor = 1000 / stepsize
                df = df.drop(df.columns[0], axis=1)#ts
                print(patient, 'factor', factor)
                for ts in tsData:
                    if ts[0] == patient:
                        # read in 250 counts starting at row = ts[1] * 62.25
                        start = int(ts[1]*factor)
                        #print(ts, 'index',start,'len',len(df))
                        if start+250 < len(df): 
                            xValues.append(df[start:start+250].values)
                            yValues.append(one_hot)
                        else:
                            print(ts, 'index',start,'len',len(df))
        xData = np.array(xValues)
        yData = np.array(yValues)
        data = ((xData, yData))
        self.save_pickle(data, "data.ts")
        return data


    def save_pickle(self, data, filename):
        file = Path(os.path.join(self.master_filepath, filename))
        print('save',file)
        pickle.dump(data, open(file, 'wb'))

    def get_pickled_data(self, filename):
        file = Path(os.path.join(self.master_filepath, filename))
        if file.exists() == True:
            print('load',file)
            return (pickle.load(open(file, 'rb')))
        return None


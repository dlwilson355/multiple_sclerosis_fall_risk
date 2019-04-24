import numpy as np
import pandas as pd
import tensorflow as tf
import keras
import os
import glob
import random
import time
import datetime
from skimage.transform import resize
from skimage.color import gray2rgb

class MatrixPreLoader(object):
    def __init__(self, directory, num_patients_to_use = "ALL", print_loading_progress = False, debug = False):
        self.master_directory = directory
        self.print_loading_progress = print_loading_progress
        self.debug = debug
        self.patients = self.get_patient_list(num_patients_to_use)
        self.preloaded_concatenated_dataframes = self.preload_concatenated_dataframes()
        self.preloaded_activity_start_and_end_indicies = self.preload_activity_start_and_end_indicies()

    def Get_patients(self):
        return self.patients

    def Get_concatenated_dataframes(self):
        return self.preloaded_concatenated_dataframes

    def Get_activity_start_and_end_indicies(self):
        return self.preloaded_activity_start_and_end_indicies

    def Get_number_of_sensors(self):
        patient = self.Get_patients()[0]
        patient_data = self.Get_concatenated_dataframes()[patient]
        return patient_data.shape[1]

    def get_patient_list(self, num_patients):
        patients = glob.glob(os.path.join(self.master_directory, "*", ""))
        self.print_if_debug("Found %d patients, listed below." % (len(patients)))
        self.print_if_debug(patients)
        patients = self.get_session1_lab_data_directory(patients)
        self.print_if_debug("Corresponding lab data directories listed below.")
        self.print_if_debug(patients)
        patients = self.remove_patients_with_incomplete_data(patients)
        self.print_if_debug("Found %d patients with no missing data, listed below." % (len(patients)))
        self.print_if_debug(patients)
        if (not num_patients == "ALL"):
            self.print_if_debug("Removing %d patients." % (len(patients)-num_patients))
            patients = patients[0:num_patients]
            self.print_if_debug("Using %d patients for training.  Listed below." % (len(patients)))
            self.print_if_debug(patients)
        return (patients)

    # returns the test start and end timestamps from the corresponding patient directory
    def get_test_start_and_end_indicies(self, patient):
        starts = []
        ends = []
        activities = []
        annotations_filepath = os.path.join(patient, "annotations.csv")
        annotation_data = pd.read_csv(annotations_filepath)
        previous_activity_type = ""
        sample_patient_data = self.get_sample_data(patient)
        for row in range(annotation_data.shape[0]):
            activity_type = annotation_data.iloc[row, 2]
            if (not activity_type == previous_activity_type):
                activity_start_timestamp = annotation_data.iloc[row, 4]
                activity_start_timestamp = self.convert_to_milliseconds(activity_start_timestamp)
                activity_start_timestamp = pd.to_datetime(activity_start_timestamp, unit="ms")
                activity_end_timestamp = annotation_data.iloc[row, 5]
                activity_end_timestamp = self.convert_to_milliseconds(activity_end_timestamp)
                activity_end_timestamp = pd.to_datetime(activity_end_timestamp, unit="ms")
                activity_start_index = sample_patient_data.index.get_loc(activity_start_timestamp, method="nearest")
                activity_end_index = sample_patient_data.index.get_loc(activity_end_timestamp, method="nearest")
                starts.append(activity_start_index)
                ends.append(activity_end_index)
                activities.append(activity_type)
            previous_activity_type = activity_type
        return ((starts, ends,activities))

    # returns a list of directories from the session 1 lab MC10 test corresponding to the list of passed patient directories
    def get_session1_lab_data_directory(self, patient_directories):
        directories = []
        for patient_directory in patient_directories:
            directories.append(os.path.join(patient_directory, "Session_1", "Lab", "MC10"))
        return (directories)

    # returns a list of patients that no longer contains patients with incomplete data
    def remove_patients_with_incomplete_data(self, patient_list):
        trimmed_patient_list = []
        for patient in patient_list:
            if (self.has_complete_lab_data(patient)):
                trimmed_patient_list.append(patient)
        return (trimmed_patient_list)

    # returns the sensor readings from the lab data filepath as a dataframe with each column representing a type of reading from a particular sensor
    def get_concatenated_patient_data(self, patient_filepath):
        csv_filepaths = [y for x in os.walk(patient_filepath) for y in glob.glob(os.path.join(x[0], '*.csv'))]
        self.print_if_debug("Loading data for %s" % (patient_filepath))
        self.print_if_debug("Found: %s" % (csv_filepaths))
        dataframes = []
        for filepath in csv_filepaths:
            self.print_if_debug("Loading %s" % (filepath))
            if ('accel.csv' in filepath or 'gyro.csv' in filepath or 'elec.csv' in filepath): # make sure the file type is correct
                df = pd.read_csv(filepath)
                df = self.interpolate(df)
                #df = df.drop(df.columns[0], axis=1)
                dataframes.append(pd.DataFrame(df.values))
        result = pd.concat(dataframes, axis=1, ignore_index=True)
        return (result)

    # interpolates the dataframe such that all have measurements taken every 8 ms
    def interpolate(self, df):
        df.rename(columns={ df.columns[0]: "date" }, inplace=True)
        if (self.is_milliseconds(df.iloc[1, 0])):
            df['datetime'] = pd.to_datetime(df['date'], unit="ms")
        else:
            df['datetime'] = pd.to_datetime(df['date'], unit="us")
        df = df.set_index('datetime')
        df.drop(['date'], axis=1, inplace=True)
        df = df.resample('8ms').pad()
        return (df)

    # returns a boolean representing if the data from the lab tests at the passed directory contains a complete set of sensor readings
    def has_complete_lab_data(self, data_filepath):
        csv_filepaths = [y for x in os.walk(data_filepath) for y in glob.glob(os.path.join(x[0], '*.csv'))]
        valid_csv_filepaths = 0
        for csv_filepath in csv_filepaths: # only count files with valid names
            if ("accel.csv" in csv_filepath or "gyro.csv" in csv_filepath or "elec.csv" in csv_filepath or "annotations.csv" in csv_filepath):
                valid_csv_filepaths += 1
        return (valid_csv_filepaths == 21)

    # picks one of the patient's sensor data files and returns it as a dataframe
    def get_sample_data(self, patient_filepath):
        timestamp_filepath = [y for x in os.walk(patient_filepath) for y in glob.glob(os.path.join(x[0], 'accel.csv'))][0]
        self.print_if_debug("Checking %s" % (timestamp_filepath))
        df = pd.read_csv(timestamp_filepath)
        df = self.interpolate(df)
        return (df)

    def is_milliseconds(self, timestamp):
        return (not (len(str(timestamp)) >= 16))

    # if the passed timestamp is in microseconds, converts it to milliseconds
    def convert_to_milliseconds(self, timestamp):
        if (not self.is_milliseconds(timestamp)):
            return (int(timestamp/1000 + .5))
        return (timestamp)

    # returns a dicitonary with preloaded dataframes with all the sensor readings concatenated vertically for each patient to improve performance
    def preload_concatenated_dataframes(self):
        concatenated_dataframes = {}
        for patient in self.patients:
            concatenated_dataframes[patient] = self.get_concatenated_patient_data(patient)
        self.print_if_loading("Preloaded concatenated dataframes.")
        return (concatenated_dataframes)

    # returns a preloaded dictionary represending the indicies in the concatenated dataframe corresponding to the start and end times of each activity to improve performance
    def preload_activity_start_and_end_indicies(self):
        start_end_indicies = {}
        for patient in self.patients:
            start_end_indicies[patient] = self.get_test_start_and_end_indicies(patient)
        self.print_if_loading("Preloaded activity start and end indicies.")
        return (start_end_indicies)

    # prints out the statement if printing out loading progress (to track perfomance)
    def print_if_loading(self, string):
        if (self.print_loading_progress):
            print(string)

    # prints out the statement if dubugging
    def print_if_debug(self, string):
        if (self.debug):
            print(string)

class MatrixDataGenerator(keras.utils.Sequence):
    def __init__(self, preLoader, matrix_dimensions = "NONE", rgb = False, twoD = False, add_gaussian_noise = False, batch_size=32, grab_data_from = (0, .75), print_loading_progress = False, debug = False):
        self.matrix_dimensions = matrix_dimensions
        self.rgb = rgb
        self.twoD = twoD
        self.add_gaussian_noise = add_gaussian_noise
        self.batch_size = batch_size
        self.valid_selection_range = grab_data_from
        self.print_loading_progress = print_loading_progress
        self.debug = debug
        self.preLoader = preLoader
        self.len = int(np.floor(len(self.preLoader.Get_patients()) / self.batch_size)) + 1
        self.normalize = True

    def SetNormalize(self, normalize):
        self.normalize = normalize

    def __len__(self):
        'Denotes the number of batches per epoch'
        return self.len

    def __getitem__(self, index=0):
        'Generate one batch of data'
        patient_indexes = [random.randint(0, len(self.preLoader.Get_patients())-1) for i in range(self.batch_size)]

        # Find list of IDs
        patients_selected = [self.preLoader.Get_patients()[k] for k in patient_indexes]

        # Generate data
        X, y = self.data_generation(patients_selected)

        self.print_if_debug("X batch shape: %s" % (X.shape, ))
        self.print_if_debug("y batch shape: %s" % (y.shape, ))

        return X, y

    def data_generation(self, patients):
        xValues = []
        yValues = []

        self.print_if_loading("Generating batch of data with patients: %s" % (patients))
        for i, patient in enumerate(patients):
            self.print_if_loading("Loading patient %d of %d" % (i, len(patients)))
            self.print_if_debug("Generating data for %s" % (patient))
            xData = self.generate_patient_data(patient).values
            if self.normalize:
                xData = tf.keras.utils.normalize(xData, axis=-1)
            if (not self.matrix_dimensions == "NONE"):
                xData = resize(xData, self.matrix_dimensions)
            if (self.rgb):
                xData = gray2rgb(xData)
                self.print_if_debug("xData shape after rgb conversion: %s" % (xData.shape,))
            elif (self.twoD):
                xData = xData.reshape((xData.shape[0], xData.shape[1]))
                self.print_if_debug("xData shape after reshaping as 2D: %s" % (xData.shape,))
            else:
                xData = xData.reshape((xData.shape[0], xData.shape[1],1))
                self.print_if_debug("xData shape after reshaping as gray 3D: %s" % (xData.shape,))
            if (not self.add_gaussian_noise == None):
                self.print_if_debug("Adding gaussian noise.")
                noise = np.random.normal(0, self.add_gaussian_noise, xData.shape)
                xData = np.add(xData, noise)
            yData = self.convert_patient_to_one_hot(patient)
            self.print_if_debug("yData shape: %s" % (yData.shape,))
            xValues.append(xData)
            yValues.append(yData)
        xData = np.array(xValues)
        yData = np.array(yValues)
        return xData, yData

    def generate_patient_data(self, patient):
        self.print_if_debug("Getting matrix for %s" % (patient))
        (starts, ends,activities) = self.preLoader.Get_activity_start_and_end_indicies()[patient]
        patient_data = self.preLoader.Get_concatenated_dataframes()[patient]
        self.print_if_debug((starts, ends))
        self.print_if_debug(patient_data)
        dimension = patient_data.shape[1] * len(starts) # this dimension represents both the length and width of the dataframe
        dataframes = []
        final = pd.DataFrame()
        for i in range(len(starts)):
            start = starts[i]
            end = ends[i]
            self.print_if_debug("Window range from index %d to %d." % (start, end))
            difference = end - start
            valid_start_range = (int(start + difference * self.valid_selection_range[0]), int(start + difference * self.valid_selection_range[1] - dimension))
            if (valid_start_range[1] <= valid_start_range[0]): # need to find a better solution for this, this is necessary if the activity is too short to have a valid range
                valid_start_range = (valid_start_range[0], valid_start_range[0]+1)
            start_index = random.randint(valid_start_range[0], valid_start_range[1]-1)
            end_index = start_index + dimension
            self.print_if_debug("Selected window starts from index %d to %d." % (start_index, end_index))
            df = pd.DataFrame(patient_data.iloc[start_index:end_index, ].values)
            dataframes.append(df)
        result = pd.concat(dataframes, axis=1, ignore_index=True)
        self.print_if_debug("Resulting Matrix")
        self.print_if_debug(result)
        return (result)

    # takes the patient's directory and the directories of all the patients as arguments and returns the one hot encoding corresponding to that patient
    def convert_patient_to_one_hot(self, patient):
        one_hot = [0 for i in self.preLoader.Get_patients()]
        for i in range(len(self.preLoader.Get_patients())):
            if (patient == self.preLoader.Get_patients()[i]):
                one_hot[i] = 1
        return (np.array(one_hot))

    # prints out the statement if printing out loading progress (to track perfomance)
    def print_if_loading(self, string):
        if (self.print_loading_progress):
            print(string)

    # prints out the statement if dubugging
    def print_if_debug(self, string):
        if (self.debug):
            print(string)

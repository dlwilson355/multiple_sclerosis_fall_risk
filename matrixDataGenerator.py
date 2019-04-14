import numpy as np
import pandas as pd
import tensorflow as tf
import keras
import os
import glob
import random
from skimage.transform import resize
from skimage.color import gray2rgb

class MatrixDataGenerator(keras.utils.Sequence):
    def __init__(self, directory, matrix_dimensions = "NONE", rgb = False, batch_size=32, grab_data_from = (0, .75), debug = False):
        self.master_directory = directory
        self.matrix_dimensions = matrix_dimensions
        self.rgb = rgb
        self.batch_size = batch_size
        self.valid_selection_range = grab_data_from
        self.debug = debug
        self.patients = self.get_patient_list()
        self.len = int(np.floor(len(self.patients) / self.batch_size))

    def __len__(self):
        'Denotes the number of batches per epoch'
        return self.len

    def __getitem__(self, index=0):
        'Generate one batch of data'
        patient_indexes = [random.randint(0, len(self.patients)-1) for i in range(self.batch_size)]

        # Find list of IDs
        patients_selected = [self.patients[k] for k in patient_indexes]

        # Generate data
        X, y = self.data_generation(patients_selected)

        self.print_if_debug("X batch shape: %s" % (X.shape, ))
        self.print_if_debug("y batch shape: %s" % (y.shape, ))

        return X, y

    def data_generation(self, patients):
        xValues = []
        yValues = []

        self.print_if_debug("Generating batch of data with patients: %s" % (patients))
        for i, patient in enumerate(patients):
            self.print_if_debug("Generating data for %s" % (patient))
            xData = self.generate_patient_data(patient).values
            xData = tf.keras.utils.normalize(xData, axis=-1)
            if (not self.matrix_dimensions == "NONE"):
                xData = resize(xData, self.matrix_dimensions)
            if (self.rgb):
                xData = gray2rgb(xData)
                self.print_if_debug("xData shape after rgb conversion: %s" % (xData.shape,))
            else:
                xData = xData.reshape((xData.shape[0], xData.shape[1], 1))
                self.print_if_debug("xData shape after reshaping: %s" % (xData.shape,))
            one_hot = self.convert_patient_to_one_hot(patient)
            xValues.append(xData)
            yValues.append(one_hot)
        xData = np.array(xValues)
        yData = np.array(yValues)
        return xData, yData

    def generate_patient_data(self, patient):
        (starts, ends) = self.get_test_start_and_end(patient)
        patient_data = self.get_concatenated_patient_data(patient)
        dimension = patient_data.shape[1] * len(starts) # this dimension represents both the length and width of the dataframe
        dataframes = []
        final = pd.DataFrame()
        for i in range(len(starts)):
            start = self.get_corresponding_index(patient, starts[i])
            end = self.get_corresponding_index(patient, ends[i])
            difference = end - start
            valid_start_range = (int(start + difference * self.valid_selection_range[0]), int(start + difference * self.valid_selection_range[1] - dimension))
            if (valid_start_range[1] <= valid_start_range[0]): # need to find a better solution for this, this is necessary if the activity is too short to have a valid range
                valid_start_range = (valid_start_range[0], valid_start_range[0]+1)
            start_index = random.randint(valid_start_range[0], valid_start_range[1]-1)
            end_index = start_index + dimension
            df = pd.DataFrame(patient_data.iloc[start_index:end_index, ].values)
            dataframes.append(df)
        result = pd.concat(dataframes, axis=1, ignore_index=True)
        return (result)

    def get_patient_list(self):
        patients = glob.glob(self.master_directory + "\\*\\")
        self.print_if_debug("all patients")
        self.print_if_debug(patients)
        patients = self.get_session1_lab_data_directory(patients)
        self.print_if_debug("lab data directories")
        self.print_if_debug(patients)
        patients = self.remove_patients_with_incomplete_data(patients)
        self.print_if_debug("patients with no missing data")
        self.print_if_debug(patients)
        return (patients)

    # returns the test start and end timestamps from the corresponding directory
    def get_test_start_and_end(self, directory):
        starts = []
        ends = []
        annotations_filepath = os.path.join(directory, "annotations.csv")
        annotation_data = pd.read_csv(annotations_filepath)
        previous_activity_type = ""
        for row in range(annotation_data.shape[0]):
            activity_type = annotation_data.iloc[row, 2]
            if (not activity_type == previous_activity_type):
                activity_start_timestamp = annotation_data.iloc[row, 4]
                activity_end_timestamp = annotation_data.iloc[row, 5]
                starts.append(activity_start_timestamp)
                ends.append(activity_end_timestamp)
            previous_activity_type = activity_type
        return ((starts, ends))

    # returns a list of directories from the session 1 lab MC10 test corresponding to the list of passed patient directories
    def get_session1_lab_data_directory(self, patient_directories):
        directories = []
        for patient_directory in patient_directories:
            directories.append(os.path.join(patient_directory, "Session_1", "Lab", "MC10"))
        return (directories)

    # takes the patient's directory and the directories of all the patients as arguments and returns the one hot encoding corresponding to that patient
    def convert_patient_to_one_hot(self, patient):
        one_hot = [0 for i in self.patients]
        for i in range(len(self.patients)):
            if (patient == self.patients[i]):
                one_hot[i] = 1
        return (one_hot)

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
            # make sure the file type is correct
            if ('accel' in filepath or 'gyro' in filepath or 'elec' in filepath):
                df = pd.read_csv(filepath)
                df = df.drop(df.columns[0], axis=1)
                dataframes.append(df)
        result = pd.concat(dataframes, axis=1)
        return (result)

    # returns a boolean representing if the data from the lab tests at the passed directory contains a complete set of sensor readings
    def has_complete_lab_data(self, data_filepath):
        return (len([y for x in os.walk(data_filepath) for y in glob.glob(os.path.join(x[0], '*.csv'))]) == 21)

    # returns the index of the sensor data corresponding to the passed timestamp
    def get_corresponding_index(self, patient_filepath, timestamp):
        timestamp_filepath = [y for x in os.walk(patient_filepath) for y in glob.glob(os.path.join(x[0], 'accel.csv'))][0]
        self.print_if_debug("Checking %s" % (timestamp_filepath))
        df = pd.read_csv(timestamp_filepath)
        current_index = 0
        current_timestamp = self.convert_to_milliseconds(df.iloc[current_index, 0])
        while (abs(timestamp - current_timestamp) > 8):
            difference = timestamp - current_timestamp
            current_index += int(difference/8)
            current_timestamp = self.convert_to_milliseconds(df.iloc[current_index, 0])
        return (current_index)

    # if the passed timestamp is in microseconds, converts it to milliseconds
    def convert_to_milliseconds(self, timestamp):
        if (len(str(timestamp)) == 16):
            return (int(timestamp/1000 + .5))
        return (timestamp)

    # prints out the statement if dubugging
    def print_if_debug(self, string):
        if (self.debug):
            print(string)
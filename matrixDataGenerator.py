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
    def __init__(self, dataset_directory, num_patients_to_use = "ALL", activity_types = "ALL", print_loading_progress = False):
        self.master_directory = dataset_directory
        self.print_loading_progress = print_loading_progress
        self.patients = self.get_patient_list(num_patients_to_use)
        self.preloaded_patient_sensor_data = self.preload_patient_sensor_data(self.patients)
        self.preloaded_activity_start_and_end_indicies = self.preload_activity_start_and_end_indicies(self.patients, self.preloaded_patient_sensor_data, activity_types)
        self.dimension = self.calculate_default_image_dimension(self.patients, self.preloaded_patient_sensor_data, self.preloaded_activity_start_and_end_indicies)
        self.num_activities = self.calculate_number_activities(self.patients, self.preloaded_activity_start_and_end_indicies)

    def Get_patients(self):
        return self.patients

    def Get_patient_sensor_data(self):
        return self.preloaded_patient_sensor_data

    def Get_activity_start_and_end_indicies(self):
        return self.preloaded_activity_start_and_end_indicies

    def Get_dimension(self):
        return (self.dimension)

    def Get_num_activities(self):
        return (self.num_activities)

    def Get_number_of_sensors(self):
        patient = self.Get_patients()[0]
        patient_data = self.preloaded_patient_sensor_data[patient]
        return len(patient_data)

    def get_patient_list(self, num_patients):
        patients = glob.glob(os.path.join(self.master_directory, "*", ""))
        patients = self.get_session1_lab_data_directory(patients)
        patients = self.remove_patients_with_incomplete_data(patients)
        if (not num_patients == "ALL"):
            patients = patients[0:num_patients]
        return (patients)

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

    # returns a dictionary containing dictionaries of dataframes sensor data for each patient
    def preload_patient_sensor_data(self, patients):
        self.print_if_loading("Preloading patient sensor data.")
        dataframes = {}
        for patient_filepath in patients:
            self.print_if_loading("Loading data for %s." % (patient_filepath))
            csv_filepaths = [y for x in os.walk(patient_filepath) for y in glob.glob(os.path.join(x[0], '*.csv'))]
            sensor_readings = {}
            index = 0
            for filepath in csv_filepaths:
                if ('accel.csv' in filepath or 'gyro.csv' in filepath or 'elec.csv' in filepath): # make sure the file type is correct
                    df = pd.read_csv(filepath)
                    df = self.interpolate(df)
                    sensor_readings[index] = df
                    index += 1
            dataframes[patient_filepath] = sensor_readings
        return (dataframes)

    # returns a dictionary of patients containing dictionaries of different sensor numbers containing lists of tuples of activity start and end indicies
    def preload_activity_start_and_end_indicies(self, patients, sensor_data, activity_types):
        self.print_if_loading("Preloading activity start and end indicies.")
        all_indicies = {}
        for patient_filepath in patients:
            sensor_indicies = {}
            timestamps = self.get_start_and_end_timestamps(patient_filepath, activity_types)
            for i in sensor_data[patient_filepath]:
                sensor_indicies[i] = self.get_corresponding_indicies(timestamps, sensor_data[patient_filepath][i])
            all_indicies[patient_filepath] = sensor_indicies
        return (all_indicies)

    # returns a list of tuples of activity start and end timestamps for the patient
    def get_start_and_end_timestamps(self, patient, activity_types):
        timestamps = []
        annotations_filepath = os.path.join(patient, "annotations.csv")
        annotation_data = pd.read_csv(annotations_filepath)
        previous_activity_type = ""
        activity_types_found = []
        for row in range(annotation_data.shape[0]):
            activity_type = annotation_data.iloc[row, 2]
            if ((activity_type in activity_types) and (not activity_type == previous_activity_type)):
                activity_types_found.append(activity_type)
                activity_start_timestamp = annotation_data.iloc[row, 4]
                activity_start_timestamp = self.convert_to_milliseconds(activity_start_timestamp)
                activity_start_timestamp = pd.to_datetime(activity_start_timestamp, unit="ms")
                activity_end_timestamp = annotation_data.iloc[row, 5]
                activity_end_timestamp = self.convert_to_milliseconds(activity_end_timestamp)
                activity_end_timestamp = pd.to_datetime(activity_end_timestamp, unit="ms")
                timestamps.append((activity_start_timestamp, activity_end_timestamp))
            previous_activity_type = activity_type
        
        # warning message for if some of the activities passed by the user were not found in the dataset
        for activity in activity_types:
            if (not activity in activity_types_found):
                print("WARNING: Did not find activity named %s for %s." % (activity, patient))

        return (timestamps)

    # returns the indicies as a list of tuples of (start_index, end_index) corresponding to the timestamps for the sensor measurements
    def get_corresponding_indicies(self, timestamps, dataframe):
        indicies = []
        for timestamp in timestamps:
            start_index = dataframe.index.get_loc(timestamp[0], method="nearest")
            end_index = dataframe.index.get_loc(timestamp[1], method="nearest")
            indicies.append((start_index, end_index))
        return (indicies)

    # returns the "dimension" (both the width and height) of the data samples that will be generated
    def calculate_default_image_dimension(self, patients, sensor_data, indicies):
        activity_dimension = 0
        sample_patient_sensors = sensor_data[patients[0]]
        for i in sample_patient_sensors:
            activity_dimension += sample_patient_sensors[i].shape[1]
        dimension = activity_dimension * len(indicies[patients[0]][0])
        return (dimension)

    # returns the number of activities found in the dataset and provides a warning if it is not the same for each patient
    def calculate_number_activities(self, patients, indicies):
        num_activities = len(indicies[patients[0]][0])

        # warning message for it not all the patients have the same number of activities found in the dataset
        for patient in patients:
            if (not len(indicies[patient][0]) == num_activities):
                print("WARNING: Patient %s has %d activities." % (patient, len(indicies[patient][0])))

        return (num_activities)

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

    def is_milliseconds(self, timestamp):
        return (not (len(str(timestamp)) >= 16))

    # if the passed timestamp is in microseconds, converts it to milliseconds
    def convert_to_milliseconds(self, timestamp):
        if (not self.is_milliseconds(timestamp)):
            return (int(timestamp/1000 + .5))
        return (timestamp)

    # prints out the statement if printing out loading progress (to track perfomance)
    def print_if_loading(self, string):
        if (self.print_loading_progress):
            print(string)

class MatrixDataGenerator(keras.utils.Sequence):
    def __init__(self, preLoader, matrix_dimensions = "NONE", rgb = False, twoD = False, normalize = True, add_gaussian_noise = 0, zero_sensors = 0, batch_size=32, grab_data_from = (0, .75), overflow="AFTER", print_loading_progress = False):
        self.matrix_dimensions = matrix_dimensions
        self.rgb = rgb
        self.twoD = twoD
        self.gaussian_noise_variance = add_gaussian_noise
        self.num_sensors_to_zero = zero_sensors
        self.batch_size = batch_size
        self.valid_selection_range = grab_data_from
        self.overflow_direction = overflow
        self.print_loading_progress = print_loading_progress
        self.preLoader = preLoader
        self.len = int(np.floor(len(self.preLoader.Get_patients()) / self.batch_size)) + 1
        self.normalize = normalize

    # returns the number of batches per epoch
    def __len__(self):
        return self.len

    # generates a batch of data, the index argument doesn't do anything
    def __getitem__(self, index=0):
        patient_indexes = [random.randint(0, len(self.preLoader.Get_patients())-1) for i in range(self.batch_size)]
        patients_selected = [self.preLoader.Get_patients()[k] for k in patient_indexes]
        X, y = self.data_generation(patients_selected)

        return X, y

    # calls the function to create a data matrix for the each patient in patients and then transforms it appropriately with resizing, adding rgb channels, adding gaussing noise, reshaping, ect...
    def data_generation(self, patients):
        xValues = []
        yValues = []

        self.print_if_loading("Generating batch of data with patients: %s" % (patients))
        for i, patient in enumerate(patients):
            self.print_if_loading("Generating data sample for patient %d of %d" % (i, len(patients)))
            xData = self.generate_patient_matrix(patient).values
            if self.normalize:
                xData = tf.keras.utils.normalize(xData, axis=-1)
            if (not self.matrix_dimensions == "NONE"):
                xData = resize(xData, self.matrix_dimensions)
            if (self.rgb):
                xData = gray2rgb(xData)
            elif (self.twoD):
                xData = xData.reshape((xData.shape[0], xData.shape[1]))
            else:
                xData = xData.reshape((xData.shape[0], xData.shape[1],1))
            if (self.gaussian_noise_variance > 0):
                noise = np.random.normal(0, self.gaussian_noise_variance, xData.shape)
                xData = np.add(xData, noise)
            if (self.num_sensors_to_zero > 0):
                self.zero_columns(xData, self.num_sensors_to_zero)
            yData = self.convert_patient_to_one_hot(patient)

            # warning for if data sample contains 'nan' values
            if (np.isnan(xData).any().any()):
                print("WARNING: NAN found in generated data sample for patient %s." % (patient))

            xValues.append(xData)
            yValues.append(yData)
        xData = np.array(xValues)
        yData = np.array(yValues)
        return xData, yData

    # generates a data matrix for a patient by grabbing sequential sensor readings for each sensor for each activity and aligning them vertically to form a square shaped data matrix
    def generate_patient_matrix(self, patient):
        patient_data = self.preLoader.Get_patient_sensor_data()[patient]
        patient_indicies = self.preLoader.Get_activity_start_and_end_indicies()[patient]
        dimension = self.preLoader.Get_dimension()
        dataframes = []
        for activity_number in range(self.preLoader.Get_num_activities()):
            for sensor_number in patient_data:
                activity_start_index = patient_indicies[sensor_number][activity_number][0]
                activity_end_index = patient_indicies[sensor_number][activity_number][1]
                difference = activity_end_index - activity_start_index
                valid_start_index_range = (int(activity_start_index + difference * self.valid_selection_range[0]), int(activity_start_index + difference * self.valid_selection_range[1] - dimension))
                if (valid_start_index_range[1] <= valid_start_index_range[0]): # guarantees not overlap between training and validation data
                    
                    # Warning for if the activity window is too small such that there is potential overflow outside the specified bounds the data generator is supposed to grab data from.
                    print("WARNING: Overlap detected.  Overflowing selection window to prevent overlap.")

                    if (self.overflow_direction == "BEFORE"):
                        selected_start_index = int(activity_start_index + difference * self.valid_selection_range[1] - dimension)
                    else:
                        selected_start_index = int(activity_start_index + difference * self.valid_selection_range[0])
                else:
                    selected_start_index = random.randint(valid_start_index_range[0], valid_start_index_range[1]-1)
                selected_end_index = selected_start_index + dimension
                df = pd.DataFrame(patient_data[sensor_number].iloc[selected_start_index:selected_end_index, ].values)
                dataframes.append(df)
        result = pd.concat(dataframes, axis=1, ignore_index=True)
        return (result)

    # takes a data matrix (represented as a numpy array) as input and randomly zeros out the specified number of columns
    def zero_columns(self, matrix, num_columns):
        for column in range(num_columns):
            column_index = np.random.randint(matrix.shape[1])
            matrix[:, column_index] = np.zeros(matrix.shape[1:])
        return (matrix)

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
import numpy as np
import pandas as pd
import tensorflow as tf
import keras
import os
import glob
import random
from pathlib import Path
import pickle
from skimage.transform import resize
from skimage.color import gray2rgb

class PreLoader(object):
    def __init__(self, directory):
        self.master_directory = directory
        self.max_rows = 59000
        recname = "records.rec"
        dataname = "data.rec"
        self.records = self.get_pickled_data(recname)
        self.data = self.get_pickled_data(dataname)
        if self.records == None or self.data == None:
            self.generate()
            self.save_pickle(self.data, dataname)
            self.save_pickle(self.records, recname)
        else:
            self.patients = []
            patient = ''
            for rec in self.records:
                if patient != rec[0]:
                    patient = rec[0]
                    self.patients.append(patient)

    def get_data(self):
        return self.data

    def get_records(self):
        return self.records

    def get_patients(self):
        return self.patients

    def get_number_of_patients(self):
        return len(self.patients)

    def generate(self):
        records = []
        data = []
        self.patients = []
        # for all patients
        patients = glob.glob(os.path.join(self.master_directory, "*", ""))
        patients = self.get_session1_lab_data_directory(patients)
        for patient in patients:
            added = False
            patient_name = self.get_patient_name(patient)
            # ignore known incomplete patients
            if patient_name in ('S0018'):
                continue
            annotation_data = self.load_annotation(patient)
            # for all sensors
            csv_filepaths = [y for x in os.walk(patient) for y in glob.glob(os.path.join(x[0], '*.csv'))]
            for csv_filepath in csv_filepaths: 
                #if ("accel.csv" in csv_filepath or "gyro.csv" in csv_filepath or "elec.csv" in csv_filepath):
                if ("elec.csv" in csv_filepath):
                    sensor_name = self.get_sensor_name(csv_filepath)
                    # ignore known problem sensors
                    #if (sensor_name in ('dorsal_foot_left - accel.csv','dorsal_foot_left - gyro.csv','proximal_lateral_shank_left - accel.csv','proximal_lateral_shank_left - gyro.csv','sacrum - accel.csv','sacrum - gyro.csv','tibialis_anterior_right - accel.csv','tibialis_anterior_right - elec.csv')):
                    #    continue
                    # for all activities
                    df = pd.read_csv(csv_filepath)
                    previous_activity_type = ""
                    for row in range(annotation_data.shape[0]):
                        activity_type = annotation_data.iloc[row, 2]
                        # using only the Balance activities
                        if ('Balance' not in activity_type):
                            continue
                        if (not activity_type == previous_activity_type):
                            activity_start_timestamp = annotation_data.iloc[row, 4]
                            activity_end_timestamp = annotation_data.iloc[row, 5]
                            activity_start_index,activity_end_index,frequency = self.find_activity(df,activity_start_timestamp,activity_end_timestamp)
                            print('%s, %s, %s, %d, %d' %(patient_name, sensor_name, activity_type, activity_start_index, activity_end_index))
                            if activity_start_index <= 0:
                                print('bad record')
                                continue
                            record = (patient_name, sensor_name, activity_type)
                            records.append(record)
                            if frequency == 1:
                                step = 2
                            else:
                                step = 1
                            end = activity_start_index+self.max_rows*step
                            if end > activity_end_index:
                                print('bad record')
                                continue
                            data.append(df[activity_start_index: end:step].values)
                            if added == False:
                                added = True
                                self.patients.append(patient_name)
                        previous_activity_type = activity_type

        self.data = data
        self.records = records


    def get_session1_lab_data_directory(self, patient_directories):
        directories = []
        for patient_directory in patient_directories:
            directories.append(os.path.join(patient_directory, "Session_1", "Lab", "MC10"))
        return (directories)

    def load_annotation(self, patient):
        annotations_filepath = os.path.join(patient, "annotations.csv")
        print('----------------------------------------------------')
        print(annotations_filepath)
        return pd.read_csv(annotations_filepath)

    def get_patient_name(self, folder):
        p = folder[4:].find("\\") + 1
        return folder[4+p:4+p+5]

    def get_sensor_name(self, csv_filepath):
        i = csv_filepath.find('MC10\\')
        temp = csv_filepath[i+5:]
        i1 = temp.find('\\')
        start = temp[0:i1]
        e = temp.rfind('\\')
        return start + ' - ' + temp[e+1:]

    def find_activity(self, df, activity_start_timestamp, activity_end_timestamp):
        activity_start_index = -1
        activity_end_index = -1
        frequency = -1
        for col in df.columns:
            if col == 'Timestamp (ms)':
                ivals = df[df[col]-activity_start_timestamp > 0].index.values
                if ivals.shape[0] > 0:
                    activity_start_index = ivals.astype(int)[0]
                ivals = df[df[col]-activity_end_timestamp > 0].index.values
                if ivals.shape[0] > 0:
                    activity_end_index = ivals.astype(int)[0]
                frequency = df[col].values[1]-df[col].values[0]
                break
            elif col == 'Timestamp (microseconds)':
                activity_start_timestamp = activity_start_timestamp*1000
                activity_end_timestamp = activity_end_timestamp*1000
                ivals = df[df[col]-activity_start_timestamp > 0].index.values
                if ivals.shape[0] > 0:
                    activity_start_index = ivals.astype(int)[0]
                ivals = df[df[col]-activity_end_timestamp > 0].index.values
                if ivals.shape[0] > 0:
                    activity_end_index = ivals.astype(int)[0]
                frequency = int((df[col].values[1]-df[col].values[0]+500)/1000)
                break
        return (activity_start_index,activity_end_index,frequency)

    def save_pickle(self, data, filename):
        file = Path(os.path.join(self.master_directory, filename))
        print('save',file)
        pickle.dump(data, open(file, 'wb'))

    def get_pickled_data(self, filename):
        file = Path(os.path.join(self.master_directory, filename))
        if file.exists() == True:
            print('load',file)
            return (pickle.load(open(file, 'rb')))
        return None

class DataGenerator(keras.utils.Sequence):
    def __init__(self, preLoader, matrix_dimensions = "NONE", rgb = False, twoD = False, batch_size=32, grab_data_from = (0, .75)):
        self.matrix_dimensions = matrix_dimensions
        self.rgb = rgb
        self.twoD = twoD
        self.batch_size = batch_size
        self.valid_selection_range = grab_data_from
        self.preLoader = preLoader
        if self.twoD:
            if (self.matrix_dimensions == "NONE"):
                # number of observations, steps in sequence
                self.matrix_dimensions = (320,800)
            self.len = int(self.batch_size / self.preLoader.get_number_of_patients())
        else:
            if (self.matrix_dimensions == "NONE"):
                # number of observations, steps in sequence
                self.matrix_dimensions = (224,224)
            self.len = int(np.floor(self.preLoader.get_number_of_patients() / self.batch_size)) + 1

    def __len__(self):
        'Denotes the number of batches per epoch'
        return self.len

    def __getitem__(self, index=0):
        if self.twoD:
            xData = np.ndarray((self.batch_size,self.matrix_dimensions[0],self.matrix_dimensions[1]))
        elif self.rgb:
            xData = np.ndarray((self.batch_size,self.matrix_dimensions[0],self.matrix_dimensions[1], 3))
        else:
            xData = np.ndarray((self.batch_size,self.matrix_dimensions[0],self.matrix_dimensions[1], 1))
        yData = np.ndarray((self.batch_size,self.preLoader.get_number_of_patients()))


        data_set = self.preLoader.get_data()
        records = self.preLoader.get_records()
        len_data_set = len(data_set)
        num_rows = len(data_set[0])

        start = int(self.valid_selection_range[0] * num_rows)
        end = int(self.valid_selection_range[1] * num_rows)
        length = xData.shape[2]
        if start+length >= end:
            length = end-start-1

        for i in range(self.batch_size):
            j = random.randint(0,len_data_set-1)
            s = random.randint(start,end-length-1)
            #s = int((start+i+index) % num_rows)
            e = s + length
            data = data_set[j][s:e]
            #print(records[j],start,end,s,e)
            if self.twoD:
                x = data.reshape((xData.shape[1],xData.shape[2]))
            else:
                x = resize(data,(xData.shape[1],xData.shape[2],xData.shape[3]))
            xData[i] = tf.keras.utils.normalize(x, axis=-1)
            if np.isnan(xData[i]).any():
                if np.isnan(x).any():
                    print('both NAN')
                else:
                    print('Norm NAN')
            #xData[i] = x
            yData[i] = self.convert_patient_to_one_hot(records[j][0])
        return xData, yData

    def convert_patient_to_one_hot(self, patient):
        one_hot = np.zeros(self.preLoader.get_number_of_patients())
        for i in range(len(self.preLoader.get_patients())):
            if (patient == self.preLoader.get_patients()[i]):
                one_hot[i] = 1
                break
        return one_hot

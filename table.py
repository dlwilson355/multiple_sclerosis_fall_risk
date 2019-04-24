import numpy as np
import pandas as pd
from pathlib import Path
import os
import glob

class DataTable(object):
    def __init__(self, directory):
        self.master_directory = directory
        outputfile = 'table.csv'
        file = Path(outputfile)
        if file.exists():
            self.f = open(outputfile, 'a')
        else:
            self.f = open(outputfile, 'w+')
            self.WriteLabels()

    def run(self):
        patients = glob.glob(os.path.join(self.master_directory, "*", ""))
        patients = self.get_session1_lab_data_directory(patients)
        # for all patients
        for patient in patients:
            patient_name = self.get_patient_name(patient)
            annotation_data = self.load_annotation(patient)
            # for all sensors
            csv_filepaths = [y for x in os.walk(patient) for y in glob.glob(os.path.join(x[0], '*.csv'))]
            for csv_filepath in csv_filepaths: 
                if ("accel.csv" in csv_filepath or "gyro.csv" in csv_filepath or "elec.csv" in csv_filepath):
                    sensor_name = self.get_sensor_name(csv_filepath)
                    # for all activities
                    df = pd.read_csv(csv_filepath)
                    previous_activity_type = ""
                    for row in range(annotation_data.shape[0]):
                        activity_type = annotation_data.iloc[row, 2]
                        if (not activity_type == previous_activity_type):
                            activity_start_timestamp = annotation_data.iloc[row, 4]
                            activity_end_timestamp = annotation_data.iloc[row, 5]
                            activity_start_index,activity_end_index = self.find_activity(df,activity_start_timestamp,activity_end_timestamp)
                            self.WriteRecord(df,patient_name, sensor_name, activity_type, activity_start_timestamp, activity_end_timestamp,activity_start_index,activity_end_index)
                        previous_activity_type = activity_type

    def get_session1_lab_data_directory(self, patient_directories):
        directories = []
        for patient_directory in patient_directories:
            directories.append(os.path.join(patient_directory, "Session_1", "Lab", "MC10"))
        return (directories)

    def load_annotation(self, patient):
        annotations_filepath = os.path.join(patient, "annotations.csv")
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
        for col in df.columns:
            if col == 'Timestamp (ms)':
                ivals = df[df[col]-activity_start_timestamp > 0].index.values
                if ivals.shape[0] > 0:
                    activity_start_index = ivals.astype(int)[0]
                ivals = df[df[col]-activity_end_timestamp > 0].index.values
                if ivals.shape[0] > 0:
                    activity_end_index = ivals.astype(int)[0]
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
                break
        return (activity_start_index,activity_end_index)

    def WriteLabels(self):
        self.f.write('patient_name')
        self.f.write(',')
        self.f.write('sensor_name')
        self.f.write(',')
        self.f.write('activity_type')
        self.f.write(',')
        self.f.write('activity_start_timestamp')
        self.f.write(',')
        self.f.write('activity_end_timestamp')
        self.f.write(',')
        self.f.write('activity_start_index')
        self.f.write(',')
        self.f.write('activity_end_index')
        self.f.write(',')
        self.f.write('steps')
        self.f.write(',')
        self.f.write('min1')
        self.f.write(',')
        self.f.write('max1')
        self.f.write(',')
        self.f.write('mean1')
        self.f.write(',')
        self.f.write('stdev1')
        self.f.write(',')
        self.f.write('min2')
        self.f.write(',')
        self.f.write('max2')
        self.f.write(',')
        self.f.write('mean2')
        self.f.write(',')
        self.f.write('stdev2')
        self.f.write(',')
        self.f.write('min3')
        self.f.write(',')
        self.f.write('max3')
        self.f.write(',')
        self.f.write('mean3')
        self.f.write(',')
        self.f.write('stdev3')
        self.f.write('\n')
        self.f.flush()

    def WriteRecord(self, df, patient_name, sensor_name, activity_type, activity_start_timestamp, activity_end_timestamp,activity_start_index,activity_end_index):
        print(patient_name, sensor_name, activity_type, activity_start_timestamp, activity_end_timestamp,activity_start_index,activity_end_index)
        self.f.write(patient_name)
        self.f.write(',')
        self.f.write(sensor_name)
        self.f.write(',')
        self.f.write(activity_type)
        self.f.write(',')
        self.f.write(str(activity_start_timestamp))
        self.f.write(',')
        self.f.write(str(activity_end_timestamp))
        self.f.write(',')
        self.f.write(str(activity_start_index))
        self.f.write(',')
        self.f.write(str(activity_end_index))
        self.f.write(',')
        self.f.write(str(activity_end_index-activity_start_index))
        self.f.write(',')
        df = df.drop(df.columns[0], axis=1)
        for c, col in enumerate(df.columns):
            self.f.write(str(min(df[col])))
            self.f.write(',')
            self.f.write(str(max(df[col])))
            self.f.write(',')
            self.f.write(str(np.mean(df[col])))
            self.f.write(',')
            self.f.write(str(np.std(df[col])))
            self.f.write(',')
        while  c < 2:
            self.f.write(',')
            self.f.write(',')
            self.f.write(',')
            c += 1
        self.f.write('\n')
        self.f.flush()

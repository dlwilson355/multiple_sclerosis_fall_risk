import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import os
from PIL import Image
from matrixDataGenerator import MatrixDataGenerator, MatrixPreLoader

class Visualize(object):
    def __init__(self, directory, num_patients,asplt):
        self.asplt = asplt
        if self.asplt:
            self.subfolder = 'plt' 
        else:
            self.subfolder = 'img' 
        file = Path(self.subfolder)
        if file.exists() == False:
            os.mkdir(self.subfolder)
        activities_to_load = ["30s Chair Stand Test", "Tandem Balance Assessment", "Standing Balance Assessment", "Standing Balance Eyes Closed", "ADL: Normal Walking", "ADL: Normal Standing", "ADL: Normal Sitting", "ADL: Slouch sitting", "ADL: Lying on back", "ADL: Lying on left side", "ADL: Lying on right side"]
        self.preLoader = MatrixPreLoader(directory = directory, num_patients_to_use = num_patients, activity_types = activities_to_load, print_loading_progress = True)
        self.patients = self.preLoader.Get_patients()
        self.train_gen =  MatrixDataGenerator(self.preLoader,rgb = False, twoD = True, batch_size = 1, add_gaussian_noise = .01, overflow = "BEFORE", grab_data_from = (0,0.7), print_loading_progress = False)
        self.test_gen =  MatrixDataGenerator(self.preLoader,rgb = False, twoD = True, batch_size = 1, add_gaussian_noise = .01, overflow = "BEFORE", grab_data_from = (0.7,1), print_loading_progress = False)
        return

    def PatientName(self, folder):
        p = folder[4:].find("\\") + 1
        return folder[4+p:4+p+5]

    def ActivityName(self, activity,start,end):
        act_name = activity.replace(':', '-') + ' from ' + str(start) + ' to ' + str(end)
        return act_name

    def PatientFromY(self,y):
        i = 0
        max = 0
        index = 0
        for n in y:
            if n > max:
                index = i
                max = n
            #print(n)
            i += 1
        return index

    def GetFileName(self,patient_name,act_name, index):
        if index >= 0:
            name = ' item'+ str(index)
        else:
            name = ''
        filename = self.subfolder + '\\' + patient_name + ' ' + act_name + name + '.png'
        return filename

    def AllPatients(self):
        for patient in self.patients:
            patient_data = self.preLoader.Get_patient_sensor_data()[patient]
            patient_indicies = self.preLoader.Get_activity_start_and_end_indicies()[patient]
            patient_name = self.PatientName(patient)
            for activity_number in range(self.preLoader.Get_num_activities()):
                for sensor_number in patient_data:
                    start = patient_indicies[sensor_number][activity_number][0]
                    end = patient_indicies[sensor_number][activity_number][1]
                    df = pd.DataFrame(patient_data[sensor_number].iloc[start:end, ].values)
                    act_name = self.ActivityName(patient_indicies[sensor_number][activity_number][2],start,end)
                    filename = self.GetFileName(patient_name,act_name, -1)
                    if self.asplt:
                        plt.plot(df)
                        plt.ylabel(act_name)
                        plt.xlabel(patient_name)
                        plt.savefig(filename)
                    else:
                        img = Image.fromarray(df.values, 'L')
                        img.save(filename)
                    print(filename)

    def Items(self,gen,label):
        for index in range(10):
            x,y = gen.__getitem__(index)
            p = self.PatientFromY(y[0])
            patient_name = self.PatientName(self.patients[p])
            filename = self.GetFileName(patient_name,label, index)
            if self.asplt:
                plt.ylabel(label)
                plt.xlabel(patient_name)
                plt.plot(x[0])
                plt.savefig(filename)
            else:
                img = Image.fromarray(x[0], 'L')
                img.save(filename)
            print(filename)

    def run(self):
        self.AllPatients()
        self.Items(self.train_gen, 'train')
        self.Items(self.test_gen, 'test')

        return


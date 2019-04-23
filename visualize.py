import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import os
from matrixDataGenerator import MatrixDataGenerator, MatrixPreLoader

class Visualize(object):
    def __init__(self, directory, length, rgb, twoD,num_patients):
        file = Path('plt')
        if file.exists() == False:
            os.mkdir('plt')
        self.preLoader = MatrixPreLoader(directory = directory, num_patients_to_use = num_patients, print_loading_progress = True, debug = True)
        self.num_sequences = self.preLoader.Get_number_of_sensors()
        self.gen =  MatrixDataGenerator(self.preLoader,rgb = rgb, twoD = twoD, batch_size = 1, grab_data_from = (0,1), print_loading_progress = False, debug = False)
        return

    def PatientName(self, folder):
        p = folder[4:].find("\\") + 1
        return folder[4+p:4+p+5]

    def ActivityName(self, activity):
        act_name = activity.replace(':', '-')
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

    def run(self):
        patients = self.preLoader.Get_patients()

        for patient in patients:
            patient_data = self.preLoader.Get_concatenated_dataframes()[patient]
            (starts, ends, activities) = self.preLoader.Get_activity_start_and_end_indicies()[patient]
            patient_name = self.PatientName(patient)
            for i in range(len(starts)):
                start = starts[i]
                end = ends[i]
                df = pd.DataFrame(patient_data.iloc[start:end, ].values)
                plt.plot(df)
                act_name = self.ActivityName(activities[i])
                plt.ylabel(act_name)
                plt.xlabel(patient_name)
                filename = 'plt\\' + patient_name + ' ' + act_name + '.png'
                plt.savefig(filename)
                print(filename)

        for index in range(100):
            x,y = self.gen.__getitem__(index)
            p = self.PatientFromY(y[0])
            patient_name = self.PatientName(patients[p])
            activity = self.gen.Get_last_activity()
            act_name = self.ActivityName(activity)
            plt.ylabel(act_name)
            plt.xlabel(patient_name)
            plt.plot(x[0])
            filename = 'plt\\item'+ str(index) + ' ' + patient_name + ' ' + act_name + '.png'
            plt.savefig(filename)
            print(filename)

        return


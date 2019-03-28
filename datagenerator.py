import numpy as np
import os
import keras
from data_reader import DataReader

class DataGenerator(keras.utils.Sequence):
    def __init__(self, folder, type, num, batchSize, steps):
        self.master_filepath = folder # the master filepath in which all of the data is located
        self.len = int(np.ceil(num / float(batchSize)))
        self.num = num
        self.batchSize = batchSize
        self.steps = steps
        reader = DataReader(self.master_filepath)
        self.data = reader.get_data(False,True)
        return

    def GetData(self,index):
        return self.__getitem__(index)

    def __getitem__(self, index):
        x = np.ndarray((self.num,self.steps,1))
        y = np.ndarray((self.num,1))
        for i in range(self.num):
            l = self.data[1].shape[0]
            c = np.random.randint(0,l)
            p = index * self.num + i
            if p > self.data[0].shape[1]-self.steps:
                p = 0
            x[i] = self.data[0][c][p:p+self.steps]
            y[i] = self.data[1][c]
        return x, y

    def __len__(self):
        return self.len

    def on_epoch_end(self):
        return

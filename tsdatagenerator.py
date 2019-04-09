import numpy as np
import os
import keras
from data_reader import DataReader

class DataGenerator(keras.utils.Sequence):
    def __init__(self, folder, type, num, batchSize, steps, numFeatures):
        self.master_filepath = folder # the master filepath in which all of the data is located
        self.len = int(np.ceil(num / float(batchSize)))
        self.num = num
        self.numFeatures = numFeatures
        self.batchSize = batchSize
        self.steps = steps
        self.type = type
        reader = DataReader(self.master_filepath)
        self.data = reader.get_tsdata()
        self.pos = 1
        if 'train' == type:
            self.low = 0
            self.high = 4
        elif 'validate' == type:
            self.low = 5
            self.high = 8
        else:
            self.low = 9
            self.high = 9
        return

    def GetData(self,index):
        return self.__getitem__(index)

    def getY(self, y):
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

    def __getitem__(self, index):
        x = np.ndarray((self.num,self.steps,3))
        y = np.ndarray((self.num,self.numFeatures))
        j = 0
        for i in range(self.num):
            record = (self.pos * index) + self.low + i - j
            mod = record % 10
            if mod > self.high or mod < self.low:
                self.pos += 10
                j = i
                record = (self.pos * (index+1)) + self.low
            if record > self.data[0].shape[0]:
                self.pos = 0
                record = self.low + i - j
            pos = index * self.num + i
            if pos > self.data[0].shape[1]-self.steps:
                pos = 0
            x[i] = self.data[0][record][pos:pos+self.steps]
            y[i] = self.data[1][record]
            #print(self.type,'record',record,'pos',pos,'y',self.getY(y[i]))
        return x, y

    def __len__(self):
        return self.len

    def on_epoch_end(self):
        return

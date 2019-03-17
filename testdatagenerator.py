import numpy as np
import os
import keras
import cv2

class TestDataGenerator(keras.utils.Sequence):
    def __init__(self, num, batchSize, steps):
        self.len = int(np.ceil(num / float(batchSize)))
        self.num = num
        self.batchSize = batchSize
        return

    def GetData(self):
        return self.__getitem__(0)

    def __getitem__(self, index):
        x = np.ndarray((self.num,self.steps,1))
        y = np.ndarray((self.num,1))
        for i in range(self.num):
            c = np.random.randint(0,2)
            if c > 0:
                t = np.linspace(0.0,0.8,num=self.steps)
                for s in range(self.steps):
                    x[i][s] = t[s]
                y[i] = 0.9
            else:
                v = 0
                t = [0,1]*int(self.steps/2)
                for s in range(self.steps):
                    x[i][s] = t[s]
                y[i] = 0
        return x, y

    def __len__(self):
        return self.len

    def on_epoch_end(self):
        return

import numpy as np
import tensorflow as tf
import keras
from keras.applications.resnet50 import ResNet50

class ResNetImp(object):
    def __init__(self):
        return      

    def ResNet(self, input_shape=None, classes=1000):
        base_model = ResNet50(include_top=False, weights=None,  pooling=None, classes=classes,input_shape=input_shape)
        x = base_model.output
        x = keras.layers.GlobalAveragePooling2D()(x)
        x = keras.layers.Dense(1024, activation='relu')(x)
        predictions = keras.layers.Dense(classes, activation='softmax')(x)
        model = keras.models.Model(inputs=base_model.input, outputs=predictions)
        return model


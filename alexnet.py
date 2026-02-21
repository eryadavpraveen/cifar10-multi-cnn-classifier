import tensorflow as tf 
from tensorflow.keras import layers,models
import matplotlib.pyplot as plt
import  numpy as np
import os


def build_alexnet():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(96,(3,3) ,activation='relu',input_shape = (32,32,3)),   
        tf.keras.layers.MaxPooling2D((2,2)),
        tf.keras.layers.Conv2D(256,(3,3) , activation='relu',padding = 'same'),
        tf.keras.layers.MaxPooling2D((2,2)),
        tf.keras.layers.Conv2D(384,(3,3) , activation='relu',padding = 'same'),
        tf.keras.layers.Conv2D(384,(3,3) , activation='relu',padding = 'same'),
        tf.keras.layers.Conv2D(256,(3,3) , activation='relu',padding = 'same'),
        tf.keras.layers.MaxPooling2D((2,2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(4096,activation = 'relu'),
        tf.keras.layers.Dropout(.5),
        tf.keras.layers.Dense(4096,activation = 'relu'),
        tf.keras.layers.Dropout(.5),
        tf.keras.layers.Dense(10,activation = 'relu'),
         ])
    return model


'''
Stride tells:
How many pixels the filter moves each time during convolution.
Think of it as the step size of the filter.
'''


"""
actual alexnet model:-

def build_alexnet():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(96,(11,11), strides=(4,4) ,activation='relu',input_shape = (227,227,3)),
        tf.keras.layers.MaxPooling2D((3,3),strides=(2,2)),
        tf.keras.layers.Conv2D(256,(5,5) , activation='relu',padding = 'same'),
        tf.keras.layers.MaxPooling2D((3,3),strides=(2,2)),
        tf.keras.layers.Conv2D(384,(3,3) , activation='relu',padding = 'same'),
        tf.keras.layers.Conv2D(384,(3,3) , activation='relu',padding = 'same'),
        tf.keras.layers.Conv2D(256,(3,3) , activation='relu',padding = 'same'),
        tf.keras.layers.MaxPooling2D((3,3),strides=(2,2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(4096,activation = 'relu'),
        tf.keras.layers.Dropout(.5),
        tf.keras.layers.Dense(4096,activation = 'relu'),
        tf.keras.layers.Dropout(.5),
        tf.keras.layers.Dense(1000,activation = 'relu'),
         ])
    return model
"""
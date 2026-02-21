import tensorflow as tf 
from tensorflow.keras import layers,models
import matplotlib.pyplot as plt
import  numpy as np
import os
from alexnet import build_alexnet
from leenet import build_lenet
from vgg import build_vgg
from resnet import build_resnet

(x_train,y_train),(x_test,y_test) = tf.keras.datasets.cifar10.load_data()
x_train,x_test = x_train/255.0, x_test/255.0

if not os.path.exists("save_models"):
    os.makedirs('save_models')

def train_and_evaluate(model,model_name):
    model.compile(optimizer = 'adam',loss = 'sparse_categorical_crossentropy',metrics = ['accuracy'])
    checkpointing = tf.keras.callbacks.ModelCheckpoint(
        filepath = f"save_models/{model_name}.h5",monitor = 'val_accuracy',save_best_only =True,verbose = 1 )
    
    
    model.fit(x_train,y_train , epochs = 20 ,batch_size= 64,validation_split = .2 , verbose = 1,callbacks = [checkpointing] )
    
    
    model.evaluate(x_test,y_test,verbose = 1 )

    # Clear memory
    tf.keras.backend.clear_session()
    



if __name__ == "__main__":
    train_and_evaluate(build_alexnet(),'alex_net')
    train_and_evaluate(build_lenet(),'lenet')
    train_and_evaluate(build_vgg(),'vgg')
    train_and_evaluate(build_resnet(),'resnet')



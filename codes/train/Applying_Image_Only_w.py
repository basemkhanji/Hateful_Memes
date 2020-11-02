import pandas as pd
import numpy as np
from numpy import loadtxt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score 
from keras.models import Model , Sequential
import keras.layers
import tensorflow as tf
from keras.layers import Dense , Dropout, Flatten, GRU , SpatialDropout1D, LSTM,Activation, BatchNormalization
from keras.constraints import maxnorm
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras import layers
import joblib
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# in this code we try to mine the text for hints to a better score
df_train = pd.read_csv("train_proc_withtxtDNN.csv")
print(df_train)
datagen_full = ImageDataGenerator(rescale=1./255.)
bat_s = 42
Full_generator=datagen_full.flow_from_dataframe(
    dataframe= df_train,
    directory="../../data/",
    x_col="img",
    y_col="label",
    seed = 42,
    batch_size=bat_s,
    shuffle=False,
    class_mode='raw',
    target_size=(200,200) # one should optimize this on a differnt data sample, not obvious if it is a good starting point or not
)

from keras.models import model_from_json
#Load the model weigths form h5 file :
json_file = open('./modelimageonly.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("./modelimgeonly_weights.h5")
print("Loaded model from disk")
loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
score = loaded_model.evaluate_generator(generator=Full_generator, steps = Full_generator.n//bat_s )

print("Testing Loss: {:.4f}".format(score[0] ))
print("Testing Accuracy: {:.4f}".format(score[1]))

df_train['img_embd_DNN'] = loaded_model.predict_generator( Full_generator,steps= len(Full_generator))[:,0]
print(df_train)
df_train.to_csv('train_proc_withimgDNN_NEW_andtxtDNN_NEW.csv', encoding='utf-8', index=False)



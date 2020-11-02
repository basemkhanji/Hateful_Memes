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

'''
from PIL import Image
import statistics 
from statistics import mode 
list_width  = []
list_length = []
directory = '../../data/img/'
for filename in os.listdir(directory):
    if filename.endswith(".png"):
        im = Image.open(os.path.join(directory, filename))
        #print(im.format, im.size, im.mode)
        list_length.append(im.size[0])
        list_width.append(im.size[1])
    else:
        continue
print(max(list_length))
print(max(list_width))
print(min(list_length))
print(min(list_width))
print(mode(list_length))
print(mode(list_width))
print(statistics.mean(list_length))
print(statistics.mean(list_width))
from collections import Counter
print(Counter(list_length).most_common(1))
print(Counter(list_width).most_common(1))
'''
# in this code we try to mine the text for hints to a better score
df_train = pd.read_csv("../../proc_data/train_proc_withtxtDNN_NEW.csv")
print(df_train)
# Get the text labels and ids ... 
df_train_img = df_train[['id' , 'label' , 'img' ]]
Y_Labels     = df_train_img['label'] 
X_Features   = df_train_img['img']
print(X_Features.describe())
print(X_Features)
print(Y_Labels)
bat_s = 64
# get the images 
datagen = ImageDataGenerator(rescale=1./255., validation_split=0.4)
train_generator=datagen.flow_from_dataframe(
    dataframe= df_train_img,
    directory="../../data/",
    x_col="img",
    y_col="label",
    subset="training",
    batch_size=bat_s,
    seed=42,
    shuffle=True,
    class_mode="raw",
    target_size=(200,200) # one should optimize this on a differnt data sample, not obvious if it is a good starting point or not
)
print('=============================================')
print(train_generator)
print('=============================================')
test_generator=datagen.flow_from_dataframe(
    dataframe= df_train_img,
    directory="../../data/",
    x_col="img",
    y_col="label",
    subset="validation",
    batch_size=bat_s,
    seed=42,
    shuffle=True,
    #class_mode="categorical",
    class_mode="raw",
    target_size=(200,200) )

#default
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(200, 200 , 3) ))
#model.add(BatchNormalization()) # newly added
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())# this converts our 3D feature maps to 1D feature vectors
#model.add(BatchNormalization()) # newly add
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(1))
model.add(Activation('sigmoid'))
model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

model.fit_generator(generator= train_generator,
                    #steps_per_epoch = len(train_generator)//bat_s,
                    steps_per_epoch = 3200//bat_s,
                    epochs = 30 , # 30 , pixel: 200x200
                    validation_data = test_generator,
                    validation_steps= 3200//bat_s, #len(test_generator)//bat_s,
                    use_multiprocessing=True,
                    workers=3)
model_json = model.to_json()
with open("modelimageonly.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("modelimgeonly_weights.h5")
#model.save("modelimgeonly.h5")
print("Saved model to disk")

score = model.evaluate_generator(generator=test_generator )
print("Testing Loss: {:.4f}".format(score[0] ))
print("Testing Accuracy: {:.4f}".format(score[1]))

datagen_full = ImageDataGenerator(rescale=1./255.)
Full_generator=datagen.flow_from_dataframe(
    dataframe= df_train_img,
    directory="../../data/",
    x_col="img",
    batch_size=bat_s,
    seed=42,
    shuffle=False,
    class_mode="raw",
    target_size=(320,320) # one should optimize this on a differnt data sample, not obvious if it is a good starting point or not
)
df_train['img_embd_DNN'] = model.predict_generator( Full_generator , steps=len(Full_generator), verbose=1 )[:,0]
#print(df_train)
df_train.to_csv('train_proc_withimgDNNNEW_andtxtDNNNEW.csv', encoding='utf-8', index=False)



import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score 
from keras.models import Model
#import keras.layers
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential , Model
from keras.layers import SpatialDropout1D, Dense , GRU , SpatialDropout1D, LSTM , Dropout, Flatten, Activation, BatchNormalization,Conv1D, GlobalMaxPool1D , Bidirectional , merge, Input , concatenate
from keras import layers
from keras.constraints import maxnorm
from keras.layers.convolutional import Conv2D, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.sequence import pad_sequences
import os

def create_embedding_matrix(filepath, word_index, embedding_dim):
    vocab_size = len(word_index) + 1  # Adding again 1 because of reserved 0 index
    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    with open(filepath) as f:
        for line in f:
            word, *vector = line.split()
            if word in word_index:
                idx = word_index[word] 
                embedding_matrix[idx] = np.array(
                    vector, dtype=np.float32)[:embedding_dim]
    return embedding_matrix

df_train = pd.read_csv("../../proc_data/traindev.csv")[:100]
# Get the text labels and ids ... 
#X_train, X_test , Y_train , Y_test = train_test_split( df_train , df_train['label']  , test_size = 0.3 , random_state = 7 )

X_Text_train    = df_train['text']
X_Images_Train  = df_train['img']
Y_train         = df_train['label']
print(X_Images_Train)
#1- Make Text data
maxlen = 25
embedding_dim = 20
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(X_Text_train)
X_text_train = tokenizer.texts_to_sequences(X_Text_train)
X_text_train = pad_sequences(X_text_train, padding='post', maxlen=maxlen)
vocab_size = len(tokenizer.word_index) + 1

#2- make Image data 
bat_s = 64
# get the images 
datagen = ImageDataGenerator(rescale=1./255.)
X_image_train=datagen.flow_from_dataframe(
    dataframe= df_train,
    directory="../../data/",
    x_col="img",
    y_col="label",
    subset="training",
    color_mode ="grayscale",
    batch_size=bat_s,
    shuffle=False,
    class_mode="raw",
    target_size=(50,50))

# Text Model
embedding_matrix = create_embedding_matrix(
    '../mode/trained_models/glove.6B.50d.txt',
    tokenizer.word_index, embedding_dim)

text_input = Input(shape=(maxlen,))
model_text = layers.Embedding(vocab_size, embedding_dim, 
                              weights=[embedding_matrix], 
                              input_length=maxlen, 
                              trainable=False)(text_input)
model_text = Dropout(0.2)(model_text)
model_text = Bidirectional( LSTM(50 , return_sequences=True))(model_text) 
model_text = GlobalMaxPool1D()(model_text)
model_text = Dropout(0.2)(model_text)
model_text = layers.Dense(20, activation='relu')(model_text)
model_text = Dropout(0.1) (model_text)
model_text = layers.Dense(20, activation='relu')(model_text)
##################################################################################################################
image_input = Input(shape = (50, 50 , 1) )
model_image = Conv2D(32, (3, 3) )(image_input)
#mode_image = BatchNormalization() # newly added
model_image = Activation('relu')(model_image)
'''
model_image = MaxPooling2D(pool_size=(2, 2))(model_image)
model_image = Conv2D(32, (3, 3))(model_image)
model_image = Activation('relu')(model_image)
model_image = MaxPooling2D(pool_size=(2, 2))(model_image)
model_image = Conv2D(64, (3, 3))(model_image)
model_image = Activation('relu')(model_image)
'''
model_image = MaxPooling2D(pool_size=(2, 2))(model_image)
model_image = Flatten()(model_image)
#mode_image = BatchNormalization() # newly add
model_image = Dense(64)(model_image)
model_image = Activation('relu')(model_image)
model_image = Dropout(0.2)(model_image)

joint = concatenate( [model_image, model_text] ) #, mode='concat')
joint = Dense(64, activation='relu')(joint)
joint = Dropout(0.5)(joint)
joint = Dense(32,activation='relu')(joint)

prediction = Dense(1, activation = 'sigmoid')(joint)
Full_Model = Model(inputs=[text_input,image_input], outputs=[prediction])

Full_Model.compile(loss='categorical_crossentropy',
                   optimizer='rmsprop',
                   metrics=['accuracy'])
print('Starting Training now: ')
Full_Model.fit( [np.array(X_image_train), np.array(X_text_train)], np.array(Y_train),
                epochs=10 , batch_size=25 ,
                verbose=1, validation_split=0.2, shuffle=True)

model_json = Full_Model.to_json()
with open("MultiModel.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("MultiModel_weights.h5")
print("Saved model to disk")
'''
score = Full_Model.evaluate_generator(generator=test_generator )
print("Testing Loss: {:.4f}".format(score[0] ))
print("Testing Accuracy: {:.4f}".format(score[1]))
'''


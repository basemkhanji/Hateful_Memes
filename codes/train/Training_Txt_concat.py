import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score 
from keras.models import Model
#import keras.layers
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Dense , GRU , SpatialDropout1D, LSTM , Dropout, Flatten, Activation, BatchNormalization,Conv1D, MaxPooling1D ,GlobalMaxPool1D , Bidirectional , Input, concatenate
from keras import layers
from sklearn.metrics import accuracy_score , roc_auc_score
from keras.preprocessing.sequence import pad_sequences

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

df_train  = pd.read_csv("../../proc_data/traindev.csv")
df_phase2 = pd.read_csv("../../proc_data/test_unseen.csv")
df_phase1 = pd.read_csv("../../proc_data/test_seen.csv")

X_text_train  = df_train['text'][:6000]
X_image_train = df_train['Img_txt'][:6000]
Y_train       = df_train['label'][:6000]
X_text_test   = df_train['text']
X_image_test  = df_train['Img_txt']
Y_test        = df_train['label']

X_T_phase1 = df_phase1['text']
X_I_phase1 = df_phase1['Img_txt']
X_T_phase2 = df_phase2['text']
X_I_phase2 = df_phase2['Img_txt']

maxlen = 50
embedding_dim = 50
tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(X_text_train)
vocab_size = len(tokenizer.word_index) + 1

# padd all data sets
X_text_train = tokenizer.texts_to_sequences(X_text_train)
X_image_train = tokenizer.texts_to_sequences(X_image_train)
X_text_test  = tokenizer.texts_to_sequences(X_text_test)
X_image_test  = tokenizer.texts_to_sequences(X_image_test)
X_text_train  = pad_sequences(X_text_train, padding='post', maxlen=maxlen)
X_text_test   = pad_sequences(X_text_test, padding='post', maxlen=maxlen)
X_image_train = pad_sequences(X_image_train, padding='post', maxlen=maxlen)
X_image_test  = pad_sequences(X_image_test, padding='post', maxlen=maxlen)

X_T_phase1 = tokenizer.texts_to_sequences(X_T_phase1)
X_I_phase1 = tokenizer.texts_to_sequences(X_I_phase1)
X_T_phase2 = tokenizer.texts_to_sequences(X_T_phase2)
X_I_phase2 = tokenizer.texts_to_sequences(X_I_phase2)
X_T_phase1  = pad_sequences(X_T_phase1, padding='post', maxlen=maxlen)
X_I_phase1  = pad_sequences(X_I_phase1, padding='post', maxlen=maxlen)
X_T_phase2  = pad_sequences(X_T_phase2, padding='post', maxlen=maxlen)
X_I_phase2  = pad_sequences(X_I_phase2, padding='post', maxlen=maxlen)

embedding_matrix = create_embedding_matrix(
    '../mode/trained_models/glove.6B.50d.txt',
    tokenizer.word_index, embedding_dim)

text_input = Input(shape=(maxlen,))
model_text = layers.Embedding(vocab_size, embedding_dim, 
                           weights=[embedding_matrix], 
                           input_length=maxlen, 
                              trainable=False)(text_input)
model_text= Bidirectional(LSTM(50 , return_sequences=True))(model_text)
model_text= GlobalMaxPool1D()(model_text)
model_text= Dropout(0.2)(model_text)
model_text= layers.Dense(120, activation='relu')(model_text) # was 5
model_text= Dropout(0.1)(model_text)
model_text= layers.Dense(64, activation='relu')(model_text)
model_text= layers.Dense(32, activation='relu')(model_text)
model_text= Dense(6, activation='sigmoid')(model_text)


image_input = Input(shape=(maxlen,))
model_image=layers.Embedding(vocab_size, embedding_dim, 
                           weights=[embedding_matrix], 
                           input_length=maxlen, 
                           trainable=True)(image_input)
#1 - best arch 
model_image= Bidirectional(LSTM(50 , return_sequences=True))(model_image)
model_image= GlobalMaxPool1D()(model_image)
model_image= Dropout(0.2)(model_image)
model_image= layers.Dense(64, activation='relu')(model_image)
model_image= Dropout(0.1) (model_image)
model_image= layers.Dense(32, activation='relu')(model_image)
model_image= Dense(6, activation='sigmoid')(model_image)

joint = concatenate( [model_image, model_text] )
prediction = layers.Embedding( vocab_size, embedding_dim)(joint)
prediction = layers.Dropout(0.5)(prediction)
#prediction= layers.Conv1D(128, 7, padding="valid", activation="relu", strides=3)(prediction)
prediction= layers.Conv1D(128, 7, padding="valid", activation="relu", strides=3)(prediction)
prediction= layers.GlobalMaxPooling1D()(prediction)
#prediction = Bidirectional(LSTM(50 , return_sequences=True))(joint)
#prediction = GlobalMaxPool1D()(joint)
prediction = layers.Dense(64, activation='relu')(joint)
#prediction = layers.Conv1D(filters=32, kernel_size=3, padding='same', activation='relu')(prediction)
#prediction = MaxPooling1D(pool_size=2)(prediction)
#prediction = Bidirectional(LSTM(100 , return_sequences=True))(prediction)
prediction = Dropout(0.1)(prediction)
prediction = layers.Dense(32, activation='relu')(prediction)
prediction = Dense(1, activation = 'sigmoid')(prediction)

Full_Model = Model(inputs=[text_input,image_input], outputs=[prediction])
Full_Model.summary()
Full_Model.compile(loss='binary_crossentropy',
                   optimizer='rmsprop',
                   metrics=['accuracy'])
Full_Model.fit( [np.array(X_image_train), np.array(X_text_train)], np.array(Y_train),
                epochs=40,
                verbose=1, shuffle=True)

import pickle
# saving
with open('../mode/tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
model_json = Full_Model.to_json()
with open("../mode/text_image_model.json", "w") as json_file:
   json_file.write(model_json)
Full_Model.save_weights("../mode/text_image_model_w.h5")
print("Saved txt model to disk")

#Y_pred_test = Full_Model.predict([X_text_test,X_image_test])
#print("Testing AUC:", roc_auc_score(Y_pred_test, Y_test))
#plot_history(history)
print(Full_Model.predict( [X_text_test,X_image_test]  ))
df_train['ti_DNN'] = Full_Model.predict( [X_text_test,X_image_test] ) #[:0]
print(df_train)
df_train.to_csv('train_proc_tiDNN.csv', encoding='utf-8', index=False)


#X_T_phase1  = pad_sequences(X_T_phase1, padding='post', maxlen=maxlen)
#X_I_phase1  = pad_sequences(X_I_phase1, padding='post', maxlen=maxlen)

df_phase1['ti_DNN'] = Full_Model.predict( [ X_T_phase1, X_I_phase1]  )
df_phase2['ti_DNN'] = Full_Model.predict( [ X_T_phase2, X_I_phase2]  )
df_phase1.to_csv('seen_ph1.csv', encoding='utf-8', index=False)
df_phase2.to_csv('unseen_ph2.csv', encoding='utf-8', index=False)


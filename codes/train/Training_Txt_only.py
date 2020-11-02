import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score 
from keras.models import Model
#import keras.layers
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Dense , GRU , SpatialDropout1D, LSTM , Dropout, Flatten, Activation, BatchNormalization,Conv1D, MaxPooling1D ,GlobalMaxPool1D , Bidirectional
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


df_train = pd.read_csv("../../proc_data/traindev.csv")
X_train, X_test , Y_train , Y_test = train_test_split( df_train['text'].values , df_train['label']  ,
                                                       test_size = 0.3 , random_state = 7 )

maxlen = 10
embedding_dim = 10
tokenizer = Tokenizer(num_words=50000)
tokenizer.fit_on_texts(X_train)
X_train = tokenizer.texts_to_sequences(X_train)
X_test  = tokenizer.texts_to_sequences(X_test)
vocab_size = len(tokenizer.word_index) + 1


X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)
#Text_full_tokenized =  pad_sequences( Text_full_tokenized , padding='post', maxlen=maxlen)
# we need to make use of google's gloVe
# take the function from here: https://realpython.com/python-keras-text-classification :

embedding_matrix = create_embedding_matrix(
    '../mode/trained_models/glove.6B.50d.txt',
    tokenizer.word_index, embedding_dim)

model = Sequential()
model.add(layers.Embedding(vocab_size, embedding_dim, 
                           weights=[embedding_matrix], 
                           input_length=maxlen, 
                           trainable=True))
#1 - best arch 
model.add(Bidirectional(LSTM(50 , return_sequences=True))) # newlly added
model.add(GlobalMaxPool1D())
#model.add(Bidirectional(LSTM(150 , return_sequences=True))) # newlly added
model.add(Dropout(0.2))
model.add(layers.Dense(120, activation='relu')) # was 5
model.add(Dropout(0.1)) # was 0.5
model.add(layers.Dense(1024, activation='relu')) # was 5
model.add(layers.Dense(20, activation='relu')) # was 5
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.summary()

history = model.fit(X_train, Y_train,
                    epochs=100,
                    verbose=True,
                    validation_data=(X_test, Y_test),
                    batch_size= 128)
#class_weight=class_weights )#220

loss, accuracy = model.evaluate(X_test, Y_test, verbose=False)
print("Testing Accuracy:  {:.4f}".format(accuracy))

import pickle
# saving
with open('../mode/tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
model_json = model.to_json()
with open("../mode/text_model.json", "w") as json_file:
   json_file.write(model_json)
model.save_weights("../mode/text_model_w.h5")
print("Saved txt model to disk")

Y_pred_test = model.predict(X_test)
print("Testing AUC:", roc_auc_score(Y_pred_test, Y_test))
#plot_history(history)
#df_train['txt_embd_DNN'] = model.predict_proba(Text_full_tokenized)
#print(df_train)
#df_train.to_csv('train_proc_withtxtDNN.csv', encoding='utf-8', index=False)



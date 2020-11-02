import pandas as pd
import numpy as np
from xgboost import XGBClassifier , plot_importance
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score , roc_auc_score
from sklearn.model_selection import RandomizedSearchCV ,GridSearchCV 
from sklearn.model_selection import StratifiedKFold
import joblib
from sklearn.utils import class_weight
from sklearn.neural_network import MLPClassifier
import catboost
from catboost import CatBoostClassifier

def add_log_features(df):
    df['log_Bluriness']  = np.log(df['Bluriness'])
    df['log_Imagedim']   = np.log(df['Imagedim'])
    df['log_paletCol_1'] = np.log(df['paletCol_1'])
    df['log_paletCol_2'] = np.log(df['paletCol_2'])
    df['log_paletCol_3'] = np.log(df['paletCol_3'])
    df['log_paletCol_4'] = np.log(df['paletCol_4'])
    df['log_paletCol_5'] = np.log(df['paletCol_5'])
    return df

drop_columns = ['id','img' , 'text' , 'DenseNet','Yolo','DenseNet_obj','Yolo_obj','Img_txt','palette_color' ,
                'polarity_scores','Bluriness' , 'Imagedim','paletCol_1','paletCol_2', 'paletCol_3','paletCol_4','paletCol_5', 'txt_embd_DNN', 'img_embd_DNN'  ]
Tune_parms = False
Other_classifiers = True
#df_train  = pd.read_csv("../../proc_data/traindev.csv")[5000:]
df_train  = pd.read_csv("train_proc_tiDNN.csv")
Y_Labels  = df_train['label']
df_train  = add_log_features(df_train)
df_train  = df_train.drop(columns=drop_columns )

print(df_train)
print(df_train.describe())
print(df_train.columns)

class_weights = class_weight.compute_class_weight('balanced',
                                                  np.unique(Y_Labels),
                                                  Y_Labels)

X_Features= df_train.drop(columns=['label'])
# X_train, X_test , Y_train , Y_test = train_test_split( X_Features, Y_Labels, test_size = 0.25 , random_state = 100 )
X_train = X_Features[4000:]
X_test  = X_Features[:4000]
Y_train = Y_Labels[4000:]
Y_test  = Y_Labels[:4000]
# 1- first model :
model_catboost = CatBoostClassifier(verbose=0, n_estimators=1000 , learning_rate=0.01  , depth=8 , loss_function='CrossEntropy') # Gold: 0.82 AUC
#2- Optimized model
#Best parameters set according to tunning below:
#{'border_count': 100, 'depth': 10, 'iterations': 1000, 'l2_leaf_reg': 1, 'learning_rate': 0.01, 'thread_count': 4}
model_catboost = CatBoostClassifier(verbose =0,  border_count= 100, depth= 10,
                                    iterations= 1000, l2_leaf_reg= 1, learning_rate= 0.001,
                                    thread_count= 4 ,
                                    #loss_function='CrossEntropy',
                                    loss_function='Logloss',
                                    class_weights=class_weights ) # Gold: 0.82 AUC

model_catboost.fit( X_train , Y_train )
Y_pred_cb_test  = model_catboost.predict( X_test )
joblib.dump(model_catboost,'catboost_model.txt' )
print('Acuracy score for CatBoost: ' , accuracy_score( Y_pred_cb_test , Y_test  ))
cat_probs = model_catboost.predict_proba(X_test)[:,1]
cat_pred  = model_catboost.predict(X_test)
print('ROC AUC(CATBOOST) = %.5f' % roc_auc_score(Y_test, cat_pred))
df_train['cb_proba'] = model_catboost.predict_proba(X_Features)[:,1]
#df_train['cb_proba'].plot.hist(bins= 100 , alpha = 0.5)

df_test_raw = pd.read_csv("seen_ph1.csv")
print(df_test_raw.head())
test_Features = add_log_features(df_test_raw)
test_Features = test_Features.drop(columns = drop_columns)
df_test_raw['proba'] = model_catboost.predict_proba( test_Features )[:,1]
df_test_raw['label'] = df_test_raw['proba'].apply( lambda x: 1 if x >0.5 else 0 )
df_sub = df_test_raw[['id' , 'proba' , 'label']]
print(df_sub)
df_sub.to_csv('./Submision_phase1.csv', encoding='utf-8', index=False)


df_phase2_raw = pd.read_csv("./unseen_ph2.csv")
print(df_phase2_raw.head())
phase2_Features = add_log_features(df_phase2_raw)
phase2_Features = phase2_Features.drop(columns = drop_columns)
df_phase2_raw['proba'] = model_catboost.predict_proba( phase2_Features )[:,1]
df_phase2_raw['label'] = df_phase2_raw['proba'].apply( lambda x: 1 if x >0.5 else 0 )
df_sub = df_phase2_raw[['id' , 'proba' , 'label']]
print(df_sub)
df_sub.to_csv('./Submision_phase2.csv', encoding='utf-8', index=False)


if Tune_parms:
    # tune the classifier :
    params = {'depth':[2,6,8,10],
              'iterations':[1000,1500,2000],
              'learning_rate':[0.01,0.02, 0.05, 0.1], 
              'l2_leaf_reg':[1,5,10,100],
              'scale_pos_weight': [ 1, 5 , 10 , 20 , 40 ,50],
              #'border_count':[10,20,50,100,200],
              'thread_count':[4]
    }
    
    model_cat      = catboost.CatBoostClassifier(verbose =0)
    model_catboost = GridSearchCV( model_cat
                                   , params, cv=3,
                                   #, cat_features= [0, 1 , 14, 15 ] ,
                                   #cv_splitter=StratifiedKFold(n_splits=3) ,
                                   scoring='roc_auc' )
    model_catboost.fit( X_train , Y_train )
    print("Best parameters set found on development set:")
    print()
    print(model_catboost.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    means = model_catboost.cv_results_['mean_test_score']
    stds  = model_catboost.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"% (mean, std * 2, params))

if Other_classifiers:
    from keras.models import Model
    import keras.layers
    import tensorflow as tf
    # Try Neural Networks :
    from numpy import loadtxt
    from keras.models import Sequential
    from keras.layers import Dense , GRU
    #cv = RepeatedKFold(n_splits=10, n_repeats=5, random_state=50)
    model_xgb = XGBClassifier(objective = "binary:logistic" , cv= 3 , max_depth=5, n_estimators=1000, learning_rate=0.05 , nthread=5 , scale_pos_weight=1, seed=27)
    model_xgb.fit( X_train , Y_train )
    # predicting the labels for the same train sample
    Y_pred_train_xgb = model_xgb.predict( X_train )
    # predicting the labels for the test sample
    Y_pred_test_xgb  = model_xgb.predict( X_test )
    joblib.dump(model_xgb,'xgb_model.txt' ) 
    print('ROC AUC(CATBOOST) = %.5f' % roc_auc_score(Y_test, Y_pred_test_xgb))
    #from matplotlib import pyplot
    #print(model_catboost.feature_importances_)
    #plot_importance(model_catboost)
    #pyplot.show()
    
    """
    model_DNN = Sequential()
    model_DNN.add(Dense(12 , input_dim=len(X_Features.columns), activation='relu'))
    model_DNN.add(Dense(24, activation='relu'))
    model_DNN.add(Dense(8, activation='relu'))
    model_DNN.add(Dense(15, activation='relu'))
    model_DNN.add(Dense(1, activation='sigmoid'))
    model_DNN.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    model_DNN.fit( X_train , Y_train , epochs=200, batch_size=10, validation_split=0.2,shuffle=True)
    _, accuracy = model_DNN.evaluate(X_train , Y_train )
    print('Accuracy: %.2f' % (accuracy*100))
    Y_pred_test_DNN= model_DNN.predict_classes( X_test )
    print('Acuracy score for DNN     : ' , accuracy_score( Y_pred_test_DNN, Y_test ))
    print('ROC AUC(CATBOOST) = %.5f' % roc_auc_score(Y_test, Y_pred_test_DNN))
    """


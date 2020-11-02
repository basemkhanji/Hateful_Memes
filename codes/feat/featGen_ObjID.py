import pandas as pd
import numpy as np
import joblib
from imageai.Prediction import ImagePrediction
import swifter
from imageai.Detection import ObjectDetection
from collections import Counter
import spacy
import statistics 
from statistics import mode 
import cv2
from colorthief import ColorThief
import profanity_check as pfty
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from textblob import Word, TextBlob
from profanity_check import predict, predict_prob
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pickle
from profanity_filter import ProfanityFilter
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.sequence import pad_sequences
import time
#----------------------------------------------------------
# Config ....
sid = SentimentIntensityAnalyzer()
nlp = spacy.load("en_core_web_sm")
pf = ProfanityFilter(nlps={'en': nlp})
nlp.add_pipe(pf.spacy_component, last=True)
stop = stopwords.words('english')
special_char = ['~','@','$','#','%','^','&','*','(',')','-','_',',',';','/','\\','>','<','|','[',']','}','{','"','\'','`' , '?' , '!' , '...']
path_dir = '/home/bassem/DataDriven_HatfulMemes/data/'
print('----------------------------------------------------------------------')
# Get features from text :

def getsentiment( text ):
    # remove stopp words:
    text = text.replace('[^\w\s]','')
    text = " ".join(x for x in text.split() if x not in stop)
    text = " ".join( [Word(word).lemmatize() for word in text.split()] )
    return TextBlob(text).sentiment[0]
                    

def sim_spacy( x  , y ):
    nlp(x ).similarity( nlp( y ) )

def Make_Features( df ):
    
    multiple_prediction = ImagePrediction()
    multiple_prediction.setModelTypeAsDenseNet()
    multiple_prediction.setModelPath("../mode/trained_models/DenseNet-BC-121-32.h5")
    multiple_prediction.loadModel()
    detector = ObjectDetection()
    detector.setModelTypeAsYOLOv3()
    detector.setModelPath( "../mode/trained_models/yolo.h5")
    detector.loadModel()

    df['DenseNet'] = df['img'].swifter.apply(lambda x: multiple_prediction.predictImage( path_dir+ x , result_count=3))
    df['Yolo']          = df['img'].swifter.apply(lambda x: detector.detectObjectsFromImage( path_dir+ x , output_image_path='./new.jpg', minimum_percentage_probability=20))
    df['DenseNet_obj']  = df['DenseNet'].swifter.apply(lambda x : x[0][0] + ' and ' + x[0][1] + ' and ' + x[0][2]) 
    df['Yolo_obj']      = df['Yolo'].swifter.apply(lambda x: ' '.join(word for word in [l['name'] for l in x])  , axis =1) 
    df['Img_txt']       = df['DenseNet_obj'] + ' ' + df['Yolo_obj']  
    df['palette_color'] = df['img'].swifter.apply(lambda x:  ColorThief(path_dir+ x).get_palette(color_count=5) )
    df['Bluriness']     = [cv2.Laplacian( cv2.imread(path_dir+x , 0 ) ,  cv2.CV_64F  ).var() for x in df['img'] ]
    df['Imagedim']      = [cv2.imread(path_dir+x ).flatten().shape[0] for x in df['img'] ]
    df['Yolo_unique']   = df['Yolo_obj'].swifter.apply(lambda x: len(set(x)) )
    df['Yolo_N_obj']    = df['Yolo_obj'].swifter.apply(lambda x: len( x ) )
    #print(df.describe() )
    # First cross variable between text and image :
    df['sim_txt_img_gen']  = df.swifter.apply(lambda x: nlp(x.text).similarity( nlp(x.DenseNet_obj ) )  , axis = 1)
    df['sim_txt_img_objs'] = df.swifter.apply(lambda x: nlp(x.text).similarity( nlp( ' and '.join(word[0] for word in x.Yolo_obj) ) )  , axis = 1)
    # extract dominant colors from image   
    df['paletCol_1']    = df['palette_color'].swifter.apply(lambda x: (x[0][0]* 65536 + x[0][1] * 256 + x[0][2]))
    df['paletCol_2']    = df['palette_color'].swifter.apply(lambda x: (x[1][0]* 65536 + x[1][1] * 256 + x[1][2]))
    df['paletCol_3']    = df['palette_color'].swifter.apply(lambda x: (x[2][0]* 65536 + x[2][1] * 256 + x[2][2]))
    df['paletCol_4']    = df['palette_color'].swifter.apply(lambda x: (x[3][0]* 65536 + x[3][1] * 256 + x[3][2]))
    df['paletCol_5']    = df['palette_color'].swifter.apply(lambda x: (x[4][0]* 65536 + x[4][1] * 256 + x[4][2]))
    # Get Blurry status
    # Get shapes
    df['brightness']    =  [cv2.mean(cv2.cvtColor(cv2.imread(path_dir+x ) , cv2.COLOR_BGR2HSV ))[1]/255. for x in df['img'] ]
    df['Saturation']    =  [cv2.mean(cv2.cvtColor(cv2.imread(path_dir+x ) , cv2.COLOR_BGR2HSV ))[0]/255. for x in df['img'] ]
    df['ImageValue']    =  [cv2.mean(cv2.cvtColor(cv2.imread(path_dir+x ) , cv2.COLOR_BGR2HSV ))[2]/255. for x in df['img'] ]
    
    df['word_count']     = df['text'].swifter.apply(lambda x: len(str(x).split(" ") ) )
    df['char_count']     = df['text'].str.len()
    df['stp_count']      = df['text'].swifter.apply( lambda x: len( [x for x in x.split() if x in stop] ) )
    df['spc_count']      = df['text'].swifter.apply( lambda x: len( [x for x in list(x) if x in special_char] ) )
    df['sentiment_txt']  = df['text'].swifter.apply(lambda x: getsentiment(x) )
    df['sentiment_img']  = df['DenseNet_obj'].swifter.apply(lambda x: getsentiment(x) )
    df['prfn_ftr']       =  df['text'].swifter.apply(lambda x:nlp(x)._.is_profane )
    df['Quant'] = df['text'].swifter.apply(lambda x: len( [y for y in nlp(x).ents if str(y.label_) =='MONEY' or str(y.label_) =='DATE' or str(y.label_) =='TIME' or str(y.label_) =='PERCENT' or str(y.label_) == 'ORDINAL' or str(y.label_) =='CARDINAL' or str(y.label_) == 'QUANTITY']))
    df['Ent']  = df['text'].swifter.apply(lambda x: len( [y for y in nlp(x).ents if str(y.label_) =='PERSON' or str(y.label_) =='NORP' or str(y.label_) =='ORG' or str(y.label_) == 'LOC' or str(y.label_) =='GPE' or str(y.label_) =='WORK_OF_ART' or str(y.label_) =='EVENT']))
    
    df['polarity_scores']  = df['text'].swifter.apply(lambda x: sid.polarity_scores(x))
    df['neg_txt']  = df['polarity_scores'].swifter.apply(lambda x: x['neg'])
    df['neu_txt']  = df['polarity_scores'].swifter.apply(lambda x: x['neu'])
    df['pos_txt']  = df['polarity_scores'].swifter.apply(lambda x: x['pos'])
    df['com_txt']  = df['polarity_scores'].swifter.apply(lambda x: x['compound'])
    #df = df.drop(columns=['DenseNet' ,'DenseNet_obj', 'Yolo' , 'Yolo_obj' , 'palette_color', 'polarity_scores'])
    return df

# extract features from raw test in df:
# Read the data in dataframes
df_train     = pd.read_json("../../data/train.jsonl",lines=True)
df_dev_seen  = pd.read_json("../../data/dev_seen.jsonl",lines=True)
df_dev_unseen= pd.read_json("../../data/dev_unseen.jsonl",lines=True)

df_taindev_all = pd.concat([df_train,df_dev_seen,df_dev_unseen], ignore_index=True)
df_test_unseen = pd.read_json("../../data/test_unseen.jsonl", lines=True)
df_test_seen   = pd.read_json("../../data/test_seen.jsonl", lines=True)
print('name is :' , df_taindev_all)
# Load txt model
maxlen=50
with open('../mode/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
from keras.models import model_from_json
#Load the model weigths form h5 file :
json_file_txt = open('../mode/text_model.json', 'r')
loaded_txt_model = json_file_txt.read()
json_file_txt.close()
loaded_txt_model = model_from_json(loaded_txt_model)
loaded_txt_model.load_weights("../mode/text_model_w.h5")
print("Loaded model from disk")
json_img_file = open('../mode/Image_model/modelimageonly.json', 'r')
loaded_img_model = json_img_file.read()
json_img_file.close()
loaded_img_model = model_from_json(loaded_img_model)
loaded_img_model.load_weights("../mode/Image_model/modelimgeonly_weights.h5")
loaded_img_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

tic = time.perf_counter()
list_df = [ df_taindev_all , df_test_seen,df_test_unseen]
list_of_names = ['traindev' , 'test_seen' , 'test_unseen']
i=0
for Mydf in list_df:
    # extract features from raw test in df:
    
    Mydf = Make_Features(Mydf)
    # add text DNN
    Text = Mydf['text'].values
    Text_tokenized = tokenizer.texts_to_sequences(Text)
    Text_tokenized = pad_sequences( Text_tokenized , padding='post', maxlen=maxlen)
    Mydf['txt_embd_DNN'] = loaded_txt_model.predict_proba(Text_tokenized)    
    # add Image DNN:
    datagen_full  = ImageDataGenerator(rescale=1./255.) ; bat_s = 64
    Full_generator= datagen_full.flow_from_dataframe(dataframe=Mydf,directory="../../data/",
                                                     x_col="img",
                                                     y_col="label", seed = 42,
                                                     batch_size= 1,shuffle=False,
                                                     class_mode=None, target_size=(200,200))
    #print( len(Full_generator))
    Mydf['img_embd_DNN'] = loaded_img_model.predict_generator( Full_generator
                                                               ,steps= None
                                                               , verbose = 1)[:,0]
    #print(Mydf)
    # 2- define the df of labels:
    Mydf = Mydf.round(4)
    Mydf.to_csv('../../proc_data/'+ list_of_names[i]+'.csv', encoding='utf-8', index=False)
    i = i+1
toc = time.perf_counter()
print(f"Downloaded the tutorial in {(toc - tic)/3600:0.4f} hours")






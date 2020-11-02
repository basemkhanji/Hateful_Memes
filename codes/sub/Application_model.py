import pandas as pd
import numpy as np
from xgboost import XGBClassifier , plot_importance
import joblib
import spacy
import profanity_check as pfty
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from textblob import Word, TextBlob
from profanity_check import predict, predict_prob
import swifter

nlp = spacy.load("en_core_web_sm")
from profanity_filter import ProfanityFilter
pf = ProfanityFilter(nlps={'en': nlp})
nlp.add_pipe(pf.spacy_component, last=True)
stop = stopwords.words('english')
special_char = ['~','@','$','#','%','^','&','*','(',')','-','_',',',';','/','\\','>','<','|','[',']','}','{','"','\'','`']

df_test_raw = pd.read_json("../data/test_unseen.jsonl", lines=True)
print(df_test_raw.head())

def Make_Features( df ):
    df['Freq'] = df['text'].map(df['text'].value_counts())
    df['word_count'] = df['text'].swifter.apply(lambda x: len(str(x).split(" ") ) )
    df['char_count'] = df['text'].str.len()
    df['stpw_count'] = df['text'].swifter.apply( lambda x: len( [x for x in x.split() if x in stop] ) )
    
    df['spchar_count'] = df['text'].swifter.apply( lambda x: len( [x for x in list(x) if x in special_char] ) )
    #prepare for sentiment analysis:
    # 1- remove the punctuations :
    df['text_modif'] = df['text'].str.replace('[^\w\s]','')
    # 2- remove stop words:
    df['text_modif'] = df['text_modif'].swifter.apply(lambda x: " ".join(x for x in x.split() if x not in stop))
    df['text_modif'] = df['text_modif'].swifter.apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))
    df['sentiment'] = df['text_modif'].swifter.apply(lambda x: TextBlob(x).sentiment[0] )
    # finally add profanity_check
    df['profane_modf'] =  predict_prob(df['text_modif'])
    #print(df[['word_count','char_count','stpw_count','spchar_count', 'profane', 'sentiment']].head())
    df['profane_pfilter'] =  df['text'].swifter.apply(lambda x:nlp(x)._.is_profane )
    # spacy to identify people organizations etc ...
    df['org']   = df['text'].swifter.apply(lambda x: len( [y for y in nlp(x).ents if str(y.label_) =='ORG']))
    df['Money'] = df['text'].swifter.apply(lambda x: len( [y for y in nlp(x).ents if str(y.label_) =='MONEY']))
    df['tDate'] = df['text'].swifter.apply(lambda x: len( [y for y in nlp(x).ents if str(y.label_) =='DATE' or str(y.label_) =='TIME']))
    df['Pers']  = df['text'].swifter.apply(lambda x: len( [y for y in nlp(x).ents if str(y.label_) =='PERSON']))
    df['GPE']   = df['text'].swifter.apply(lambda x: len( [y for y in nlp(x).ents if str(y.label_) =='GPE']))
    print(df.describe() )
    return df

# Load xgb model : 
xgb_model = joblib.load('xgb_model.txt')

# test :
df_test     = Make_Features(df_test_raw)
df_test_app = df_test.drop(columns = ['id' , 'img', 'text' , 'text_modif'])
df_test['proba'] = xgb_model.predict_proba( df_test_app )[:,1]
df_test['label'] = df_test['proba'].apply( lambda x: 1 if x >0.5 else 0 )
#df_test['label'] = xgb_model.predict(df_test_app)
print(df_test)
df_test = df_test[['id' , 'proba' , 'label']]
df_test.to_csv('Submision.csv', encoding='utf-8', index=False)



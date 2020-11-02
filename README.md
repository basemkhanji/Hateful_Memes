# Hateful_Memes

Th repository contains all the codes I used to get to rank 60 in the follwoing compitition:
https://www.drivendata.org/competitions/70/hateful-memes-phase-2/ my username there is bkhanji

The code is python-based and it is split into two main phases:
1- Adding features: this is done using codes/feat/featGen_ObjID.py  , the idea is to extract all features form text and images in columns. Text is explored using Spacy package, while I used DenseNet and Yolo to identify items in the image and turn them into text and cross-section the result with cleaned/lemmentized text

2- Training: this features training single feature text or image against part of the data using Keras neural network setup, then combine the output with the column features from the step above. Another approach is to train the text-image object using functional API in keras and provide that output to be combined with column features using CatBoost.

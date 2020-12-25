# Hateful_Memes

Th repository contains codes I used to get to rank 55 in the follwoing compitition:
https://www.drivendata.org/competitions/70/hateful-memes-phase-2/ my username there is bkhanji, the compition is sponsored by facebook and it has more than 3000 contestant/teams.

The code is python-based and it is split into two phases:

#1- Adding features: 

This is done using codes/feat/featGen_ObjID.py, the idea is to extract all features from text and images in columns. Text is explored and cleaned using Spacy package, while I used DenseNet and Yolo to identify items in the image and turn them into text. Finally I cross-sectioned the result cleaned text from the text column and text exctracted from the image. 

#2- Training: 

Training is done using text or image as input against part of the data using Keras neural network setup. Following I combine the output DNN clssifiers with the column-features from the step above. 
Another approach which I did not have the time to explore: train the text-image object using functional API in keras and provide that output to be combined with column features using CatBoost. I wrote the needed script to do that but lacked the computing power to carry out this step.

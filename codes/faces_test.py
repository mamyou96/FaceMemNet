# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 16:44:39 2021

@author: anonymous
"""

import numpy as np
from PIL import Image
from cv2 import resize
import tensorflow as tf
from sklearn.model_selection import train_test_split
import multiprocessing as mp
import pandas as pd
import os
import csv
from keras_vggface.vggface import VGGFace
from tensorflow.keras.layers import Dense, Activation, Dropout, Flatten, Input, Convolution2D, ZeroPadding2D, MaxPooling2D, Activation
from tensorflow.keras.models import Model, Sequential

#memorability-scores.xlsx path
mem_score_xlsx = "faces/Memorability Scores/memorability-scores.xlsx"
#images
face_images = "faces/10k US Adult Faces Database/Face Images/"

def euclidean_distance_loss(y_true, y_pred):
    return tf.math.sqrt(tf.math.reduce_sum(tf.math.square(y_pred - y_true), axis=-1))


def load_image(image_file):
    image = Image.open(image_file).resize((224,224)).convert("RGB")
    return np.array(image, dtype=np.uint8)


def load_split(split_file):
    faces_ds = pd.read_excel(split_file)
    X_train, X_test= train_test_split(faces_ds, test_size=0.2, random_state=42)
    scores_train = list(X_train['Hit Rate (HR)'])
    names_train = list(X_train['Filename'])
    scores_test = list(X_test['Hit Rate (HR)'])
    names_test = list(X_test['Filename'])
    FA_test = list(X_test['False Alarm Rate (FAR)'])
    FA_train = list(X_train['False Alarm Rate (FAR)'])
    X_train = [[names_train[i],scores_train[i]] for i in range(len(scores_train))]
    X_test = [[names_test[i],scores_test[i]] for i in range(len(scores_test))]
    X_train = X_train[:int(len(X_train)/2)]
    X_valid = X_test[:int(len(X_test)/2)]
    X_test = X_test[int(len(X_test)/2):]
    return X_train
    

def lamem_generator(split_file, batch_size=32):
        num_samples = len(split_file)
        for offset in range(0, num_samples, batch_size):
            if offset+batch_size>num_samples:
                batch_samples = split_file[offset:]
            else:
                batch_samples = split_file[offset:offset+batch_size]
            inputs = mp.Pool().map(load_image, [face_images+ i[0] for i in batch_samples])
			final_labels = [[i[1]] for i in batch_samples]
            yield([np.array(inputs),np.array(inputs),np.array(inputs)], np.array(final_labels))

cors = []


for i in range(30):  

	model = tf.keras.models.load_model("models/epoch_"+str(i+5)+".h5", custom_objects={'euclidean_distance_loss': euclidean_distance_loss})          
    test_split = load_split(mem_score_xlsx)
    test_split = lamem_generator(test_split)

    true_values = []
    predictions = []
    for idx, (img, target) in enumerate(test_split):
        predict = model.predict(img)
        true_values.append(target)
        predictions.append(predict)
        print(idx) 
        print("True memorability score: " + str(target)) 
        print("Predicted memorability score: " + str(predict))   


    y_pred = np.asarray(predictions[0])
    y_test = np.asarray(true_values[0])



    for i in range(len(predictions)-1):
        y_pred = np.vstack((y_pred, np.asarray(predictions[i+1])))
        y_test = np.vstack((y_test, np.asarray(true_values[i+1])))
    

    from scipy.stats import spearmanr
    coef, p = spearmanr(y_test, y_pred)
    temp = 0
    for i in range(len(y_pred)):
      temp += abs(y_pred[i]-y_test[i])**2
    temp = temp/len(y_pred)
    print("error: " + str(temp))

    print('Spearmans correlation coefficient: %.3f' % coef)
    cors.append(coef)
print(np.mean(cors))
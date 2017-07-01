# -*- coding: utf-8 -*-
"""
Created on Fri Jun 23 13:37:51 2017

@author: Tom
"""

import pandas as pd 
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers.recurrent import LSTM
from pandas import DataFrame
import random
from keras.layers.embeddings import Embedding
from keras.datasets import imdb
from keras.preprocessing import sequence
from keras.layers import Dropout
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras import metrics

def reduceDataMatrixByMac(dataMatrix, macs_list):
    all_samples = dataMatrix
    
    reduced_samples = []
    labels = []
    for sample in all_samples:
        mac = sample[-4]
        if mac in macs_list:
            reduced_samples.append(sample)
    
    reduced_samples = np.array(reduced_samples)
    
    labels = np.array(reduced_samples[:,-3])
    
    #Clear irrelavent features
    reduced_samples = np.array(reduced_samples[:,:-5])
    #Scale Features
#    reduced_samples = MinMaxScaler().fit_transform([ s.astype(float) for s in reduced_samples ])

#    reduced_samples = MinMaxScaler().fit_transform(reduced_samples)
    return reduced_samples, labels

def createWindows(reduced_samples, labels, window_size=1):
#    for sample_index in range(len(all_samples) - window_size - 1):
    windows = []
    for i in range(len(reduced_samples) - window_size + 1):
        windows.append(reduced_samples[i:(i + window_size), :])
    
    windows = np.array(windows)
    windows_labels = labels[:len(windows)]
    return windows, windows_labels


def getWindowsAndLabels(file_path, window_size=1):
    csv_data = pd.read_csv(file_path)
    samples = csv_data.as_matrix()
    
    #Get MACs
    macs = set(samples[:,-4])
    
    #Get classes
    classes = list(set(samples[:,-3]))
    classes.sort()
    
    classes_encoded = LabelEncoder().fit_transform(classes)
    classes_encoded_utils = np_utils.to_categorical(classes_encoded)
    
    #Split samples by macs
    ret_windows_labels = []
    ret_windows = []
    for mac, i in zip(macs, range(len(macs))):
        reduced_samples, labels = reduceDataMatrixByMac(samples,[mac])
        windows, windows_labels = createWindows(reduced_samples, labels, window_size=window_size)
#        windows = sequence.pad_sequences(reduced_samples, maxlen=window_size)
        #Create encoded labels
        train_window_labels = []
        for windows_label in windows_labels:
            train_window_labels.append(classes_encoded_utils[classes.index(windows_label)])
        ret_windows_labels.append(np.array(train_window_labels))
        #Get Windows
        ret_windows.append(windows)
    
    return ret_windows, ret_windows_labels, classes
       

TEST_FILE_PATH = r"IoT_data_test.csv"
TRAINING_FILE_PATH = r"IoT_data_training_above_3000_one_mac.csv"
VALIDATION_FILE_PATH = r"IoT_data_validation_above_3000.csv"

#TRAINING_FILE_PATH = r"IoT_data_training_lights_tv.csv"
#VALIDATION_FILE_PATH = r"IoT_data_validation_lights_tv.csv"

window_size = 25

train_windows, train_windows_labels, classes = getWindowsAndLabels(TRAINING_FILE_PATH,window_size=window_size)

##### LSTM #####

# fix random seed for reproducibility
np.random.seed(123)

# create and fit the LSTM networks
#batch_size = 2900 - window_size + 1
batch_size = 1
#models = {}
#i = 0
#for trainX, trainY in zip(train_windows, train_windows_labels):
#    # trainY[i] is the same for all i because it is the same mac, so we get the same label
#    if trainY[0].tostring() not in models:
#        model = Sequential()
#        model.add(LSTM(11, batch_input_shape=(batch_size, window_size, trainX.shape[2]), stateful=True))
#        model.add(Dense(len(classes),  activation='softmax'))
#        model.compile(loss='categorical_crossentropy', optimizer='adam')            
#        models[trainY[0].tostring()] = model
#    
#    model = models[trainY[0].tostring()]
##    for i in range(1):
#    model.fit(trainX, trainY, epochs=1, batch_size=batch_size, verbose=2, shuffle=False)
#    model.reset_states()
#    i = i + 1
#    print('#################',float(i)/len(train_windows)*100,'%')

model = Sequential()
#model.add(LSTM(50, batch_input_shape=(batch_size, window_size, train_windows[0].shape[2]), stateful=False))
#model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
#model.add(MaxPooling1D(pool_size=2))
model.add(LSTM(23, input_shape=(window_size, train_windows[0].shape[2]), stateful=False))
model.add(Dropout(0.2))
model.add(Dense(len(classes),  activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=[metrics.categorical_accuracy])            
print(model.summary())
run_time = 0
loop = 1
#max_samples = 100
for trainX, trainY in zip(train_windows, train_windows_labels):
    for i in range(loop):
#        if len(trainX) > max_samples:
#            trainX = np.array(random.sample(trainX.tolist(),max_samples))
#            trainY = np.array(random.sample(trainY.tolist(),max_samples))
#        model.fit(trainX, trainY, epochs=1, batch_size=batch_size, verbose=2, shuffle=False)
#        model.fit(trainX, trainY, epochs=1, batch_size=batch_size, shuffle=False)
        model.fit(trainX, trainY, epochs=4, shuffle=True)

#        model.reset_states()
        run_time = run_time + 1
        print(classes[np.argmax(np.array(trainY[0]))],'#################',run_time/(len(train_windows)*loop)*100,'%')
    


######################  
#Do on validation data
######################
test_windows, test_windows_labels, tmp = getWindowsAndLabels(VALIDATION_FILE_PATH,window_size=window_size)

correct = True
total_samples = 0
total_errors = 0
for testX, testY in zip(test_windows, test_windows_labels):
    
#    max_test_predicts = []
#    all_test_predicts = []
#    for model in models.values():
#        model.reset_states()
#        test_predicts = model.predict(testX, batch_size=batch_size)
#        all_test_predicts.append(test_predicts)
#        max_test_predicts.append(np.average(np.array([np.max(p) for p in test_predicts])))
##        max_test_predicts.append(np.argmax(np.bincount(np.array([np.argmax(p) for p in test_predicts]))))
#        
#    model = models[train_windows_labels[np.argmax(np.array(max_test_predicts))]]
        
     
    
#    if len(testX) > max_samples:
#        testX = np.array(random.sample(testX.tolist(),max_samples))
#        testY = np.array(random.sample(testY.tolist(),max_samples))
    
    # make predictions
    
    
    
#    model.reset_states()
#    test_predicts = model.predict(testX, batch_size=batch_size)
#    scores = model.evaluate(testX, testY, batch_size=batch_size, verbose=0)
    scores = model.evaluate(testX, testY, verbose=0)

    
#    max_test_predicts = np.array([np.argmax(p) for p in test_predicts])
    max_testY = np.array([np.argmax(y) for y in testY])
   
    total_samples = total_samples + len(testY)
#    total_errors = total_errors + sum(max_test_predicts != max_testY)
    
    print(classes[np.argmax(np.bincount(max_testY))], "Accuracy: %.2f%%" % (scores[1]*100))
    
#    if classes[np.argmax(np.bincount(max_test_predicts))] != classes[np.argmax(np.bincount(max_testY))]:
#        correct = False
#        print('ERROR Label =',classes[np.argmax(np.bincount(max_test_predicts))], 'True label =', classes[np.argmax(np.bincount(max_testY))])
#    else:
#        print('Label =',classes[np.argmax(np.bincount(max_test_predicts))], 'True label =', classes[np.argmax(np.bincount(max_testY))])
    
    
    

#print()
#if correct:
#    print('All Correct!')
#
#print('Error Rate:', float(total_errors)/total_samples*100, '%')
#print(total_errors, '/', total_samples)




#    t1= DataFrame(data=testY)
#    t2= DataFrame(data=testPredict)
#    x=pd.concat([t1,t2], ignore_index=True,  axis=1)
#    xt=DataFrame(data=x)
#    xt.to_csv("results.csv")






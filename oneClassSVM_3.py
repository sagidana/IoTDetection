# -*- coding: utf-8 -*-
"""
Created on Thu Jun 01 19:27:25 2017

@author: David
"""
from __future__ import print_function
import numpy as np
import pandas as pd 
from sklearn import preprocessing, cluster,svm
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from collections import Counter
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.feature_selection import SelectPercentile, VarianceThreshold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss,accuracy_score
import random


TEST_FILE_PATH = r"IoT_data_test.csv"
TRAINING_FILE_PATH = r"IoT_data_training_red.csv"
VALIDATION_FILE_PATH = r"IoT_data_validation_red.csv"

def trainOnData(samples,nu = 0.5, gamma='auto'):
    #Scale the features
#    samples = MaxAbsScaler().fit_transform(samples)
    
    #Create classifier
    clf = svm.OneClassSVM(kernel='rbf', gamma=gamma)
    clf.fit(samples)
    return clf

def testOnData(samples,classifier):
    #Scale the features
#    samples = MaxAbsScaler().fit_transform(samples)
    
    #Predict
    predicted_labels = classifier.predict(samples)
    
    #produce
    return predicted_labels

def reduceDataMatrixByClass(dataMatrix, class_list):
    all_samples = dataMatrix
    
    reduced_samples = []
     
    for sample in all_samples:
        label = sample[-3]
        if label in class_list:
            reduced_samples.append(sample)
    
    reduced_samples = np.array(reduced_samples)
    
    return reduced_samples

def reduceDataMatrixByMAC(dataMatrix, mac_list):
    all_samples = dataMatrix
    
    reduced_samples = []
     
    for sample in all_samples:
        mac = sample[-4]
        if mac in mac_list:
            reduced_samples.append(sample)
    
    reduced_samples = np.array(reduced_samples)
    
    return reduced_samples

def getMacList(dataMatrix):
    mac_list = dataMatrix[:,-4]
    return mac_list

def get_predict_score(predictes):
    #init to 0
    min_predicts = []
    for prediction_vector in predictes:
        num_of_anamoly = prediction_vector.tolist().count(-1)
        score = float(num_of_anamoly) / len(prediction_vector)
        min_predicts.append(score)
    return min_predicts

#Read train data
csv_data = pd.read_csv(TRAINING_FILE_PATH)
all_samples = csv_data.as_matrix()

#Get classes
classes = list(set(all_samples[:,-3]))

#classes = ['thermostat',
#classes = [ 'refrigerator, 'lights' ]
#           'baby_monitor',
#           'watch',
#           'smoke_detector',
#           'refrigerator',
#           'water_sensor',
#           'security_camera',
#           'motion_sensor',
#           'socket']

#Split samples by class
reduced_samples = []
for cur_class,i in zip(classes,range(len(classes))):
    reduced_samples.append(reduceDataMatrixByClass(all_samples,[cur_class]))
    #Remove complicated\overfitted(MAC) features
    reduced_samples[i] = reduced_samples[i][:,:-5]
#    reduced_samples[i] = VarianceThreshold().fit_transform( reduced_samples[i])


#Train classifer for each class    
clfs = []
for cur_class_samples in reduced_samples:
    clfs.append(trainOnData(cur_class_samples))

percentage_of_bad_samples = 0.1
for i in range(len(reduced_samples)):
    curr_samples = reduced_samples[i]
    num_of_bad_samples = int(len(curr_samples)*percentage_of_bad_samples)
    for j in range(len(reduced_samples)):
        if j != i:
            trainX = np.array(random.sample(trainX.tolist(),max_samples))
#exit()
######################
#Do on validation data
######################
csv_test_data = pd.read_csv(VALIDATION_FILE_PATH)
test_samples = csv_test_data.as_matrix()


#Get MACs
macs = set(test_samples[:,-4])

#Split samples by mac
reduced_test_samples = []
test_labels = []
for cur_mac,i in zip(macs,range(len(macs))):
    reduced_test_samples.append(reduceDataMatrixByMAC(test_samples,[cur_mac]))
    #Get labels
    test_labels.append(reduced_test_samples[i][0][-3])
    #Remove complicated\overfitted(MAC) features
    reduced_test_samples[i] = reduced_test_samples[i][:,:-5]
#    reduced_test_samples[i] = VarianceThreshold().fit_transform( reduced_test_samples[i])

    

#Predict each mac by all classifer on test
threshold = 0.1
all_predicts = [] 
for cur_mac_index,label in zip(reduced_test_samples,test_labels):
    predictes = []
    for clf in clfs:
        predictes = predictes + [testOnData(cur_mac_index,clf)]
    min_predicts = get_predict_score(predictes)
#    if label in classes:
#        for l,s in zip(test_labels,min_predicts):
#            print(l, str(s))
        
    class_index = min_predicts.index(min(min_predicts))
    all_predicts = all_predicts + [class_index]
    if label in classes:
        if label == classes[class_index]:
            print ('CORRECT! True label = ' + label + ' ' + 'Predict label = ' + classes[class_index])
        else:
            print ('True label = ' + label + ' ' + 'Predict label = ' + classes[class_index])
    #labels_test = reduced_test_samples[:,-3]



    
    #testOnData(reduced_test_samples,labels_test,clf)


#labels_true = all_samples[:,-3]

#Returns the count of each device
#print Counter(labels_true)

#Remove complicated\overfitted(MAC) features
#all_samples = all_samples[:,:-5]

#Scale the features
#all_samples = MinMaxScaler().fit_transform(all_samples)

#kmeans = KMeans(n_clusters=11).fit(all_samples)

#labels_pred = kmeans.labels_

#print Counter(labels_pred )


#labels_true = X[:,-4]
#le = preprocessing.LabelEncoder()
#labels_true = le.fit_transform(labels_true)

#X[:,-5] = le.fit_transform(X[:,-5])

#X = X[:,:-4]
#X = VarianceThreshold().fit_transform(X)

#plt.scatter(X_2D_spar[:,0],X_2D_spar[:,1],c=labels_true)


#plt.show()


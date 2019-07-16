#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 12:08:46 2017

@author: nhannguyen
"""
import pandas as pd
import os
import pickle, json
import numpy as np

import matplotlib.pyplot as plt
from datetime import date, timedelta
from sklearn.preprocessing import MinMaxScaler


def datetime_span(start, span):
    allDay = list()
    for i in range(span):
        allDay.append(start + timedelta(days=i))
    return allDay

def get_data(n_files=1):
    prefix = '../DATA/'
    input_files = [f for f in os.listdir(prefix) if f.startswith('Trace') and f.endswith('json')]
    if n_files is not None:
        input_files = input_files[:n_files]
    dict={}
    for f in input_files:
        print(prefix + f)
        with open(prefix +f) as file:
            dict.update(json.load(file))
            
    return dict

def readDATA(NoFile):

    dict= get_data(NoFile)
    fig = plt.figure(figsize=[8,5])    
    pdata ={}  
    
    thr = 400
    leg=[];
    count=0
    for kk, v in dict.items():
        start = date(v['uploadYear'],v['uploadMonth'],v['uploadDay'])
        
        lifetime = (date.today()-start).days
        if sum(v['dailyViewcount'][-500:])/len(v['dailyViewcount'][-500:]) > thr and v['totalView']> 1500000 and lifetime >700:
            
            pdata[kk] = pd.DataFrame( {
                    'ds': datetime_span(start,len(v['dailyViewcount'])),
                     'y': v['dailyViewcount']
                     })
            df = pdata[kk]
            plt.plot(df['ds'], df['y'],label = kk)
            count +=1
            if count < 10:
                leg.append(kk)
                
    leg.append('...')  
    leg.append('...') 
    leg.append('...') 
    leg.append(kk)     
    plt.xticks(rotation=45)#(rotation='vertical')
    #plt.xlabel('Time')
    plt.ylabel('Solicitaion')
    plt.legend(leg)
    plt.show()
    fig.savefig('RESULT/Pics/Traces.pdf',bbox_inches='tight')
    return pdata

def createWindow_dataset(dataset, lookBack=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-lookBack):
		a = dataset[i:(i+lookBack), 1]
		dataX.append(a)
		dataY.append(dataset[i + lookBack, 1])
	return np.array(dataX), np.array(dataY)

def DataSplit(dataset, trainSize, testSize, validSize, lookBack):
    #print(dataset.shape)
    tr_s = max(-400 - trainSize, -dataset.shape[0]) 
    tr_e = min(tr_s + trainSize, -testSize-validSize-2*lookBack)
    t_s = tr_e 
    t_e = t_s + testSize + lookBack
    v_s = t_e - lookBack
    v_e = v_s + validSize + lookBack
    return dataset[tr_s:tr_e,:],dataset[t_s:t_e,:],dataset[v_s:v_e,:]
#
#def DataSplit(dataset, trainSize, testSize, validSize, lookBack):
#    #print(dataset.shape)
##    tr_s = max(-400 - trainSize, -dataset.shape[0]) 
##    tr_e = min(tr_s + trainSize, -testSize-validSize-2*lookBack)
##    t_s = tr_e - lookBack
##    t_e = t_s + testSize + lookBack
##    v_s = t_e 
##    v_e = v_s + validSize + lookBack
#    return dataset[0:-testSize-validSize-lookBack,:],\
#            dataset[-testSize-validSize-2*lookBack:-validSize-2*lookBack,:],\
#            dataset[-validSize-lookBack:,:]

def Data4Linear(dataset,lookBack,trainSize = 300, testSize=30,validSize=14):
    
    train, test, valid = DataSplit(dataset, trainSize, testSize, validSize, lookBack)
    X_train, y_train = createWindow_dataset(train, lookBack)
    X_test, y_test = createWindow_dataset(test, lookBack)
    X_valid, y_valid = createWindow_dataset(valid, lookBack)
    return X_train, y_train, X_test, y_test,X_valid, y_valid


def Data4NN(dataset,lookBack,trainSize = 300, testSize=30,validSize=14):
#    scaler = MinMaxScaler(feature_range=(0,1))
#    data = dataset.copy()
#    data_scale = scaler.fit_transform(data[:,1].reshape(-1,1).astype('float64'))
#    data[:,1]=data_scale.flatten()
    
    scaler = MinMaxScaler(feature_range=(0,1))
    data = dataset.copy()
    data_scale = scaler.fit_transform(data[:,1].reshape(-1,1).astype('float64'))
    data[:,1]=data_scale.flatten()
    
    trainSet, testSet, validSet = DataSplit(data, trainSize, testSize, validSize, lookBack)
    #print(testSet.shape)
    trainX, trainY = createWindow_dataset(trainSet, lookBack)
    testX, testY = createWindow_dataset(testSet, lookBack)
    validX, validY = createWindow_dataset(validSet, lookBack)
    
    trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    #print(testX.shape)
    testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
    #print(validX.shape)
    validX = np.reshape(validX, (validX.shape[0], 1, validX.shape[1]))
    
    return trainX, trainY, testX, testY, validX, validY, scaler

def createSaveDict(dataset, lbacklist, trainSize, testSize,validSize):
    saveNN={}
    saveLinear={}
    for lookBack in lbacklist:
        saveLinear[str(lookBack)]={}
        saveNN[str(lookBack)]={}
                
        X_train, y_train, X_test, y_test,X_valid, y_valid =  \
            Data4Linear(dataset,lookBack,trainSize, testSize,validSize)
        saveLinear[str(lookBack)]={u"X_train":X_train,u"y_train": y_train, \
                   u"X_test":X_test, u"y_test": y_test,\
                   u"X_valid":X_valid, u"y_valid": y_valid}
        data = dataset.copy()
        trainX, trainY, testX, testY, validX, validY, scaler = \
            Data4NN(data,lookBack,trainSize, testSize,validSize)
        saveNN[str(lookBack)]={u"trainX":trainX,u"trainY": trainY, \
                   u"testX":testX, u"testY": testY, \
                   u"validX":validX, u"validY": validY, u"scaler":scaler}
    return saveNN, saveLinear
def main():

    pdata = readDATA(100)
    listID = [f for f in pdata]
    
    with open('SimDATA/listID.json', 'w') as f:
        json.dump(listID, f)
    
    trainSize = 200
    testSize = 30
    validSize = 14
    
    lbacklist = [1,2, 3, 5, 7, 14, 30]
    
    count =0
    for key in pdata:
        dataset = pdata[key].values.copy()
        saveNN, saveLinear = createSaveDict(dataset, lbacklist, trainSize, testSize,validSize)
        
        outName='SimDATA/DataSim_ID_{0}_train{1}_test{2}_valid{3}.pkl'.format(key,trainSize, testSize,validSize)
        with open(outName,'wb') as file:
            pickle.dump([saveNN,saveLinear],file)
        count +=1
        print("Trace {0} : {1}".format(count,key))
#    outNameLinear='SimDATA/'+ 'Data4Linear_ID_' + key  + '_tsz' + str(testSize)+'.pkl'
#    with open(outNameLinear,'w') as file:
#        pickle.dump(saveLinear,file)

if __name__=="__main__":
    main()
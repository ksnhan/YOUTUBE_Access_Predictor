#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 10:23:13 2018

@author: nhannguyen
"""
from IPython import get_ipython
get_ipython().magic('reset -sf') 

import os
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import json
import time
import numpy as np

def mae_diff(a,b):
    import numpy as np
    denominator = np.abs(a) + np.abs(b)
    with np.errstate(divide='ignore', invalid='ignore'):
        d = np.true_divide(np.abs(a - b), denominator)
        d[denominator == 0] = 0
    return np.mean(np.abs(a-b)), np.mean(d)*100

def extractKey(f):
    startID_pos = f.find('_ID_') + 4
    endID_pos = f.find('_train')
    startTag = f.find('_ID_')
    endTag = f.find('.pkl')
    key = f[startID_pos:endID_pos]
    tag = f[startTag:endTag]
    return key, tag



def SES(x,alpha):
    xIn = x.copy()
    cum=1
    if cum :
        xIn = np.cumsum(xIn.copy(),axis=1)
        
    start_time = time.time()
    testSize,wLength = xIn.shape
    pOut = []
    for i in range(testSize):
        tOut = []
        for k in range(wLength):
            if k == 0:
                tOut.append(xIn[i,k])
            else:
                tOut.append(alpha*xIn[i,k]+(1-alpha)*tOut[k-1])
        
        if cum :
            if wLength > 1:
                tOut = np.diff(tOut.copy())
        
        
        pOut.append(tOut[-1])
    run_time =  time.time()- start_time
    return pOut,run_time
        
def DES(x,alpha):
    xIn = x.copy()
    cum=1
    if cum :
        xIn = np.cumsum(xIn.copy(),axis=1)
    #print(xIn)
    start_time = time.time()
    testSize,wLength = xIn.shape
    pOut = []
    for i in range(testSize):
        S1 = []
        S2 = []
        L =[]
        T =[]
        pEst=[]
        
        for k in range(wLength):
            if k == 0:
                S1.append(xIn[i,k])
                S2.append(xIn[i,k])
            else:
                S1.append(alpha*xIn[i,k]+(1-alpha)*S1[k-1])
                S2.append(alpha*S1[k]+(1-alpha)*S2[k-1])
            L.append(2 *S1[k] -S2[k])
            T.append(alpha/(1-alpha)*(S1[k]-S2[k]))
            
            pEst.append(L[k]+T[k])
        
        #print(pEst)
        if cum :
            if wLength > 1:
                pEst = np.diff(pEst.copy())
        #print(pEst)
        pOut.append(pEst[-1]) 
    run_time =  time.time()- start_time
    return pOut,run_time                    
                
    

prefix = 'SimDATA'
input_files = [f for f in os.listdir(prefix) if f.startswith('DataSim') and f.endswith('.pkl')]

outExpSmooth = 'RESULT/JSON_FF/ResultExpSmooth_'

count = 0
for f in input_files:
    count += 1
    print(f + ' : ' +str(count) +'/' +str(len(input_files)))
    key, test_size = extractKey(f)
    with open(prefix+ '/' + f,'rb') as file:
            saveNN,saveLinear = pickle.load(file)
            
    saveResultES ={}
    saveResultES['SES'] = {}
    saveResultES['DES'] = {}
    for w in saveLinear:
        xIn_train = saveLinear[w]['X_train'].copy()
        xIn_test = saveLinear[w]['X_test'].copy()
        xIn_valid = saveLinear[w]['X_valid'].copy()
        yOut_train = saveLinear[w]['y_train'].copy()
        yOut_test = saveLinear[w]['y_test'].copy()
        yOut_valid = saveLinear[w]['y_valid'].copy()
        
        pOut_DES_train, run_time_DES_train = DES(xIn_train,0.7)
        pOut_DES_test, run_time_DES_test = DES(xIn_test,0.7)
        pOut_DES_valid, run_time_DES_valid = DES(xIn_valid,0.7)
        
        pOut_SES_train, run_time_SES_train = SES(xIn_train,0.7)
        pOut_SES_test, run_time_SES_test = SES(xIn_test,0.7)
        pOut_SES_valid, run_time_SES_valid = SES(xIn_valid,0.7)
#        plt.plot(yOut, label = key,color='red',linestyle='-')
#        plt.plot(pOut_SES, label = 'SES', color='green',linestyle=':')
#        plt.plot(pOut_DES, label = 'DES',color='blue',linestyle='--')
#        
#        plt.legend()
#        plt.show()
        
        mae_train_SES, diff_train_SES = mae_diff(yOut_train,pOut_SES_train)
        mse_train_SES = mean_squared_error(yOut_train,pOut_SES_train)
        
        mae_test_SES, diff_test_SES = mae_diff(yOut_test,pOut_SES_test)
        mse_test_SES = mean_squared_error(yOut_test,pOut_SES_test)
        
        mae_valid_SES, diff_valid_SES = mae_diff(yOut_valid,pOut_SES_valid)
        mse_valid_SES = mean_squared_error(yOut_valid,pOut_SES_valid)
       
        
        mae_train_DES, diff_train_DES = mae_diff(yOut_train,pOut_DES_train)
        mse_train_DES = mean_squared_error(yOut_train,pOut_DES_train)
        
        mae_test_DES, diff_test_DES = mae_diff(yOut_test,pOut_DES_test)
        mse_test_DES = mean_squared_error(yOut_test,pOut_DES_test)
        
        mae_valid_DES, diff_valid_DES = mae_diff(yOut_valid,pOut_DES_valid)
        mse_valid_DES = mean_squared_error(yOut_valid,pOut_DES_valid)
        
        resultSES = {'mse_test':mse_test_SES, 'mae_test':mae_test_SES, 'diff_test':diff_test_SES,
                     'mse_valid':mse_valid_SES, 'mae_valid':mae_valid_SES, 'diff_valid':diff_valid_SES,
                     'mse_train':mse_train_SES, 'mae_train':mae_train_SES, 'diff_train':diff_train_SES,
                     'run_time': run_time_SES_train+run_time_SES_test}
        resultDES = {'mse_test':mse_test_DES, 'mae_test':mae_test_DES, 'diff_test':diff_test_DES,
                     'mse_valid':mse_valid_DES, 'mae_valid':mae_valid_DES, 'diff_valid':diff_valid_DES,
                     'mse_train':mse_train_DES, 'mae_train':mae_train_DES, 'diff_train':diff_train_DES,
                     'run_time': run_time_DES_train + run_time_DES_test}
        saveResultES['SES'][w]=resultSES
        saveResultES['DES'][w]=resultDES
        print("w = {0} , MAEP_SES = {1}, MAEP_DES = {2}".format(w, diff_test_SES,diff_test_DES))
    
    with open(outExpSmooth + f[:-4] + '.json','w') as savefile:
        json.dump(saveResultES,savefile)
    print('Save ES:' + key +' ==')
        
        
        
#        print(key, xIn)
#        print(yOut)
        
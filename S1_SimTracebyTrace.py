#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 13:37:34 2017

@author: nhannguyen
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import pickle
import json
import time
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error



from IPython import get_ipython


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

#%%
def Linear(X_train, y_train, X_test, X_valid, look_back=7, method = 'DecisionTree'):
    
    start_time = time.time()
    rng = np.random.RandomState(1)
    if method == 'DecisionTree':
        m_DecisionTree =  DecisionTreeRegressor(max_depth=4)
        m_DecisionTree.fit(X_train, y_train)
        p_valid = m_DecisionTree.predict(X_valid)
        p_test = m_DecisionTree.predict(X_test)
        p_train = m_DecisionTree.predict(X_train)
    elif method == 'AdaBoost':
        m_AdaBoost = AdaBoostRegressor(DecisionTreeRegressor(max_depth=4),\
                                       n_estimators=300, random_state=rng)
        m_AdaBoost.fit(X_train, y_train)
        p_valid = m_AdaBoost.predict(X_valid)
        p_test = m_AdaBoost.predict(X_test)
        p_train = m_AdaBoost.predict(X_train)
    elif method == 'GradientBoosting':
        params = {'n_estimators': 300, 'max_depth': 4, 'min_samples_split': 2,\
                  'learning_rate': 0.01, 'loss': 'ls'}
        m_GradientBoosting =GradientBoostingRegressor(**params)
        m_GradientBoosting.fit(X_train, y_train)
        p_valid = m_GradientBoosting.predict(X_valid)
        p_test = m_GradientBoosting.predict(X_test)
        p_train = m_GradientBoosting.predict(X_train)
    else:        
        m_RandomForest = RandomForestRegressor(n_estimators=20, max_depth= 4, random_state=0)
        m_RandomForest.fit(X_train, y_train)
        p_valid = m_RandomForest.predict(X_valid)
        p_test = m_RandomForest.predict(X_test)
        p_train = m_RandomForest.predict(X_train)
    run_time =  time.time()- start_time
    return p_valid, p_test, p_train, run_time
#%%

def NeuralNet(trainX, trainY, testX, validX, look_back=7, unitsLayer1=7, unitsLayer2=0, unitType ='LSTM'):
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.layers import GRU, LSTM
    
    noEpoch = 200
    bt_size = 30
    import time
    start_time = time.time()
    model = Sequential()
    if unitType == 'GRU':
        if  unitsLayer2==0:   
            model.add(GRU(units =unitsLayer1,  input_shape = (1, look_back)))
        else:
            model.add(GRU(units = unitsLayer1,  return_sequences = True, input_shape = (1,look_back)))
        if unitsLayer2 !=0:
            model.add(GRU(units = unitsLayer2))
    else:
        if  unitsLayer2==0:   
            model.add(LSTM(units =unitsLayer1,  input_shape = (1, look_back)))
        else:
            model.add(LSTM(units = unitsLayer1,  return_sequences = True, input_shape = (1,look_back)))
        if unitsLayer2 !=0:
            model.add(LSTM(units = unitsLayer2))
    
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer = 'rmsprop',  metrics=['accuracy'])
    model.fit(trainX, trainY, epochs = noEpoch, batch_size = bt_size, verbose=0)
    #model.summary()
    
    p_test = model.predict(testX)
    p_valid = model.predict(validX)
    p_train = model.predict(trainX)
    run_time =  time.time()- start_time
    
    del model
    
    return  p_valid, p_test, p_train, run_time
#%%
def saveEvaluation(y_train,p_train,y_test,p_test,y_valid,p_valid):    
    mae_test, diff_test = mae_diff(y_test,p_test)
    mae_train, diff_train = mae_diff(y_train,p_train)  
    mae_valid, diff_valid = mae_diff(y_valid,p_valid)  
    
    mse_test = mean_squared_error(y_test,p_test)
    mse_train = mean_squared_error(y_train,p_train) 
    mse_valid = mean_squared_error(y_valid,p_valid)
    
    result = {'mse_valid':mse_valid,'mse_test':mse_test,'mse_train': mse_train, \
              'mae_valid':mae_valid,'mae_test':mae_test, 'mae_train':mae_train,\
              'diff_valid':diff_valid, 'diff_test':diff_test,  'diff_train':diff_train}
              
    return result

#%%
def EvalLinear(X_train, y_train, X_test, y_test, X_valid, y_valid, \
               look_back, method = 'DecisionTree'):
    pValid, pTest, pTrain, run_time = \
                Linear(X_train, y_train, X_test, X_valid, look_back, method)
            
    result = saveEvaluation \
            (y_train, pTrain, y_test, pTest, y_valid, pValid)
    result['run_time'] =  run_time       
    
    return result

#%%
def runLinear(tag,key,saveLinear,listLinear,outLinear):

    ##############################
    saveResultLinear ={}
    for method in listLinear: 
        saveResultLinear[method] = {}
        for look_back in saveLinear:
            result = EvalLinear(\
                    saveLinear[look_back]['X_train'], saveLinear[look_back]['y_train'], 
                    saveLinear[look_back]['X_test'], saveLinear[look_back]['y_test'], 
                    saveLinear[look_back]['X_valid'], saveLinear[look_back]['y_valid'], 
                               int(look_back), method)
            saveResultLinear[method][look_back]=result
        print('* Finish ' + method)
    with open(outLinear + tag + '.json','w') as savefile:
        json.dump(saveResultLinear,savefile)
        
    print('Save Linear:' + tag +' ==')
    del result, saveResultLinear,  savefile

#%%
def EvalNN(X_train, y_train, X_test, y_test, X_valid, y_valid, \
           scaler, look_back, unitsLayer1=7, unitsLayer2=0, unitType ='LSTM'):
    
    pValid, pTest, pTrain, run_time = \
                    NeuralNet(X_train, y_train, X_test, X_valid, \
                    look_back, unitsLayer1, unitsLayer2, unitType)         
    
    pValid = scaler.inverse_transform(pValid)
    pTest = scaler.inverse_transform(pTest)
    pTrain = scaler.inverse_transform(pTrain)
    
    y_valid = scaler.inverse_transform(y_valid.reshape(-1,1))
    y_test = scaler.inverse_transform(y_test.reshape(-1,1))
    y_train = scaler.inverse_transform( y_train.reshape(-1,1))
            
    result = saveEvaluation (y_train, pTrain, y_test ,pTest, y_valid, pValid) 
    result['run_time'] =  run_time        
    
#    import matplotlib.pyplot as plt
#    plt.plot(pTest, label='predict')
#    plt.legend()
#    plt.show()
#    plt.plot(y_test, label='real')
#    plt.legend()
#    plt.show()
    return result
        
#%%
def runLSTM(tag,key,saveNN,outNN,NNconfig):

    ############################################
    saveResultLSTM ={}
    for method in NNconfig: 
        saveResultLSTM[method] = {}
        for look_back in saveNN:
            resultLSTM = EvalNN(\
                    saveNN[look_back]['trainX'], saveNN[look_back]['trainY'], \
                    saveNN[look_back]['testX'],saveNN[look_back]['testY'], \
                    saveNN[look_back]['validX'],saveNN[look_back]['validY'],\
                    saveNN[look_back]['scaler'], int(look_back), \
                    NNconfig[method]['unitsLayer1'], NNconfig[method]['unitsLayer2'],\
                    unitType ='LSTM') 
                        
            saveResultLSTM[method][look_back]=resultLSTM
            
            print('* LSTM config_{0}_lb{1} - train maep: {2} - test maep: {3} - valid maep: {4}'.\
                  format(method,look_back,resultLSTM['diff_train'],resultLSTM['diff_test'],resultLSTM['diff_valid']))
            
    with open(outNN + 'LSTM_' + tag + '.json','w') as savefile:
        json.dump(saveResultLSTM,savefile)
    print('Save LSTM:' + key +' ==')
    del resultLSTM, saveResultLSTM,  savefile
#%%
def runGRU(tag,key,saveNN,outNN,NNconfig):
   
    ############################################
    saveResultGRU = {}
    for method in NNconfig: 
        saveResultGRU[method] = {}
        for look_back in saveNN:
            resultGRU = EvalNN(saveNN[look_back]['trainX'], saveNN[look_back]['trainY'], 
                                         saveNN[look_back]['testX'],saveNN[look_back]['testY'], 
                                         saveNN[look_back]['validX'],saveNN[look_back]['validY'], 
                                         saveNN[look_back]['scaler'], int(look_back), 
                                         NNconfig[method]['unitsLayer1'], NNconfig[method]['unitsLayer2'],
                                         unitType ='GRU') 
            saveResultGRU[method][look_back]=resultGRU 
            print('* GRU config_{0}_lb{1} - train maep: {2} - test maep: {3} - valid maep: {4}'.\
                  format(method,look_back,resultGRU['diff_train'],resultGRU['diff_test'],resultGRU['diff_valid']))
    with open(outNN + 'GRU_' + tag + '.json','w') as savefile:
        json.dump(saveResultGRU,savefile)
    print('Save GRU:' + key +' ==')
    del resultGRU, saveResultGRU,  savefile

   

   


def main():    
    prefix = 'SimDATA'
    listLinear=['DecisionTree', 'AdaBoost', 'GradientBoosting', 'RandomForest']
    NNconfig={\
                  'C1':{'unitsLayer1':1, 'unitsLayer2':0},\
                  'C2':{'unitsLayer1':7, 'unitsLayer2':0},\
                  'C3':{'unitsLayer1':14, 'unitsLayer2':0},\
                  'C4':{'unitsLayer1':1, 'unitsLayer2':2},\
                  'C5':{'unitsLayer1':7, 'unitsLayer2':14},\
                  'C6':{'unitsLayer1':14, 'unitsLayer2':28} }
    
    outNN = 'RESULT/JSON_FF/ResultNN_'
    outLinear = 'RESULT/JSON_FF/ResultLinear'

    
    input_files = [f for f in os.listdir(prefix) if f.startswith('DataSim') and f.endswith('.pkl')]
    count =0
    dem = 2
#    for f in input_files:
    for f in input_files:
        count+=1
        print(f + ' : ' +str(count) +'/' +str(len(input_files)))
        
        
        key, tag = extractKey(f)
        with open(prefix+ '/' + f,'rb') as file:
            saveNN,saveLinear = pickle.load(file)
        
        
        if os.path.isfile('RESULT/JSON_FF/ResultNN_GRU_' + tag + '.json'):
            print('Already simulated')
        else:   
            runLinear(tag,key,saveLinear,listLinear,outLinear)
#            tempo(80);
#            print (key)
#            print (tag)
            #print(saveNN)
            runLSTM(tag,key,saveNN,outNN,NNconfig)
#            tempo(80);
            runGRU(tag,key,saveNN,outNN,NNconfig)
#            tempo(80);
            
            dem =dem +1            
        if dem==3:
            break
#            get_ipython().magic('reset -sf')
#            print("Reset Console")
#            os.execv(__file__, sys.argv)  # Run a new iteration of the current script, providing any command line args from the current iteration.

            
        del saveNN,saveLinear

         
if __name__ == "__main__":
    st = time.time()
    main()
    print(time.time()-st)
    get_ipython().magic('reset -sf')
    exit()
    

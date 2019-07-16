#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 13:37:34 2017

@author: nhannguyen
"""
from IPython import get_ipython
get_ipython().magic('reset -sf')
import json
import os
import csv
import pandas as pd

def delColCSV(fname_in,fname_out):
    with open(fname_in, 'r') as fin, open(fname_out, 'w') as fout:
        reader = csv.reader(fin)
        writer = csv.writer(fout)
        for row in reader:
            writer.writerow(row[1:])
            

def extractKey(f):
    startID_pos = f.find('_ID_') + 4
    endID_pos = f.find('_train')
    startTag = f.find('_ID_')
    endTag = f.find('.pkl')
    key = f[startID_pos:endID_pos]
    tag = f[startTag:endTag]
    return key, tag

def loadSaveACC(fileList,directory):
    save={}
    for f in fileList:
        key, test_size = extractKey(f)
        save[key]={}
        with open(directory+f,'r') as file:
            save[key].update(json.load(file))
    return save

def CreateCols(save,method):
    col = list()
    key0=list(save)[0]
    config0=list(save[key0])[0]
    
    for i in list(save[key0]):
        for j in list(save[key0][config0]):
            col.append(method + i +'_LB' + j)
            print(method + i +'_LB' + j)
    return col
        
def CreateLines(save,method,mesKey):
    line = list()
    key0=list(save)
    config0=list(save[key0])[0]
    
    for i in list(save[key0]):
        for j in list(save[key0][config0]):
            col.append(method + i +'_LB' + j)
            print(method + i +'_LB' + j)
    return col


def listAppend(save,mesKey):
    out = list()
    for conf in list(save):
        for lb in list(save[conf]):
            out.append(save[conf][lb][mesKey])
    return out

NNconfig={\
          'C1':{'unitsLayer1':1, 'unitsLayer2':0},\
          'C2':{'unitsLayer1':7, 'unitsLayer2':0},\
          'C3':{'unitsLayer1':14, 'unitsLayer2':0},\
          'C4':{'unitsLayer1':1, 'unitsLayer2':2},\
          'C5':{'unitsLayer1':7, 'unitsLayer2':14},\
          'C6':{'unitsLayer1':14, 'unitsLayer2':28} }

listLinear=['DecisionTree', 'AdaBoost', 'GradientBoosting', 'RandomForest']

listES = ['DES']#['SES','DES']

prefix = 'RESULT/JSON_FF_fe/'

resultNN_LSTM_files = [f for f in os.listdir(prefix) if f.startswith('ResultNN_LSTM_') and f.endswith('.json')]
resultNN_GRU_files = [f for f in os.listdir(prefix) if f.startswith('ResultNN_GRU_') and f.endswith('.json')]
resultLinear_files = [f for f in os.listdir(prefix) if f.startswith('ResultLinear_') and f.endswith('.json')]
resultES_files = [f for f in os.listdir(prefix) if f.startswith('ResultExpSmooth_') and f.endswith('.json')]


saveNN_LSTM = loadSaveACC(resultNN_LSTM_files,prefix)
saveNN_GRU = loadSaveACC(resultNN_GRU_files,prefix)
saveLinear = loadSaveACC(resultLinear_files,prefix)
saveES = loadSaveACC(resultES_files,prefix)

Col1 = CreateCols(saveNN_GRU,'GRU_')
Col2 = CreateCols(saveNN_LSTM,'LSTM_')
Col3 = CreateCols(saveLinear,'')
Col4 = CreateCols(saveES,'')

#Cols=['Video_ID']+Col4 + Col3+Col2+Col1
Cols=['Video_ID']+ Col3+Col2+Col1





for mesKey in  {'diff_valid', 'diff_test','run_time', 'diff_train' } :# {'mse_test', 'mse_valid', 'mae_test','mae_train'} : # #'mse_test'
    df = pd.DataFrame(columns = Cols)
    
    for kk in list(saveLinear):
        val=list()
        val.append(kk)
#        val = val + listAppend(saveES[kk],mesKey) + listAppend(saveLinear[kk],mesKey) \
#        + listAppend(saveNN_LSTM[kk],mesKey) + listAppend(saveNN_GRU[kk],mesKey)
        
        val = val + listAppend(saveLinear[kk],mesKey) \
        + listAppend(saveNN_LSTM[kk],mesKey) + listAppend(saveNN_GRU[kk],mesKey)
        
        df = df.append(pd.DataFrame([val],columns = Cols), ignore_index=True)
    
        
    
    df.to_csv('RESULT/CSV/temp.csv')  
    
    delColCSV('RESULT/CSV/temp.csv','RESULT/CSV/Result_Algs_withES_'+mesKey+'.csv')






#import csv
#with open('Result_Algs.csv','w') as csvF:
##    writer = csv.writer(csvF, delimiter=',')
##    csvF.writerows(Cols)
##    csvF.write('\n')
##    
#    wr = csv.writer(csvF, dialect='excel')
#    wr.writerows(Cols)

#listMethod ={'1' : 'DecisionTree',\
#             '2' : 'AdaBoost',\
#             '3' : 'GradientBoosting',\
#             '4' : 'RandomForest',\
##             '5' : 'LSTM1',\
##             '6' : 'LSTM7',\
##             '7' : 'LSTM14',\
##             '8' : 'LSTM1_2',\
##             '9' : 'LSTM7_14',\
##             '10' : 'LSTM14_28',\
##             '11' : 'GRU1',\
##             '12' : 'GRU7',\
##             '13' : 'GRU14',\
##             '14' : 'GRU1_2',\
##             '15' : 'GRU7_14',\
##             '16' : 'GRU14_28'\
#             }
#metric ='diff_test'
#line1 = ''
#for i in listMethod:
#    for look_back in saveDecisionTree:
#        line1 += ',' + listMethod[i] +'_LB'+ str(look_back)
#with open('Result_'+metric+'.csv','w') as csvF:
##    writer = csv.writer(csvF, delimiter=',')
#    csvF.write(line1)
#    csvF.write('\n')
#    for kk in  saveDecisionTree['1']:
#        line={}
#        line = kk 
#        for i in listMethod:
#            for lb in saveDecisionTree:
#                line += ',' + eval('str(save'+ listMethod[i] + '[lb][kk]["'+metric+'"])')
#        print(line)
#        csvF.write(line)
#        csvF.write('\n')


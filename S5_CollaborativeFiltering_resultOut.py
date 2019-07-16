#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 12:40:54 2017

@author: nhannguyen
"""
from IPython import get_ipython
get_ipython().magic('reset -sf') 

# --- Import Libraries --- #
 
import pandas as pd
from scipy.spatial.distance import cosine
import numpy as np
import json
import matplotlib.pyplot as plt
import numpy as np


def extractCsv(keySet):
    filecsv = 'RESULT/CSV/Result_Algs_withES_' + keySet + '.csv'
    data = pd.read_csv (filecsv)
    if keySet == 'run_time':
        data_drop = data.drop('Video_ID', 1)
        return data_drop.sum(axis=0)
    else:
        data.iloc[:,1:] = 100 - data.iloc[:,1:]
        data_drop = data.drop('Video_ID', 1)
        return data_drop.mean(axis=0)

def getValidation(meanTest,meanValid,iStart,iEnd):
    topName = np.argmax(meanTest[iStart:iEnd])
    topTest = max(meanTest[iStart:iEnd])
    Valid = meanValid[topName]
    return topName,topTest,Valid


def readDict(dictIn):
    seriesOut = []
    TestACC = []
    ValidACC = []
    runtime = []
    for i in dictIn:
        seriesOut.append(float(i))
        TestACC.append(dictIn[i]['TestACC'])
        ValidACC.append(dictIn[i]['ValidACC'])
        runtime.append(dictIn[i]['runtime'])
    MaxPropose_TestACC = dictIn[i]["MaxPropose_TestACC"]
    MaxPropose_ValidACC = dictIn[i]["MaxPropose_ValidACC"]
    Full_runtime =  dictIn[i]["Full_runtime"]
    return seriesOut,TestACC,ValidACC,runtime,\
            MaxPropose_TestACC,MaxPropose_ValidACC, Full_runtime
    

if __name__ == "__main__":
    meanTest = extractCsv('diff_test')
    meanValid = extractCsv('diff_valid')
    sumRunTime = extractCsv('run_time')
    
    DT_name,DT_test,DT_val = getValidation(meanTest,meanValid,0,7)
    Ada_name,Ada_test,Ada_val = getValidation(meanTest,meanValid,7,14)
    Grad_name,Grad_test,Grad_val = getValidation(meanTest,meanValid,14,21)
    Ran_name,Ran_test,Ran_val = getValidation(meanTest,meanValid,21,28)
    LSTM_name,LSTM_test,LSTM_val = getValidation(meanTest,meanValid,28,70)
    GRU_name,GRU_test,GRU_val = getValidation(meanTest,meanValid,70,112)
    
    a = np.arange(0,113,7)
    method=['DT','Ada','Grad','Ran',
            'LSTM_C1','LSTM_C2','LSTM_C3','LSTM_C4','LSTM_C5','LSTM_C6',
            'GRU_C1','GRU_C2','GRU_C3','GRU_C4','GRU_C5','GRU_C6']
    for i in range(len(a)):
        if i < len(a)-2:
            exec(method[i]+'_runtime = sumRunTime[a[i]:a[i+1]]')
    
#%%
    
    with open('RESULT/CF_JSON/30/ProposedCF_Result_Top.json','r') as f:
        dict_Top_Mod= json.load(f)
        
    Top_Mod_seriesOut,Top_Mod_TestACC,Top_Mod_ValidACC,Top_Mod_runtime, \
    Top_Mod_MaxPropose_TestACC,Top_Mod_MaxPropose_ValidACC, Top_Mod_Full_runtime \
        = readDict(dict_Top_Mod)
        
    with open('RESULT/CF_JSON/30/ProposedCF_Result_Ratio.json','r') as f:
        dict_percent_Mod= json.load(f)
    percent_Mod_seriesOut,percent_Mod_TestACC,percent_Mod_ValidACC,percent_Mod_runtime, \
    percent_Mod_MaxPropose_TestACC,percent_Mod_MaxPropose_ValidACC, percent_Mod_Full_runtime \
    = readDict(dict_percent_Mod)
    
    
    with open('RESULT/CF_JSON/30/OriginalCF_Result_Top.json','r') as f:
        dict_Top_Org= json.load(f)
    Top_Org_seriesOut,Top_Org_TestACC,Top_Org_ValidACC,Top_Org_runtime, \
    Top_Org_MaxPropose_TestACC,Top_Org_MaxPropose_ValidACC, Top_Org_Full_runtime \
    = readDict(dict_Top_Org)    
    with open('RESULT/CF_JSON/30/OriginalCF_Result_Ratio.json','r') as f:
        dict_percent_Org= json.load(f)
    percent_Org_seriesOut,percent_Org_TestACC,percent_Org_ValidACC,percent_Org_runtime, \
    percent_Org_MaxPropose_TestACC,percent_Org_MaxPropose_ValidACC, percent_Org_Full_runtime \
    = readDict(dict_percent_Org)
        

    
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    lStyles = ['_', '-.', '--', ':','-']
    mar = ['o','s','v','>','^','d','p','h']
    
    
    prefixPic = 'RESULT/Pics/'
    
    leg =['IBCF-RPM','Original Top-K CF','LSTM_C3 - W=3','GRU_C2 - W=1', 
               'DecisionTree - W=1','RandomForest - W=1','AdaBoost - W=1','GradientBoosting-W=1',
               'Best recom.-Full info.']
    
    #%%
    
    
    
    fig=plt.figure(figsize=(7, 6))
    
    plt.plot(Top_Mod_seriesOut,Top_Mod_TestACC,label = 'IBCF-RPM',
             color=colors[0],marker=mar[0], markersize=10, fillstyle='none')
    plt.plot(Top_Org_seriesOut,Top_Org_TestACC,label = 'Original CF',
             color=colors[2],marker=mar[1], markersize=10, fillstyle='none')
        
    
    params = { 'alpha':0.5,'linestyle':lStyles[1] ,'color':colors[3]}
    plt.axhline(y=LSTM_test, label = LSTM_name ,**params)
    params = { 'alpha':0.5,'linestyle':lStyles[2] ,'color':colors[4]}
    plt.axhline(y=GRU_test,  label = GRU_name , **params)
    
    params = { 'alpha':0.5,'linestyle':lStyles[1] ,'color':colors[1]}
    plt.axhline(y=DT_test,label = DT_name ,**params)
    params = { 'alpha':0.5,'linestyle':lStyles[4] ,'color':colors[5]}
    plt.axhline(y=Ran_test, label = Ran_name , **params)
    params = { 'alpha':0.5,'linestyle':lStyles[2] ,'color':colors[2]}
    plt.axhline(y=Ada_test, label = Ada_name , **params)
    params = { 'alpha':0.5,'linestyle':lStyles[3] ,'color':colors[6]}
    plt.axhline(y=Grad_test, label = Grad_name , **params)
    params = { 'alpha':0.5,'linestyle':lStyles[4] ,'color':colors[2]}
    #plt.axhline(y=dictNo['MaxPropose_ValidACC_Top'][-1], label = 'Max Recommend' , **params)
    plt.axhline(y=Top_Mod_MaxPropose_TestACC, label = 'Best - Full info.' , **params)

    plt.xlabel('Top K')
    plt.ylabel('Accuracy')
    plt.xticks( np.arange(21) )
    
    plt.legend(leg,loc='lower right')
    plt.show()
    
    fig.savefig(prefixPic+'top_K_Test.pdf',bbox_inches='tight')
    
    #%%
    fig=plt.figure(figsize=(7, 6))
    
    plt.plot(Top_Mod_seriesOut,Top_Mod_ValidACC,label = 'IBCF-RPM',
             color=colors[0],marker=mar[0], markersize=10, fillstyle='none')
    plt.plot(Top_Org_seriesOut,Top_Org_ValidACC,label = 'Original CF',
             color=colors[2],marker=mar[1], markersize=10, fillstyle='none')
        
    
    params = { 'alpha':0.5,'linestyle':lStyles[1] ,'color':colors[3]}
    plt.axhline(y=LSTM_val, label = LSTM_name ,**params)
    params = { 'alpha':0.5,'linestyle':lStyles[2] ,'color':colors[4]}
    plt.axhline(y=GRU_val,  label = GRU_name , **params)
    
    params = { 'alpha':0.5,'linestyle':lStyles[1] ,'color':colors[1]}
    plt.axhline(y=DT_val,label = DT_name ,**params)
    params = { 'alpha':0.5,'linestyle':lStyles[4] ,'color':colors[5]}
    plt.axhline(y=Ran_val, label = Ran_name , **params)
    params = { 'alpha':0.5,'linestyle':lStyles[2] ,'color':colors[2]}
    plt.axhline(y=Ada_val, label = Ada_name , **params)
    params = { 'alpha':0.5,'linestyle':lStyles[3] ,'color':colors[6]}
    plt.axhline(y=Grad_val, label = Grad_name , **params)
    params = { 'alpha':0.5,'linestyle':lStyles[4] ,'color':colors[2]}
    #plt.axhline(y=dictNo['MaxPropose_ValidACC_Top'][-1], label = 'Max Recommend' , **params)
    plt.axhline(y=Top_Mod_MaxPropose_ValidACC, label = 'Best - Full info.' , **params)

    plt.xlabel('Top K')
    plt.ylabel('Accuracy')
    plt.xticks( np.arange(21) )
    
    plt.legend(leg,loc='lower right')
    plt.show()
    
    fig.savefig(prefixPic+'top_K_Valid.pdf',bbox_inches='tight')
    

#%%
    
    

    
    fig=plt.figure(figsize=(7, 6))
    plt.plot(percent_Mod_seriesOut,percent_Mod_TestACC,label = 'IBCF-RPM',
             color=colors[0],marker=mar[0], markersize=10, fillstyle='none')
    plt.plot(percent_Org_seriesOut,percent_Org_TestACC,label = 'Original CF',
             color=colors[2],marker=mar[1], markersize=10, fillstyle='none')
    
    plt.xlabel('Implementing Ratio')
    plt.ylabel('Accuracy')
    
    params = { 'alpha':0.5,'linestyle':lStyles[1] ,'color':colors[3]}
    plt.axhline(y=LSTM_test, label = LSTM_name ,**params)
    params = { 'alpha':0.5,'linestyle':lStyles[2] ,'color':colors[4]}
    plt.axhline(y=GRU_test,  label = GRU_name , **params)
    
    params = { 'alpha':0.5,'linestyle':lStyles[1] ,'color':colors[1]}
    plt.axhline(y=DT_test,label = DT_name ,**params)
    params = { 'alpha':0.5,'linestyle':lStyles[4] ,'color':colors[5]}
    plt.axhline(y=Ran_test, label = Ran_name , **params)
    params = { 'alpha':0.5,'linestyle':lStyles[2] ,'color':colors[2]}
    plt.axhline(y=Ada_test, label = Ada_name , **params)
    params = { 'alpha':0.5,'linestyle':lStyles[3] ,'color':colors[6]}
    plt.axhline(y=Grad_test, label = Grad_name , **params)

#    params = { 'alpha':0.5,'linestyle':lStyles[3] ,'color':colors[5]}
#    plt.axhline(y=DES_val, label = DES_name , **params)
    
    params = { 'alpha':0.5,'linestyle':lStyles[4] ,'color':colors[2]}
#    plt.axhline(y=dictNo['MaxPropose_ValidACC_percent'][-1], label = 'Max Recommend' , **params)
    plt.axhline(y=percent_Mod_MaxPropose_TestACC, label = 'Best - Full info.' , **params)
    
    plt.legend(leg)
    plt.show()
    fig.savefig(prefixPic+'Ratio_Test.pdf',bbox_inches='tight')

#%%
    
    

    
    fig=plt.figure(figsize=(7, 6))
    plt.plot(percent_Mod_seriesOut,percent_Mod_ValidACC,label = 'IBCF-RPM',
             color=colors[0],marker=mar[0], markersize=10, fillstyle='none')
    plt.plot(percent_Org_seriesOut,percent_Org_ValidACC,label = 'Original CF',
             color=colors[2],marker=mar[1], markersize=10, fillstyle='none')
    
    plt.xlabel('Implementing Ratio')
    plt.ylabel('Accuracy')
    
    params = { 'alpha':0.5,'linestyle':lStyles[1] ,'color':colors[3]}
    plt.axhline(y=LSTM_val, label = LSTM_name ,**params)
    params = { 'alpha':0.5,'linestyle':lStyles[2] ,'color':colors[4]}
    plt.axhline(y=GRU_val,  label = GRU_name , **params)
    
    params = { 'alpha':0.5,'linestyle':lStyles[1] ,'color':colors[1]}
    plt.axhline(y=DT_val,label = DT_name ,**params)
    params = { 'alpha':0.5,'linestyle':lStyles[4] ,'color':colors[5]}
    plt.axhline(y=Ran_val, label = Ran_name , **params)
    params = { 'alpha':0.5,'linestyle':lStyles[2] ,'color':colors[2]}
    plt.axhline(y=Ada_val, label = Ada_name , **params)
    params = { 'alpha':0.5,'linestyle':lStyles[3] ,'color':colors[6]}
    plt.axhline(y=Grad_val, label = Grad_name , **params)

#    params = { 'alpha':0.5,'linestyle':lStyles[3] ,'color':colors[5]}
#    plt.axhline(y=DES_val, label = DES_name , **params)
    
    params = { 'alpha':0.5,'linestyle':lStyles[4] ,'color':colors[2]}
#    plt.axhline(y=dictNo['MaxPropose_ValidACC_percent'][-1], label = 'Max Recommend' , **params)
    plt.axhline(y=percent_Mod_MaxPropose_ValidACC, label = 'Best - Full info.' , **params)
    
    plt.legend(leg,loc='lower right')
    plt.show()
    fig.savefig(prefixPic+'Ratio_Valid.pdf',bbox_inches='tight')
    
#%%
#    fig=plt.figure(figsize=(6, 5))
#    plt.bar(percent_Mod_seriesOut,percent_Mod_runtime,label = 'CF method')
#    
#    plt.xlabel('Implementing Ratio')
#    plt.ylabel('Simulation Time')

    N = len(percent_Mod_runtime)
    ind = np.arange(N)  # the x locations for the groups
    width = 0.27       # the width of the bars
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    yvals = percent_Mod_runtime
    rects1 = ax.bar(ind, yvals, width, color='r')
    zvals = np.ones(N)*percent_Mod_Full_runtime
    rects2 = ax.bar(ind+width, zvals, width, color='g')
    
    ax.set_xlabel('Implementing Ratio')
    ax.set_ylabel('Simulation Time')

    ax.set_xticks(ind+0.5*width)
    ax.set_xticklabels( np.arange(0.1,0.75,0.05) )
    ax.legend( (rects1[0], rects2[0]), ('CF','Full') )
    
    fig.savefig(prefixPic+'RunTime_Ratio.pdf',bbox_inches='tight')
    plt.show()

#%%
  
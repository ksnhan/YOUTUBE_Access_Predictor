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


def Readata(filecsv):
    # --- Read Data --- #
    data = pd.read_csv(filecsv)
    
    data.iloc[:,1:] = 100 - data.iloc[:,1:]
    return data

def fillMask(data,mask):
    data_sparse = pd.DataFrame(index=data.loc[:,'Video_ID'],columns=data.columns[1:])
    data_drop = data.drop('Video_ID', 1)
    for i in range(data_sparse.shape[0]):
        for j in range(data_sparse.shape[1]):
            if mask[i][j]:
                data_sparse.iloc[i][j] = data_drop.iloc[i][j] 
    return data_sparse


def SparseGen(data, percent):
    ################### SPARSE GENERATOR ##################################
    
    np.random.seed (1569) #(15128)   #(4379) #100 #185
    mask = eval('np.random.rand'+ str(data.shape) +'<' + str(percent))
    data_sparse = fillMask(data,mask)
    
    return data_sparse,mask


    
#def getRunTime(runtime_drop,mask):
#    runtime=0
#    for i in range(runtime_drop.shape[0]):
#        for j in range(runtime_drop.shape[1]):
#            if mask[i][j]:
#                runtime += runtime_drop[i][j]
#    return runtime

def Normalize(data_sparse, data, mask):
    # normalize
    b=data_sparse.mean(axis=1)
    data_normalize = pd.DataFrame(index=data.loc[:,'Video_ID'],columns=data.columns[1:])
    data_normalize[:] = 0
    for i in range(data_sparse.shape[0]):
        for j in range(data_sparse.shape[1]):
            if mask[i][j]:          
                data_normalize.iloc[i][j] = data_sparse.iloc[i][j] - b[i]
    return data_normalize

def NeigbourMat(data_normalize,data):       
    # Neighbor again with sparse matrix
    dist_vv = pd.DataFrame(index=data.loc[:,'Video_ID'],columns=data.loc[:,'Video_ID'])
    for i in range(0,len(dist_vv.columns)) :
        for j in range(0,len(dist_vv.columns)) :
            if sum(data_normalize.iloc[j,:]*data_normalize.iloc[j,:]) \
            == 0 or sum(data_normalize.iloc[i,:]*data_normalize.iloc[i,:]) == 0:
                dist_vv.iloc[i,j] = 0
            else:
                dist_vv.iloc[i,j] = 1-cosine(data_normalize.iloc[i,:],data_normalize.iloc[j,:])

    maxNeighbour = 30        
    # Create a placeholder items for closes neighbours to an item
    vv_neighbours = pd.DataFrame(index=dist_vv.columns,columns=range(1,maxNeighbour+1))
     
    # Loop through our similarity dataframe and fill in neighbouring item names
    for i in range(0,len(dist_vv.columns)):
        vv_neighbours.iloc[i,:maxNeighbour] = dist_vv.iloc[0:,i].sort_values(ascending=False)[:maxNeighbour].index
    
    return dist_vv, vv_neighbours
#def getScore(history, similarities):
#   return sum(history*similarities)/sum(similarities)

def getScore(history, similarities):

    Nom=0;
    Den =0        
    for i in range(len(history)):
        if not np.isnan(history[i]):
            Nom += history[i]*similarities[i]
            Den += similarities[i]
    if Den != 0:
        return Nom/Den
    else:
        return
            
#    import numpy as np
#    denom= sum([not np.isnan(i) for i in history]*similarities)
#    if not np.isnan(denom):
#        return np.nansum(history*similarities)/denom

def CF_Fill(data_sparse,  vv_neighbours, dist_vv, data, topThr, modified=1):

    data_sims = pd.DataFrame(index=data.loc[:,'Video_ID'],columns=data.columns[1:])
    data_fill = pd.DataFrame(index=data.loc[:,'Video_ID'],columns=data.columns[1:])


    data_sims.iloc[:,:]=data_sparse.iloc[:,:]
    data_fill.iloc[:,:]=data_sparse.iloc[:,:]
    
    if modified:
        c=data_sparse.min(axis=0) #data_sparse.mean(axis=0, skipna=True) #
        for i in range(0,len(data_sims.index)):
            for j in range(len(data_sims.columns)):
                if pd.isnull(data_sims.iloc[i][j]):
                    data_sims.iloc[i][j] = c[j]


    for i in range(0,len(data_sims.index)):
        for j in range(len(data_sims.columns)):
            ID = data_sims.index[i]
            #method = data_sims.columns[j]
            #print(i,j)
            if pd.isnull(data_sparse.iloc[i][j]):
                #print(i,j)
                ID_top_names = vv_neighbours.loc[ID][1:topThr+1]
                ID_top_simsValue = dist_vv.loc[ID].sort_values(ascending=False)[1:topThr+1]
                ACC_history = data_sims.loc[ID_top_names,data_sparse.columns[j]] 
                data_fill.iloc[i][j] = getScore(ACC_history, ID_top_simsValue)
    return data_fill

def Recommend(data_fill):
    # Get the top songs
    data_recommend = pd.DataFrame(index=data_fill.index, columns=['1','2','3','4','5','6'])
    #data_recommend.iloc[0:,0] = data_sims.iloc[:,0]
     
    # Instead of top song scores, we want to see names
    for i in range(0,len(data_fill.index)):
        data_recommend.iloc[i,:] = data_fill.iloc[i,:].sort_values(ascending=False).iloc[0:6,].index.transpose()
     
    # Print a sample
    #print( data_recommend.iloc[:10,:1])
    return data_recommend

def main(topThr = 10, percent = 0.33, modified = 0,prt=0):
    
    filecsvEval = 'RESULT/CSV_MAE/Result_Algs_withES_mae_test.csv'#'Result_diff_test.csv'
    dataTest = Readata(filecsvEval)
    dataTest_drop = dataTest.drop('Video_ID', 1)
    
    filecsv = 'RESULT/CSV_MAE/Result_Algs_withES_mae_valid.csv'#'Result_diff_test.csv'
    dataValid = Readata(filecsv)
    dataValid_drop = dataValid.drop('Video_ID', 1)
    
    rtcsv = 'RESULT/CSV/Result_Algs_withES_run_time.csv'#'Result_diff_test.csv'
    runtime_data = pd.read_csv(rtcsv)
    runtime_data_drop = runtime_data.drop('Video_ID', 1)
    Full_runtime =  sum(runtime_data_drop.sum(axis=1))
    print(Full_runtime)
    
    data_sparse, mask = SparseGen(dataTest, percent)
    
    runtime_sparse = fillMask(runtime_data,mask)
    #print(runtime_sparse)
    runtime = sum(runtime_sparse.sum(axis=1))
    #print("Run time Sparse:{0}".format(runtime))
    print("Run time Sparse/Full: {0} / {1} ".format(runtime,Full_runtime))
    
    
    data_normalize = Normalize(data_sparse, dataTest, mask)
    dist_vv, vv_neighbours = NeigbourMat(data_normalize, dataTest)
    
    
    data_fill = CF_Fill(data_sparse, vv_neighbours, dist_vv, dataTest, topThr, modified)
    data_recommend = Recommend(data_fill)
    
    FinalACC_Test = 0
    FinalACC_Valid = 0
       
    BestPropose_ValidACC = 0
    BestPropose_TestACC = 0
    
    MaxPropose_TestACC=0
    MaxPropose_ValidACC=0
    
    MaxMeasurePropose = 0;
    lenData = data_recommend.shape[0]
    for i in range(lenData):
        FinalACC_Valid += dataValid.iloc[i][data_recommend.iloc[i,0]]
        FinalACC_Test += dataTest.iloc[i][data_recommend.iloc[i,0]]
        
        BestPropose_ValidACC += max(dataValid_drop.iloc[i][:])
        ind2 = np.argmax(dataValid_drop.iloc[i][:])
        BestPropose_TestACC += dataTest.iloc[i][ind2]
        
        MaxPropose_TestACC  += max(dataTest_drop.iloc[i][:])
        ind = np.argmax(dataTest_drop.iloc[i][:])
        MaxPropose_ValidACC +=dataValid.iloc[i][ind]
        
        
        ind1 = np.argmax(data_sparse.iloc[i][:])
        MaxMeasurePropose += dataValid.iloc[i][ind1]
    print("Result: \n \
          TestACC:{0}; ValidACC:{1}; \n \
          MaxPropose_TestACC: {4}; MaxPropose_ValidACC : {5};\n \
          BestPropose_TestACC:{3}; BestPropose_ValidACC:{2}; \n \
          MaxMeasurePropose:{6}; runtime/full:{7}/{8}\n; ". \
          format(FinalACC_Test/lenData, FinalACC_Valid/lenData,
                 BestPropose_ValidACC/lenData,BestPropose_TestACC/lenData,
                 MaxPropose_TestACC/lenData, MaxPropose_ValidACC/lenData,
                 MaxMeasurePropose/lenData,runtime,Full_runtime)) 
    if prt: 
        
        print("The Average ACC for methods:")
        print(dataValid_drop.mean(axis=0))
        print("The total RT for methods:")
        print(runtime_data_drop.sum(axis=0) )
    
    Out_dict = {"TestACC":FinalACC_Test/lenData, 
                "ValidACC":FinalACC_Valid/lenData,
                "MaxPropose_TestACC":MaxPropose_TestACC/lenData,
                "MaxPropose_ValidACC" : MaxPropose_ValidACC/lenData,
                "BestPropose_TestACC" : BestPropose_TestACC/lenData,
                "BestPropose_ValidACC": BestPropose_ValidACC/lenData,
                "MaxMeasurePropose": MaxMeasurePropose/lenData,
                "runtime":runtime,
                "Full_runtime":Full_runtime}
    
    return Out_dict

def runMain(modified=1, ratio = 0.2, top =10, prefix = 'RESULT/Proposed'):
   
    
    ratioRange = np.arange(0.1, 0.75, 0.05)
    topKRange = range(1,21)
    
    dict_Top ={}
    for topThr in topKRange:
        print('With Top ' + str(topThr) + ': ')
        params_Top = {'topThr':topThr, 'percent': ratio, 'modified':modified, 'prt':0}
        dict_Top[str(topThr)] =main(**params_Top)
    with open(prefix + 'CF_Result_Top.json','w') as f1:
        json.dump(dict_Top,f1)

    dict_percent ={}
    for percent in ratioRange:
        print('With ratio ' + str(percent) + ': ')
        params_percent = {'topThr':top, 'percent': percent, 'modified':modified, 'prt':0}
        dict_percent[str(percent)] =main(**params_percent)
    with open(prefix + 'CF_Result_Ratio.json','w') as f2:
        json.dump(dict_percent,f2)

if __name__ == "__main__":
    
    runMain(modified=1, ratio = 0.2, top =10, prefix = 'RESULT/MAE')
    runMain(modified=0, ratio = 0.2, top =10, prefix = 'RESULT/MAE')
    #runMain(modified=0,'RESULT/Original')

        
    
 

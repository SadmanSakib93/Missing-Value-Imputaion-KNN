# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 13:54:25 2019

@author: Sadman Sakib
"""

#Run this file in Anaconda navigator's (SPYDER IDE)
#Then no extra library will be needed as those come automatically with anaconda

import csv
import random
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
import globalVars as gb
import pandas as pd  
import numpy as np
import operator
import math 
import copy

fileNameCSV='data.csv'
dataFrame = pd.read_csv(fileNameCSV) 
totalInstances=len(dataFrame)
headList=list(dataFrame.keys())
dataList = dataFrame.values.tolist()
dataOriginal=dataFrame.values.tolist()
dataWithRemovedValues=dataFrame.values.tolist()
allRandomRemoveRowIndex=[]
allRatioAccuForEachFeature=[]
finalAccu1NN=[]
imputingColumn=-1
zeroMargin=0.000001

#*** Remove everything from the file ***
def clearCSV(fileName):
    f = open(fileName, "w")
    f.truncate()
    f.close()
    
#*** Write on a CSV file ***
def writeCSV(outputFileName,fileContent):
    with open(outputFileName, 'a',newline='') as writeFile:
        writer = csv.writer(writeFile)
        writer.writerows(fileContent)
#*** Calculate Euclidean Distance ***
def euDistance(x,y):
    distance = math.sqrt(sum([(a - b) ** 2 for a, b in zip(x, y)]))
    return distance
#*** Calculate Manhattan Distance ***    
def manhattanDistance(x,y):
    return sum(abs(a-b) for a,b in zip(x,y))
#*** Calculate Hamming Distance ***
def hammingDistance(instance1, instance2):
    distanceMeasure=0
    for i in range(len(instance1)):
        if(instance1[i]!=instance2[i]):
            distanceMeasure=distanceMeasure+abs(instance1[i]-instance2[i])
    return distanceMeasure
#*** Generate Weightes for features ***
def generateWeights(k, distances):
    result = np.zeros(k, dtype=np.float32)
    dis=[]
    for tupleIndex in range(len(distances)):
        dis.append(distances[tupleIndex][2])
    sum = 0.0
    for i in range(k):
        if(dis[i]==0):
            dis[i]=zeroMargin
        result[i] += 1.0 / dis[i]
        sum += result[i]
    result /= sum
    return result

def makeDictionaryOfWeights(listOfTuples,kWeights):
    dictionary={}
    iterIndx=0
    for eachPredictedValue in listOfTuples:
        predictedValue=dataOriginal[eachPredictedValue[1]][imputingColumn]
        value = dictionary.get(predictedValue, 0)
        dictionary[predictedValue]=value+kWeights[iterIndx]
        iterIndx=iterIndx+1
    return dictionary

#***  mean absolute error ***
def mse(original,predicted):
    err=np.mean(np.abs((original - predicted)))
    if (err>1):
        err=1
    if(err <0):
        err=0
    return err

#*** Calculate Mean absolute percentage error (MAPE) ***
def mape(original,predicted): 
    err=np.mean(np.abs((original - predicted) / original))
    if (err>1):
        err=1
    if(err <0):
        err=0
    return err

def featureScaling(scalingMethod):
    global dataList
    global dataOriginal
    global dataWithRemovedValues
    if(scalingMethod=='MinMaxScaler'):
        scaler = preprocessing.MinMaxScaler(feature_range=(0,1))
    elif (scalingMethod=='StandardScaler'):
        scaler = StandardScaler()
    scaleFeatures=['age','trestbps','thalach']
    dataFrame[scaleFeatures] = scaler.fit_transform(dataFrame[scaleFeatures])
    dataList = dataFrame.values.tolist()
    dataOriginal=dataFrame.values.tolist()
    dataWithRemovedValues=dataFrame.values.tolist() 
    if (scalingMethod=='StandardScaler'):
        scaler.fit(dataList) 
#*** Weighted KNN ***
def weightedKnn(k, data, dataOriginal):
    for j in data: 
        del j[imputingColumn]
    for removedRowIndex in allRandomRemoveRowIndex:
        distanceEuc=[]
        distanceMan=[]
        testInstance=[]
        kthNeighborEuc=[]
        kthNeighborMan=[]
        accEuc=0
        accMan=0
        testInstance.append(data[removedRowIndex])
        for eachRowIndex in range(len(data)):
            if(removedRowIndex!=eachRowIndex):
                distanceEuc.append((removedRowIndex,eachRowIndex,euDistance(data[removedRowIndex],data[eachRowIndex])))
                distanceMan.append((removedRowIndex,eachRowIndex,manhattanDistance(data[removedRowIndex],data[eachRowIndex])))
            distanceEuc.sort(key = operator.itemgetter(2))  
            distanceMan.sort(key = operator.itemgetter(2)) 
        for j in range(k):
            kthNeighborEuc.append(distanceEuc[j])  
            kthNeighborMan.append(distanceMan[j])
        kthNeighEucWeights=generateWeights(k,kthNeighborEuc)
        kthNeighManWeights=generateWeights(k,kthNeighborMan)
        dictEuc=makeDictionaryOfWeights(kthNeighborEuc,kthNeighEucWeights)
        dictMan=makeDictionaryOfWeights(kthNeighborMan,kthNeighManWeights)
        gb.originalValue.append(dataOriginal[removedRowIndex][imputingColumn])
        gb.predictedValue.append(max(dictEuc.items(), key=operator.itemgetter(1))[0])
        accEuc=accEuc+mape(np.array(gb.originalValue),np.array(gb.predictedValue))
        gb.originalValue.clear()
        gb.predictedValue.clear()
        gb.originalValue.append(dataOriginal[removedRowIndex][imputingColumn])
        gb.predictedValue.append(max(dictMan.items(), key=operator.itemgetter(1))[0])
        accMan=accMan+mape(np.array(gb.originalValue),np.array(gb.predictedValue))
        gb.originalValue.clear()
        gb.predictedValue.clear()
    meanAccuEuc=accEuc/len(allRandomRemoveRowIndex) 
    meanAccuMan=accMan/len(allRandomRemoveRowIndex)        
    allRatioAccuForEachFeature.append(1-meanAccuEuc)    
    allRatioAccuForEachFeature.append(1-meanAccuMan) 
 
#*** KNN ***
def knn(k,data, dataOriginal):
    for j in data: 
        del j[imputingColumn]     
    for removedRowIndex in allRandomRemoveRowIndex:
        distanceEuc=[]
        distanceMan=[]
        testInstance=[]
        kthNeighborEuc=[]
        kthNeighborMan=[]
        accEuc=0
        accMan=0  
        testInstance.append(data[removedRowIndex])
        for eachRowIndex in range(len(data)):
            if(removedRowIndex!=eachRowIndex):
                distanceEuc.append((removedRowIndex,eachRowIndex,euDistance(data[removedRowIndex],data[eachRowIndex])))
                distanceMan.append((removedRowIndex,eachRowIndex,manhattanDistance(data[removedRowIndex],data[eachRowIndex])))
            distanceEuc.sort(key = operator.itemgetter(2))  
            distanceMan.sort(key = operator.itemgetter(2)) 
        for j in range(k):
            kthNeighborEuc.append(distanceEuc[j])  
            kthNeighborMan.append(distanceMan[j])
        for eachPredictedValue in kthNeighborEuc:
            gb.originalValue.append(dataOriginal[removedRowIndex][imputingColumn])
            gb.predictedValue.append(dataOriginal[eachPredictedValue[1]][imputingColumn])
        accEuc=accEuc+mape(np.array(gb.originalValue),np.array(gb.predictedValue))
        gb.originalValue.clear()
        gb.predictedValue.clear()
        for eachPredictedValue in kthNeighborMan:
            gb.originalValue.append(dataOriginal[removedRowIndex][imputingColumn])
            gb.predictedValue.append(dataOriginal[eachPredictedValue[1]][imputingColumn])
        accMan=accMan+mape(np.array(gb.originalValue),np.array(gb.predictedValue))
        gb.originalValue.clear()
        gb.predictedValue.clear()  
    meanAccuEuc=accEuc/len(allRandomRemoveRowIndex)
    meanAccuMan=accMan/len(allRandomRemoveRowIndex)            
    allRatioAccuForEachFeature.append(1-meanAccuEuc)    
    allRatioAccuForEachFeature.append(1-meanAccuMan)    

#*** KNN Categorical ***
def knnCategorical(k,data, dataOriginal):
    for j in data: 
        del j[imputingColumn] 
    for removedRowIndex in allRandomRemoveRowIndex:
        accHam=0
        distanceHam=[]
        testInstance=[]
        kthNeighborHam=[]
        testInstance.append(data[removedRowIndex])
        for eachRowIndex in range(len(data)):
            if(removedRowIndex!=eachRowIndex):
                distanceHam.append((removedRowIndex,eachRowIndex,hammingDistance(data[removedRowIndex],data[eachRowIndex])))
            distanceHam.sort(key = operator.itemgetter(2))        
        for j in range(k):
            kthNeighborHam.append(distanceHam[j])  
        for eachPredictedValue in kthNeighborHam:
            gb.originalValue.append(dataOriginal[removedRowIndex][imputingColumn])
            gb.predictedValue.append(dataOriginal[eachPredictedValue[1]][imputingColumn])
        accHam=accHam+mse(np.array(gb.originalValue),np.array(gb.predictedValue))
        gb.originalValue.clear()
        gb.predictedValue.clear() 
    meanAccuHam=accHam/len(allRandomRemoveRowIndex)
    allRatioAccuForEachFeature.append(1-meanAccuHam)   
    
#*** Weighted KNN for Categorical ***
def categoricalWeightedKnn(k, data, dataOriginal):
    for j in data: 
        del j[imputingColumn] 
    for removedRowIndex in allRandomRemoveRowIndex:
        accHam=0
        distanceHam=[]
        testInstance=[]
        kthNeighborHam=[]
        testInstance.append(data[removedRowIndex])
        for eachRowIndex in range(len(data)):
            if(removedRowIndex!=eachRowIndex):
                distanceHam.append((removedRowIndex,eachRowIndex,hammingDistance(data[removedRowIndex],data[eachRowIndex])))
            distanceHam.sort(key = operator.itemgetter(2)) 
        for j in range(k):
            kthNeighborHam.append(distanceHam[j])     
        for eachPredictedValue in kthNeighborHam:
            gb.originalValue.append(dataOriginal[removedRowIndex][imputingColumn])
            gb.predictedValue.append(dataOriginal[eachPredictedValue[1]][imputingColumn])
        accHam=accHam+mse(np.array(gb.originalValue),np.array(gb.predictedValue))
        gb.originalValue.clear()
        gb.predictedValue.clear() 
    meanAccuHam=accHam/len(allRandomRemoveRowIndex)
    allRatioAccuForEachFeature.append(1-meanAccuHam)  
       
#*** Randomly removing some values from a feature ***
def removeValues(dataList, percentage, columnIndex):
    allRandomRemoveRowIndex.clear()
    removeDataLength=int(totalInstances*(percentage/100))
    for ind in range(removeDataLength):
        allRandomRemoveRowIndex.append(random.randint(0, totalInstances-1))
    for ind in allRandomRemoveRowIndex:
        dataList[ind][columnIndex]='NA'
        dataWithRemovedValues[ind][columnIndex]='NA'

#*** Choose an Algorithm based on choice ***       
def chooseAlgo(kValue, knnType):  
    dataFrame = pd.read_csv(fileNameCSV) 
    dataOriginal=dataFrame.values.tolist()  
    if(knnType=='continiousNormal'):
        knn(kValue, dataList, dataOriginal)
    elif(knnType=='continiousWeighted'):
        weightedKnn(kValue,dataList,dataOriginal)
    elif(knnType=='categoricalNormal'):
        knnCategorical(kValue, dataList, dataOriginal)
    elif(knnType=='categoricalWeighted'):
        categoricalWeightedKnn(kValue, dataList, dataOriginal)  
        
#*** Write Results to CSV file ***        
def writeAccuracyToCSV(kValue, featureType, CSVcontent, knnType):
    global dataList
    global dataOriginal
    global allRatioAccuForEachFeature
    global imputingColumn
    for indx in featureType:
        dataList = copy.deepcopy(dataOriginal)
        imputingColumn=indx
        allRatioAccuForEachFeature.append(headList[imputingColumn])
        removeValues(dataList, 5,imputingColumn)
        chooseAlgo(kValue, knnType)
        dataList = copy.deepcopy(dataOriginal)
        removeValues(dataList, 10,imputingColumn)
        chooseAlgo(kValue, knnType)
        dataList = copy.deepcopy(dataOriginal)
        removeValues(dataList, 20,imputingColumn)
        chooseAlgo(kValue, knnType)
        CSVcontent.append(allRatioAccuForEachFeature)
        writeCSV('results.csv',CSVcontent)
        allRatioAccuForEachFeature.clear()
        CSVcontent.clear()  
#*** Call different Algorithms sequentially ***
def callAllCSVwrite(scl):
    writeAccuracyToCSV(1, gb.continuousFeatruesColumn, gb.CSVcontent1NN, 'continiousNormal')
    writeAccuracyToCSV(7, gb.continuousFeatruesColumn, gb.CSVcontentKNN, 'continiousNormal')
    writeAccuracyToCSV(7, gb.continuousFeatruesColumn, gb.CSVcontentWeightedKNN, 'continiousWeighted')   
    if(scl==1):
        featureScaling('MinMaxScaler')
    writeAccuracyToCSV(1, gb.categoricalFeaturesColumn, gb.CSVcategoricalContent1NN, 'categoricalNormal')
    writeAccuracyToCSV(7, gb.categoricalFeaturesColumn, gb.CSVcategoricalContentKNN, 'categoricalNormal')
    writeAccuracyToCSV(7, gb.categoricalFeaturesColumn, gb.CSVcategoricalContentWeightedKNN, 'categoricalWeighted')

clearCSV('results.csv')
gb.headerCSV()

print("Caclculating accuracy of data without scaling . . . .")
callAllCSVwrite(1)

#First SCALING method
print("Caclculating accuracy of data after first scaling method. . . .")
gb.headerCSV()
featureScaling('StandardScaler')
gb.CSVcontent1NN.insert(0,['*** RESULTS of SCALING 1 ***'])
gb.CSVcontent1NN.insert(0,[])
callAllCSVwrite(0)

#Second SCALING method
print("Caclculating accuracy of data after second scaling method. . . .")
gb.headerCSV()
featureScaling('MinMaxScaler')
gb.CSVcontent1NN.insert(0,['*** RESULTS of SCALING 2  ***'])
gb.CSVcontent1NN.insert(0,[])
callAllCSVwrite(0)
print("result.csv file populated with latest results . . . .")
print("***END***")

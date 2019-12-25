# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 20:59:12 2019

@author: Sadman Sakib
"""

originalValue=[]
predictedValue=[]
#continuousFeatruesColumn=[0,5,6]
#categoricalFeaturesColumn=[13]
continuousFeatruesColumn=[0,1,2]
categoricalFeaturesColumn=[3,4]
CSVcontent1NN=[]
CSVcontentKNN=[]
CSVcontentWeightedKNN=[]
CSVcategoricalContent1NN=[]
CSVcategoricalContentKNN=[]
CSVcategoricalContentWeightedKNN=[]

def headerCSV():
    global CSVcontent1NN
    global CSVcontentKNN
    global CSVcontentWeightedKNN
    global CSVcategoricalContent1NN
    global CSVcategoricalContentKNN
    global CSVcategoricalContentWeightedKNN
    CSVcontent1NN=[['','***** CONTINUOUS FEATURES *****'],['', 'Results of 1NN ', '', '', '', '', ''], 
                  ['', ' (5%) Euclidean', '(5%) Manhattan', '(10%) Euclidean', '(10%) Manhattan ', '(20%) Euclidean ', '(20%) Manhattan ']]
    
    CSVcontentKNN=[['', 'Results of KNN ', '', '', '', '', ''], 
                  ['', ' (5%) Euclidean', '(5%) Manhattan', '(10%) Euclidean', '(10%) Manhattan ', '(20%) Euclidean ', '(20%) Manhattan ']]
    
    CSVcontentWeightedKNN=[['', 'Results of WEIGHTED KNN ', '', '', '', '', ''], 
                  ['', ' (5%) Euclidean', '(5%) Manhattan', '(10%) Euclidean', '(10%) Manhattan ', '(20%) Euclidean ', '(20%) Manhattan ']]
    
    
    CSVcategoricalContent1NN=[['','***** CATEGORICAL FEATURES *****'],['', 'Results of 1NN ', '', '', '', '', ''], 
                  ['', '(5%) Hamming distance ',  '(10%) Hamming distance ', '(20%) Hamming distance ']]
    
    CSVcategoricalContentKNN=[['', 'Results of KNN ', '', '', '', '', ''], 
                  ['', '(5%) Hamming distance ',  '(10%) Hamming distance ', '(20%) Hamming distance ']]
    
    CSVcategoricalContentWeightedKNN=[['', 'Results of WEIGHTED KNN ', '', '', '', '', ''], 
                  ['', '(5%) Hamming distance ',  '(10%) Hamming distance ', '(20%) Hamming distance ']]
import numpy as np
import pandas as pd
import random

#path='/content/drive/MyDrive/MachineLearning/09-01-2023/playtennis.csv'
path='/home/white-death/MachineLearning/playtennies.csv'
Data = pd.read_csv(path)

# Shuffle the data
Data = Data.sample(frac=1, random_state=42)

# Split the data into train and test sets
SplitRatio = 0.7
SplitLoc = int(len(Data) * SplitRatio)
DataTrain = Data[:SplitLoc]
DataTest = Data[SplitLoc:]

def ProbabilityFunc(A, B, C, D):
    count, TotInstan = 0, 0
    A_idx = DataTrain.columns.get_loc(A)
    C_idx = DataTrain.columns.get_loc(C)
    for i in DataTrain.iloc[:, A_idx]:
        if(i == B and DataTrain.iloc[TotInstan, C_idx] == D):
            count += 1
        TotInstan += 1
    if TotInstan == 0:
        return 0
    return count / TotInstan

def Predict(Inp):
    LiOdYes,LiOdNo = 0,0
    count,flag = 0,0
    for i in DataTrain[DataTrain.columns[-1]]:
        if(i == 'yes'):
            count += 1
        flag += 1
    ProYes = count/flag
    ProNo = 1-ProYes
    LiOdYes, LiOdNo = ProYes, ProNo
    for i in range(len(Inp)):
        LiOdYes *= ProbabilityFunc(DataTrain.columns[i], Inp[i], DataTrain.columns[-1], 'yes')
        LiOdNo *= ProbabilityFunc(DataTrain.columns[i], Inp[i], DataTrain.columns[-1], 'no')
    if(LiOdYes > LiOdNo):
        return 'yes'
    else:
        return 'no'

def Accuracy(Data):
    correct = 0
    for i in range(len(Data)):
        instance = Data.iloc[i, :-1].values
        ActualClass = Data.iloc[i, -1]
        PredictedClass = Predict(instance)
        print(instance)
        print(PredictedClass)
        if ActualClass == PredictedClass:
            correct += 1
    return correct / len(Data)

print('Accuracy:', Accuracy(DataTest))

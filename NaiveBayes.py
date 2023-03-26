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

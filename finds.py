#find S algorithm

import pandas as pd
import numpy as np

df = pd.read_csv('a1.csv')
arr = np.array(df)[:,:-1]
target = np.array(df)[:,-1]

def findS(arr,target):
    
    flag = 0
    for index, value in enumerate(target):
        if value == "yes":
            specific_hypothesis = arr[index].copy()
            flag = 1
            break
    
    if(flag == 0):
        rows, cols = arr.shape
        specific_hypothesis = np.full(cols, '$')

             
    for index, value in enumerate(arr):
        if target[index] == "yes":
            for x in range(len(specific_hypothesis)):
                if value[x] != specific_hypothesis[x]:
                    specific_hypothesis[x] = '?'
                else:
                    pass
                 
    return specific_hypothesis

print("The final hypothesis is: ", findS(arr, target))

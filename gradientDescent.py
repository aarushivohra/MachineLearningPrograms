#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np 

iris = pd.read_csv('Iris.csv')
iris.head()


# In[2]:


X = iris[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
Y = iris['Species']
X.head()
#Y.head()


# In[3]:


pd.options.mode.chained_assignment = None
def normalise(X):
    for i in range(len(X.columns)-1):
        max_val = X[X.columns[i]].max()
        min_val = X[X.columns[i]].min()
        for j in range(len(X)):
            X.iloc[j, i] = (int(max_val) - int(X.iloc[j, i]))/(max_val - min_val)
    X.head()
normalise(X)


# In[46]:


def gradientDescent(X, Y, learning_rate, iterations):
    n = len(X)
    w = np.zeros((X.shape[1], 1))
    b = 0
    costs = 0
    
    for i in range(iterations):
        Z = np.dot(w.T, X.T) + b
        
        pred_Y = 1/(1 + 1/np.exp(Z))
        y = np.matrix(Y)
        y = y.T

        cost = -(1/n) * np.sum(y*np.log(pred_Y)+(1-y)*np.log(1-pred_Y))
        
        dw = 1/n * np.dot(X.T,(pred_Y-y).T)
        db = 1/n * np.sum(pred_Y-Y)
        
        #updating w and b
        w = w - learning_rate*dw
        b = b - learning_rate*db
        
        costs.append(cost)
        
    print(costs)
    return min(costs)
        
    


# In[9]:


X.shape


# In[7]:


X.head()


# In[8]:


Y.replace(['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'], [0, 1, 2], inplace=True)
Y.head()


# In[47]:


gradientDescent(X, Y, 0.01, 10000)


# In[ ]:





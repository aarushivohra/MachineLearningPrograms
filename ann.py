#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# In[2]:


data = load_iris()
X=data.data
y=data.target


# In[3]:


y = pd.get_dummies(y).values
y[:3]


# In[4]:


x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=20, random_state=4)


# In[5]:


learning_rate = 0.1
iterations = 5000
N = y_train.size

input_size = 4

hidden_size = 2 

output_size = 3  

results = pd.DataFrame(columns=["mse", "accuracy"])


# In[6]:


np.random.seed(10)

W1 = np.random.normal(scale=0.5, size=(input_size, hidden_size))   

W2 = np.random.normal(scale=0.5, size=(hidden_size , output_size)) 


# In[7]:


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def mean_squared_error(y_pred, y_true):
    return ((y_pred - y_true)**2).sum() / (2*y_pred.size)
    
def accuracy(y_pred, y_true):
    acc = y_pred.argmax(axis=1) == y_true.argmax(axis=1)
    return acc.mean()


# In[8]:


for itr in range(iterations):    
    
    Z1 = np.dot(x_train, W1)
    A1 = sigmoid(Z1)

    Z2 = np.dot(A1, W2)
    A2 = sigmoid(Z2)
    
    mse = mean_squared_error(A2, y_train)
    acc = accuracy(A2, y_train)
    results=results.append({"mse":mse, "accuracy":acc},ignore_index=True )
    
    E1 = A2 - y_train
    dW1 = E1 * A2 * (1 - A2)

    E2 = np.dot(dW1, W2.T)
    dW2 = E2 * A1 * (1 - A1)

    W2_update = np.dot(A1.T, dW1) / N
    W1_update = np.dot(x_train.T, dW2) / N

    W2 = W2 - learning_rate * W2_update
    W1 = W1 - learning_rate * W1_update


# In[9]:


results.mse.plot(title="Mean Squared Error")


# In[10]:


results.accuracy.plot(title="Accuracy")


# In[11]:


Z1 = np.dot(x_test, W1)
A1 = sigmoid(Z1)

Z2 = np.dot(A1, W2)
A2 = sigmoid(Z2)

acc = accuracy(A2, y_test)
print("Accuracy: {}".format(acc))


# In[ ]:





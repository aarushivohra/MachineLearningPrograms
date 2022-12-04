#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
data = pd.read_csv("./clustering.csv")
data.head()


# In[2]:


data = data.loc[:, ['ApplicantIncome', 'LoanAmount']]
data.head()


# In[3]:


X = data.values
sns.scatterplot(x = X[:, 0], y = X[:, 1])
plt.xlabel('Income')
plt.ylabel('Loan')


# In[4]:


def Euclidean_distance(x, y):
    return np.sqrt((x[0]-y[0])**2 + (x[1]-y[1])**2)

def k_means_using_NN_without_threshold(X, k):
    loop = 1
    
    #initialise cluster[]
    cluster = np.zeros(X.shape[0])
    
    #pick k random centroids    
    while loop:
        for i, row in enumerate(X):
            #threshold = 0
            if(i < k):
                cluster[i] = i           
            else:
                #initialise min distance to infinite
                min_dist = float('inf')

                #calc dist from each preceeding point
                for index, point in enumerate(X):
                    idx = index
                    if(index < i):
                        dist = Euclidean_distance(row, point)
                        if(dist < min_dist):
                            min_dist = dist
                            cluster[i] = cluster[index]     
                            
        loop = loop - 1
            
    return cluster


def k_means_using_NN_with_threshold(X, k):
    loop = 1
    
    #initialise cluster[]
    cluster = np.zeros(X.shape[0])
    
    num_clusters = 0
    
    #pick k random centroids    
    while loop:
        for i, row in enumerate(X):
            threshold = 1000
            if(i == 0):
                cluster[i] = num_clusters
                num_clusters += 1            
            else:
                #initialise min distance to infinite
                min_dist = float('inf')

                #calc dist from each preceeding point
                idx = 0
                for index, point in enumerate(X):
                    idx = index
                    if(index < i):
                        dist = Euclidean_distance(row, point)
                        
                        if(dist < threshold):    
                            if(dist < min_dist):
                                min_dist = dist
                                cluster[i] = cluster[index]     
                                
                if(min_dist == float('inf')):
                    if(num_clusters < k):
                        #create new cluster
                        cluster[i] = num_clusters
                        num_clusters += 1
                    else:
                        #ignore the point
                        cluster[i] = -1
                            
        loop = loop - 1
            
    return cluster


# In[5]:


k = 3
cluster = k_means_using_NN_without_threshold(X, k)


# In[6]:


plt.clf()
sns.scatterplot(x = X[:, 0], y = X[:, 1], hue = cluster)
plt.xlabel('Income')
plt.ylabel('Loan')
plt.show()


# In[7]:


cluster = k_means_using_NN_with_threshold(X, k)
plt.clf()
sns.scatterplot(x = X[:, 0], y = X[:, 1], hue = cluster)
plt.xlabel('Income')
plt.ylabel('Loan')
plt.show()


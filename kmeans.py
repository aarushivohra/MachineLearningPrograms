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

def k_means(X, k):
    loop = 1
    
    #initialise cluster[]
    cluster = np.zeros(X.shape[0])
    
    #pick k random centroids
    centroids = data.sample(n=k).values
    max_iter = 300
    
    while loop:
        for i, row in enumerate(X):
            #initialise min distance to infinite
            min_dist = float('inf')

            #calc dist from each centroid
            for index, centroid in enumerate(centroids):
                dist = Euclidean_distance(row, centroid)
            
                if(dist < min_dist):
                    min_dist = dist
                    cluster[i] = index
        
        #update centroids
        new_centroids = pd.DataFrame(X).groupby(by=cluster).mean().values
        max_iter = max_iter-1
        
        if (np.count_nonzero(centroids - new_centroids) == 0) or max_iter == 0:
            loop = 0
        else:
            centroids = new_centroids
            
    return centroids, cluster


# In[5]:


k = 3
centroids, cluster = k_means(X, k)
print(centroids)


# In[6]:


plt.clf()
sns.scatterplot(x = X[:, 0], y = X[:, 1], hue = cluster)
sns.scatterplot(x = centroids[:,0], y = centroids[:,1], s = 100, color = 'red')
plt.xlabel('Income')
plt.ylabel('Loan')
plt.show()


# In[ ]:





#Importing the required modules
import numpy as np
from scipy.stats import mode
 
#Euclidean Distance
def eucledian(p1,p2):
    dist = np.sqrt(np.sum((p1-p2)**2))
    return dist
 
#Function to calculate KNN
def predict(x_train, y , x_input, k):
    op_labels = []
     
    #Loop through the Datapoints to be classified
    for item in x_input: 
         
        #Array to store distances
        point_dist = []
         
        #Loop through each training Data
        for j in range(len(x_train)): 
            distances = eucledian(np.array(x_train[j,:]) , item) 
            #Calculating the distance
            point_dist.append(distances) 
        point_dist = np.array(point_dist) 
         
        #Sorting the array while preserving the index
        #Keeping the first K datapoints
        dist = np.argsort(point_dist)[:k] 
         
        #Labels of the K datapoints from above
        labels = y[dist]
         
        #Majority voting
        lab = mode(labels) 
        lab = lab.mode[0]
        op_labels.append(lab)
 
    return op_labels



##################################



# KNN
import pandas as pd
import math
import operator
from sklearn.model_selection import train_test_split

def euclideanDistance(inst1, inst2, dimensions):
    distance = 0
    for i1, i2 in zip(inst1, inst2):
        if(type(i1) == str and not i1.isalpha() and not i2.isalpha()):
            distance += pow((int(i1) - int(i2)), 2)
    return math.sqrt(distance)


def KNN(trainInst, trainOp, testInst, k):
    distance = []
    for i in trainInst:
        tempDist = euclideanDistance(trainInst[i], testInst, 30) 
        distance.append((trainInst[i], trainOp[i], tempDist))
    distance.sort(key=operator.itemgetter(2))
    
    nearestNeighbours = []
    for i in range(k):
        nearestNeighbours.append(distance[i][0])
    return nearestNeighbours
 

def getClass(neighbours):
    classVote = {}
    for i in range(len(neighbours)):
        neighbourClass = neighbours[i][1]
        if neighbourClass in classVote:
            classVote[neighbourClass] += 1
        else:
            classVote[neighbourClass] = 1
        sortedClass = sorted(classVote.items(), key=operator.itemgetter(1), reverse=True)
        return sortedClass[0][0]
    


def main():
    df = pd.read_csv('Climate Data-kNN.csv')
    X = df[df.columns.drop('Troposphere')]
    Y = df.iloc[:, -1:]
    
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.7)
    

    k = 5
    for testInst, testInst in zip(X_test, Y_test):
        nearestNeighbours = KNN(X_train, Y_train, testInst, k)
        predictedClass = getClass(nearestNeighbours)
        actualClass = testInst
        
        correct = 0
        incorrect = 0
        if(predictedClass != actualClass):
            incorrect += 1
        else:
            correct += 1
            
        print("Correct Predictions: " + correct)
        print("Incorrect Predictions: " + incorrect)
        

if __name__ == "__main__":
    main()






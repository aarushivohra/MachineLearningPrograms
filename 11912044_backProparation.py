import numpy as np
import pandas as pd
import math

def sigmoid(x) : 
    return np.array([1/(1+math.exp(-x[i])) for i in range(len(x))])


def error_total(target,predicted):
    for i in range(len(target)) : 
        for j in range(len(target[i])):
            sum += (target[i][j] - predicted[i][j])**2

    return sum/2



def backPropagation(x, hidden_layer_weights,output_layer_weights,learning_rate ,o_out,o_in,h_in,h_out,y_target):

    # error in output layer
    err_output = o_out-y_target
    d_output_weights = err_output* o_out*(1-o_out)
   

    #error in hidden layer
    err_hidden = np.dot(d_output_weights,output_layer_weights.T) 
    d_hidden_weights = err_hidden*h_out*(1-h_out)

    output_layer_weights,hidden_layer_weights = update_weights(x,hidden_layer_weights,output_layer_weights,learning_rate,h_out,d_hidden_weights,d_output_weights)
    return output_layer_weights,hidden_layer_weights


def forwardPropagation(x,y_target,hidden_layer_weights,output_layer_weights,bias,iterations) : 
    

    for i in range(iterations):

        # for hidden layer
        h_in = np.dot(x,hidden_layer_weights)  
        h_in = np.add(h_in,np.array([bias[0] for i in range(len(h_in))]))
        h_out = sigmoid(h_in)

        # for output layer
        o_in = np.dot(h_out,output_layer_weights)
        o_in = np.add(o_in,np.array([bias[1] for i in range(len(o_in))]))
        o_out = sigmoid(o_in)
        output_layer_weights,hidden_layer_weights = backPropagation(x, hidden_layer_weights,output_layer_weights,learning_rate ,o_out,o_in,h_in,h_out,y_target)
    
    print("Output final : ")
    print(o_out)
    print("\n output layer weights : ")
    print(output_layer_weights)

    print("\n Hidden layer weights : ")
    print(hidden_layer_weights)
    return 0




def update_weights(x,hidden_layer_weights,output_layer_weights,learning_rate,h_out,d_hidden_weights,d_output_weights):


    output_layer_weights = output_layer_weights - learning_rate* h_out*d_output_weights
    hidden_layer_weights = hidden_layer_weights - learning_rate* x*d_hidden_weights

    # #print(o_out)
    #print(output_layer_weights)
    #print(hidden_layer_weights)

    return output_layer_weights,hidden_layer_weights


   


print("Input layer, hidden layer neurons and output layer neurons : ")
n_input = int(input())
n_hidden_layer = int(input())
n_output_layer = int(input())

learning_rate = 0.5
iterations = 10000

x = np.array([float(input()) for i in range(n_input)])
y_target = np.array([float(input()) for i in range(n_output_layer)])

hidden_layer_weights = np.array([[float(input()) for i in range(n_input)] for k in range(n_hidden_layer)])
hidden_layer_weights = hidden_layer_weights.reshape(n_input,n_hidden_layer)

output_layer_weights = np.array([[float(input()) for i in range(n_output_layer)] for k in range(n_hidden_layer)])
output_layer_weights = output_layer_weights.reshape(n_hidden_layer,n_output_layer)

bias = np.array([0.35 , 0.60])
forwardPropagation(x,y_target,hidden_layer_weights,output_layer_weights,bias,iterations)

'''
2
2
2
0.05
0.1
0.01
0.99
0.15                   
0.25
0.20
0.30
0.4
0.5
0.45
0.55
'''
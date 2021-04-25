import numpy as np
import matplotlib.pyplot as plt
import math
from model_utils import *

# Learning Rate decay
def learningRate_decay(alpha,epoch_num):
    a = alpha*(0.8 ** (epoch_num/30))
    return a

# Function to compute the l2 regularization of weights (only W's)
def l2regularization(params,lamb):
    
    L = len(params) // 2
    total = 0 
    
    for l in range(L):
        total += np.sum(params['W'+str(l+1)]**2)
    
    reg_term = (lamb/2)*total
    return reg_term

# Creating a One Hot Matrix
def one_hot(y):
    C = np.unique(y).size
    y_hot = np.eye(C)[:,y.reshape(-1)]
    
    return y_hot

# Generates a list of exponentialy increasing learning rate required by the lrFinder 
def lrList(lr=1e-6,scale=2):
    rate_list = [lr]
    while(lr<0.5):
        lr = lr*scale
        rate_list.append(lr)
    
    return rate_list  

# Converting a vector to dictionary 
def vector_to_dict(theta,p):
    
    L = len(p) // 2
    params = {}
    pos = 0 
    for l in range(1,L+1):
        w_size = p['W'+str(l)].size
        b_size = p['b'+str(l)].size
        params['W'+str(l)] = theta[pos:pos+w_size].reshape(p['W'+str(l)].shape)
        pos += w_size
        params['b'+str(l)] = theta[pos:pos+b_size].reshape(p['b'+str(l)].shape)
        pos +=b_size
        assert(params['W'+str(l)].shape == p['W'+str(l)].shape)
        assert(params['b'+str(l)].shape == p['b'+str(l)].shape)
    
    return params

#Preding the accuracy
def accuracy(Xnorm,Y,test_Xnorm,testY,p):
    pred_y = predict_multiClass(Xnorm,p)
    #print("accuracy at all iterations:",pred_y)
    Acc_train = (pred_y.flatten()==Y.flatten())*100
    #print("Accruacy across all iterations:",Acc_train)
    acc_train = np.mean(pred_y.flatten()==Y.flatten())*100
    print('Accuracy on the Training Set: %s %%' %round(acc_train,2))

    pred_ytest = predict_multiClass(test_Xnorm,p)
    acc_test = np.mean(pred_ytest.flatten()==testY.flatten())*100
    print('Accuracy on the Test Set: %s %%' %round(acc_test,2))
    print ("Neural Network made errors in predicting %s samples out of 600 in the Test Set " 
           % np.count_nonzero(testY != pred_ytest))

# Predicting Multiclass labels
def predict_multiClass(X2,params2):
    AL,_ = forward_prop(X2,params2)
    pred = np.argmax(AL,axis=0)
    #print("accurarcy at this iteration:",pred)
    return pred

def acc_it(X,Y,p):
    pred_y = predict_multiClass(X,p)
    #print("accuracy at all iterations:",pred_y)
    Acc_train = (pred_y.flatten()==Y.flatten())*100
    #print("Accruacy across all iterations:",Acc_train)
    acc_train = np.mean(pred_y.flatten()==Y.flatten())*100
    #print('Accuracy on the Training Set: %s %%' %round(acc_train,2))
    return acc_train


# Converting a dictionary to vector
def dict_to_vector(params,grads):
    
    total = 0
    L = len(params) // 2
    theta = theta_grads = np.empty(0)
    for l in range(1,L+1):
        total += (params['W'+str(l)].size + params['b'+str(l)].size)
        theta = np.append(theta,params['W'+str(l)])
        theta = np.append(theta,params['b'+str(l)])
        
        theta_grads = np.append(theta_grads,grads['dW'+str(l)])
        theta_grads = np.append(theta_grads,grads['db'+str(l)])
    
    assert(total == theta.size)
    return theta.reshape(-1,1),theta_grads.reshape(-1,1)


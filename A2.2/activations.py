import numpy as np
import matplotlib.pyplot as plt
import math

def softmax(Z):
    A = np.exp(Z)/(np.sum(np.exp(Z),axis = 0))
    assert(A.shape == Z.shape)
    
    return A

def norm_softmax(Z):
    b = Z.max(axis=0)
    y = np.exp(Z-b)
    A = y/(np.sum(y,axis = 0))
    assert(A.shape == Z.shape)
    
    return A

# ReLU function
def relu(Z):

    A = np.maximum(0,Z)
    if type(Z) is np.ndarray:
        assert(A.shape == Z.shape)
    
    return A 

 # ReLu Gradient
def relu_grad(Z):
    grad = np.zeros(Z.shape)
    grad[Z>0] =1 
    if type(Z) is np.ndarray:
        assert(grad.shape == Z.shape)
    
    return grad


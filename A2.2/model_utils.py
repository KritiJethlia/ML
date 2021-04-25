
import numpy as np
import matplotlib.pyplot as plt
import math
from activations import *

# Initialize parameters using HE initialization
def params_init(layers):
    params = {}
    L = len(layers) 
    for l in range(L-1):
        params['W'+str(l+1)] = (np.random.randn(layers[l+1],layers[l])
                                *np.sqrt(2/layers[l]))
        
        params['b'+str(l+1)] = np.zeros((layers[l+1],1))
        
        assert(params['W' + str(l+1)].shape == (layers[l+1], layers[l]))
        assert(params['b' + str(l+1)].shape == (layers[l+1], 1))
        
    return params


# Initialization for Adam Optimization
def init_for_adam(params):
    
    L = len(params) //2
    v = {}
    s = {}
    
    for l in range(L):
        
        v["dW" + str(l+1)] = np.zeros(params['W'+str(l+1)].shape)
        v["db" + str(l+1)] = np.zeros(params['b'+str(l+1)].shape)
        s["dW" + str(l+1)] = np.zeros(params['W'+str(l+1)].shape)
        s["db" + str(l+1)] = np.zeros(params['b'+str(l+1)].shape)
        
    return v, s

    
# Forward Propagation
# (L-1) ReLu layers with Softmax output layer
def forward_prop(X,params):
    caches = {}
    A = X
    caches['A'+str(0)] = X
    L = len(params) // 2  # number of layers in the network
    for l in range(1,L):
        A_prev = A
        Z = np.dot(params['W'+str(l)],A_prev) + params['b'+str(l)]
        assert(Z.shape == (params['W'+str(l)].shape[0],A.shape[1]))
        caches['Z'+str(l)] = Z 
        A = relu(Z)
        caches['A'+str(l)] = A

    # Output Layer
    Z = np.dot(params['W'+str(L)],A) + params['b'+str(L)]
    assert(Z.shape == (params['W'+str(L)].shape[0],A.shape[1]))
    caches['Z'+str(L)] = Z 
    AL = norm_softmax(Z)
    
    return AL,caches

# Cost Function
def costFunction(AL,Y,reg=0):
    
    cost = np.mean(-np.sum(Y*np.log(AL),axis=0))
    cost += reg/AL.shape[1]
    
    return cost

# Perform the actual backprop
def compute_grads(dZ,grads,params,caches,l,m,lamb):
    
    dW = (1/m)* np.dot(dZ,caches['A'+str(l-1)].T) + (lamb/m)*params['W'+str(l)]
    assert (dW.shape == params['W'+str(l)].shape)
    grads['dW'+str(l)] = dW
    db = (1/m)* np.sum(dZ,axis=1,keepdims=True)
    assert (db.shape == params['b'+str(l)].shape)
    grads['db'+str(l)] = db
    dA_prev = np.dot(params['W'+str(l)].T,dZ)
    assert(dA_prev.shape == caches['A'+str(l-1)].shape)
    grads['dA'+str(l-1)] = dA_prev
    
    return grads

# Backpropagation
def backprop(AL,Y,caches,params,lamb=0):
    grads = {}
    L = len(params) //2
    m = AL.shape[1]
    # Backprop of the first layer
    dZ = AL-Y
    grads = compute_grads(dZ,grads,params,caches,L,m,lamb)
    
    # Backprop of other layers
    for l in reversed(range(1,L)):
        
        dA = grads['dA'+str(l)]
        dZ = dA * relu_grad(caches['Z'+str(l)])     
        assert(dZ.shape == caches['Z'+str(l)].shape)
        grads = compute_grads(dZ,grads,params,caches,l,m,lamb) 
        
    del grads['dA0']    
    return grads

# Perform gradient checking using Numerical Gradient Estimation  
# to check our implementation of backprop
def gradient_check(X,Y,params,grads):
    
    epsilon = 1e-7
    L = len(params) // 2
    theta,theta_grads = dict_to_vector(params,grads)
    num_params = theta.size
    J_plus = np.zeros((num_params,1))
    J_minus = np.zeros((num_params,1))
    grad_approx = np.zeros((num_params,1))
    
    for i in range(num_params):
        
        thetaplus = np.copy(theta)
        thetaplus[i] += epsilon
        thetaplus_dict = vector_to_dict(thetaplus,params)
        ALplus, _ = forward_prop(X,thetaplus_dict)
        J_plus[i] = costFunction(ALplus,Y)
    
        thetaminus = np.copy(theta)
        thetaminus[i] -= epsilon
        thetaminus_dict = vector_to_dict(thetaminus,params)
        ALminus, _ = forward_prop(X,thetaminus_dict)
        J_minus[i] = costFunction(ALminus,Y)
        
        grad_approx[i] = (J_plus[i] - J_minus[i])/(2*epsilon)
    
    numerator = np.linalg.norm(theta_grads-grad_approx)                              
    denominator = np.linalg.norm(theta_grads) + np.linalg.norm(grad_approx)                         
    difference = numerator/denominator 
    
    if difference > 2e-6:
        print ("\033[91m" + "There is a mistake in the backward propagation! difference = " + str(difference) + "\033[0m")
    else:
        print ("\033[92m" + "Your backward propagation works perfectly fine! difference = " + str(difference) + "\033[0m")

        
# Function to create mini batches
def random_mini_batches(X,Y,batch_size = 64):
    
    mini_batches = []
    m = X.shape[1]
    # Shuffle
    permute = np.random.permutation(m)
    X = X[:,permute]
    Y = Y[:,permute]
    
    # Partition
    complete_batches = math.floor(m/batch_size)
    for k in range(complete_batches):
        batch_X = X[:,k*batch_size:(k+1)*batch_size]
        batch_Y = Y[:,k*batch_size:(k+1)*batch_size]
        batch = (batch_X,batch_Y)
        mini_batches.append(batch)
        
    # Handling the end case
    if m % batch_size != 0:
        batch_X = X[:,(k+1)*batch_size:]
        batch_Y = Y[:,(k+1)*batch_size:]
        batch = (batch_X,batch_Y)
        mini_batches.append(batch)
        
    return mini_batches        


# Adam optimization algorithm for updating weights
def update_parameters_with_adam(params, grads, v, s, t, alpha = 0.01,
                                beta1 = 0.9, beta2 = 0.999,  epsilon = 1e-8):
   
    
    L = len(params) // 2                 
    v_corrected = {}                         
    s_corrected = {}                         
    
    for l in range(L):
        
        v["dW" + str(l+1)] = beta1*v["dW" + str(l+1)] + (1-beta1)*grads["dW" + str(l+1)]
        v["db" + str(l+1)] = beta1*v["db" + str(l+1)] + (1-beta1)*grads["db" + str(l+1)]
       
        v_corrected["dW" + str(l+1)] = v["dW" + str(l+1)] / (1-beta1**t)
        v_corrected["db" + str(l+1)] = v["db" + str(l+1)] / (1-beta1**t)
        
        s["dW" + str(l+1)] = beta2*s["dW" + str(l+1)] + (1-beta2)*(grads["dW" + str(l+1)]**2 )
        s["db" + str(l+1)] = beta2*s["db" + str(l+1)] + (1-beta2)*(grads["db" + str(l+1)]**2 )
       
        s_corrected["dW" + str(l+1)] = s["dW" + str(l+1)] / (1-beta2**t)
        s_corrected["db" + str(l+1)] = s["db" + str(l+1)] / (1-beta2**t)
       
        params["W" + str(l+1)] = params["W" + str(l+1)] - (alpha*(v_corrected["dW" + str(l+1)])/
                                      (np.sqrt(s_corrected["dW" + str(l+1)])+epsilon))
        params["b" + str(l+1)] = params["b" + str(l+1)] - (alpha*(v_corrected["db" + str(l+1)])/
                                      (np.sqrt(s_corrected["db" + str(l+1)])+epsilon))
       

    return params, v, s
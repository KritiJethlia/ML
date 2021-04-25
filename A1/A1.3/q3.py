import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from time import time

def readData(file_name):
    df = pd.read_csv(file_name)
    points, featu = df.shape
    X = df.values [:,:-1]
    y = df.values[:,-1:]
    return X, np.array(y.flat)

def split(X, y):
    points=list(range(len(X)))
    random.shuffle(points)
    train_X = X[points[:int(0.7*len(X)+1)]]
    test_X = X[points[int(0.7*len(X)):]]
    train_y = y[points[:int(0.7*len(X)+1)]]
    test_y = y[points[int(0.7*len(X)):]]
    return train_X, train_y, test_X, test_y

def visualize2D(X, y, w):
    ind = [y==0][0].flat
    c1 = X[np.array(ind)]
    
    ind = [y==1][0].flat
    c2 = X[np.array(ind)]

    plt.scatter(c1[:,1], c1[:,2], s=10, label='negative')
    plt.scatter(c2[:,1], c2[:,2], s=10, label='positive')
    
    x = np.linspace(np.min(X.T[1]),np.max(X.T[1]))

    y_line = (-1*w[1]*x - w[0])/w[2]
    plt.plot(x, y_line, '-r')

    plt.show()

def visualize3D(X, y, w):
    fig = plt.figure()
    ax = plt.axes(projection='3d')

    ind = [y==0][0].flat
    c1 = X[np.array(ind)]
    
    ind = [y==1][0].flat
    c2 = X[np.array(ind)]

    plt.scatter(c1[:,3], c1[:,1], c1[:,2] )
    plt.scatter(c2[:,3], c2[:,1], c2[:,2] )
    
    (x,y_l) = np.meshgrid( np.arange( np.min(X.T[3]), np.max(X.T[3]) ), np.arange( np.min(X.T[1]), np.max(X.T[1]) ))


    z = -1* (w[0]+w[3]*x+w[1]*y_l)/w[2]

    ax.plot_surface(x, y_l, z)
    plt.show()
    

def perceptron(max_itr, train_X, train_y, eta):
    # np.random.seed(9)
    # w = np.random.rand(train_X.shape[1])
    w = np.zeros(train_X.shape[1])
    found = False
    t1=time()
    itr=0
    while(itr < max_itr):
        ch_all = True
        points=list(range(len(train_X)))
        random.shuffle(points)
        train_X = train_X[points]
        train_y = train_y[points]
        for j in range(train_X.shape[0]) :
            itr+=1
            xi = train_X[j]
            expec_y = train_y[j]
            pred_y = 1 if w.T.dot(xi) >= 0 else -1
            expec_y = 1 if expec_y == 1 else -1
            if(pred_y * expec_y < 0):
                # itr+=1
                w +=  eta * expec_y * xi 
                ch_all = False
                # visualize(train_X, train_y, w)
        if(ch_all):
            break
    t2=time()
    print(f"Time Required to train {t2-t1}\n")
    return w

def predict(test_X, test_y, w) :
     y_predict = [f(xi, w) for xi in test_X]
     comp = test_y==y_predict
     comp = comp[comp == True]
     accuracy = comp.shape[0]*100/test_X.shape[0] 
     return comp.shape[0], accuracy

def f(x, w):
    return  1 if x.dot(w) >= 0 else 0


def codeDriver(test_case):
    X,y = readData(test_case)

    temp = np.ones(X.shape[0])
    X = np.c_[temp, X]

    train_X, train_y, test_X, test_y = split(X, y) 
    
    for lr in [0.2]:
        w = perceptron(1000000, train_X, train_y, lr)
        # print(lr,w)
        
        corr_train, acc_train = predict(train_X, train_y, w)
        print(f"Number of correct training predictions: {corr_train} out of {train_X.shape[0]}")
        print(f"Training accuracy : {acc_train}")
        corr_test, acc_test = predict(test_X, test_y, w)
        print(f"Number of correct testing predictions: {corr_test} out of {test_X.shape[0]}")
        print(f"Testing accuracy : {acc_test}")
        if(test_case=='dataset_LP_2.txt'):
            visualize3D(test_X,test_y,w)


if __name__ == '__main__' :
    print("----------------------------------")
    print("Running on First File ....\n")
    codeDriver('dataset_LP_1.txt')
    print("\n\n")
    print("----------------------------------")
    print("Running on Second File ....\n")
    codeDriver('dataset_LP_2.txt')
   



from numpy import array
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.linear_model import Perceptron
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from matplotlib import pyplot


def readData():
    data = np.genfromtxt('dataset_comb.csv', delimiter = ',',dtype=None, encoding="utf8")
    features = data[0][:-1]
    data = data[1:,1:]
    print(data)
    X = data[:,:-1]
    y = data[:,-1]
    X = X.astype(np.float)
    return X, y

def FLD(X, y, kfold):
    clf = LinearDiscriminantAnalysis()
    # scores = cross_val_score(clf, X, y, scoring='accuracy', cv= kfold, n_jobs = 1)
    # print("Fisher Linear Discriminant Results :")
    # print(scores)
    scores = cross_validate(clf, X, y, scoring=('accuracy'), cv= kfold, n_jobs = 1, return_train_score=True)
    print("\n\n -----------------------------------\n")
    print("Fisher Linear Discriminant Results :")
    print("\nTrain Accuracies :" ,scores['train_score'])
    print("Mean :",np.mean(scores['train_score']))
    print("\nTest Accuracies :" ,scores['test_score'])
    print("Mean :",np.mean(scores['test_score']))
    return scores['test_score']
    
    
def percep(X, y, kfold):
    clf = Perceptron(max_iter=10000, tol=1e-9)
    scores = cross_validate(clf, X, y, scoring=('accuracy'), cv= kfold, n_jobs = 1, return_train_score=True)
    print("\n\n -----------------------------------\n")
    print("Linear Perceptron Results : ")
    print("\nTrain Accuracies :" ,scores['train_score'])
    print("Mean :",np.mean(scores['train_score']))
    print("\nTest Accuracies :" ,scores['test_score'])
    print("Mean :",np.mean(scores['test_score']))
    return scores['test_score']

def naiveBayes(X, y, kfold):
    clf = GaussianNB(var_smoothing = 1e-11)
    scores = cross_validate(clf, X, y, scoring=('accuracy'), cv= kfold, n_jobs = 1, return_train_score=True)
    print("\n\n -----------------------------------\n")
    print("Naive Bayes Results :")
    print("\nTrain Accuracies :" ,scores['train_score'])
    print("Mean :",np.mean(scores['train_score']))
    print("\nTest Accuracies :" ,scores['test_score'])
    print("Mean :",np.mean(scores['test_score']))
    return scores['test_score']

def logisticReg(X, y, kfold):
    clf = LogisticRegression(max_iter=150)
    scores = cross_validate(clf, X, y, scoring=('accuracy'), cv= kfold, n_jobs = 1, return_train_score=True)
    print("\n\n -----------------------------------\n")
    print("Logistic Regression Results : ")
    print("\nTrain Accuracies :" ,scores['train_score'])
    print("Mean :",np.mean(scores['train_score']))
    print("\nTest Accuracies :" ,scores['test_score'])
    print("Mean :",np.mean(scores['test_score']))
    return scores['test_score']

def svm(X, y, kfold):
    clf = SVC(kernel='linear')
    scores = cross_validate(clf, X, y, scoring=('accuracy'), cv= kfold, n_jobs = 1, return_train_score=True)
    print("\n\n -----------------------------------\n")
    print("SVM : ")
    print("\nTrain Accuracies :" ,scores['train_score'])
    print("Mean :",np.mean(scores['train_score']))
    print("\nTest Accuracies :" ,scores['test_score'])
    print("Mean :",np.mean(scores['test_score']))
    return scores['test_score']

def ANN(X, y, kfold):
    clf = MLPClassifier(max_iter = 400, learning_rate = 'adaptive')
    # clf = MLPClassifier(max_iter=300)
    scores = cross_validate(clf, X, y, scoring=('accuracy'), cv= kfold, n_jobs = 1, return_train_score=True)
    print("\n\n -----------------------------------\n")
    print("ANN : ")
    print("\nTrain Accuracies :" ,scores['train_score'])
    print("Mean :",np.mean(scores['train_score']))
    print("\nTest Accuracies :" ,scores['test_score'])
    print("Mean :",np.mean(scores['test_score']))
    return scores['test_score']


if __name__ == '__main__' :
    # load()
    kfold = KFold(7, True)
    results = list()
    labels = list()
    X, y = readData()
    results.append(FLD(X, y, kfold))
    labels.append('LDA')
    results.append(percep(X, y, kfold))
    labels.append('Perceptron')
    results.append(naiveBayes(X, y, kfold))
    labels.append('NB')
    results.append(logisticReg(X, y, kfold))
    labels.append('Logistic')
    results.append(svm(X, y, kfold))
    labels.append('SVM')
    results.append(ANN(X, y, kfold))
    labels.append('ANN')
    pyplot.grid(True, linewidth=0.5, color='#e0dede', linestyle='-')
    pyplot.boxplot(results, labels=labels, showmeans=True)
    pyplot.show()





# -----------------------------------

# Fisher Linear Discriminant Results :

# Train Accuracies : [0.98697633 0.98742542 0.98633477 0.98652723 0.98684801 0.98665555
#  0.98659225]
# Mean : 0.9867656509339244

# Test Accuracies : [0.98421863 0.98267898 0.99037721 0.98845266 0.98537336 0.98729792
#  0.98844821]
# Mean : 0.9866924253936803


#  -----------------------------------

# Linear Perceptron Results : 

# Train Accuracies : [0.97639058 0.96407262 0.98165138 0.86777443 0.96554821 0.9712581
#  0.95618424]
# Mean : 0.954697081045154

# Test Accuracies : [0.97806005 0.97575058 0.97921478 0.86797537 0.96073903 0.97074673
#  0.95148248]
# Mean : 0.9548527154118434


#  -----------------------------------

# Naive Bayes Results :

# Train Accuracies : [0.98364021 0.98434593 0.9835119  0.98344774 0.98331943 0.98338359
#  0.98376957]
# Mean : 0.9836311966055382

# Test Accuracies : [0.98498845 0.97959969 0.98460354 0.98498845 0.98498845 0.98344881
#  0.98228725]
# Mean : 0.9835578075032732


#  -----------------------------------

# Logistic Regression Results : 

# Train Accuracies : [0.98928594 0.98960672 0.98960672 0.98967088 0.9883236  0.98973504
#  0.98967154]
# Mean : 0.989414350131133

# Test Accuracies : [0.99076212 0.98922248 0.9899923  0.98845266 0.98691301 0.98883757
#  0.98998845]
# Mean : 0.9891669409682288


#  -----------------------------------

# SVM : 

# Train Accuracies : [0.98922179 0.98890101 0.98915763 0.9893501  0.98954257 0.98922179
#  0.98890172]
# Mean : 0.9891852285171776

# Test Accuracies : [0.98768283 0.99153195 0.99037721 0.9865281  0.98691301 0.98806774
#  0.99075857]
# Mean : 0.9888370591976013


#  -----------------------------------

# ANN : 

# Train Accuracies : [0.91704626 0.9883236  0.97189966 0.98261372 0.98203631 0.97350356
#  0.98242238]
# Mean : 0.9711207835982716

# Test Accuracies : [0.91377983 0.98768283 0.97498075 0.97959969 0.98575828 0.97613549
#  0.98613785]

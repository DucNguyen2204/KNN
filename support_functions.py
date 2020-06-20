# -*- coding: utf-8 -*-
"""
Created on Thu Jun 17 08:57:54 2020

@author: Duc Nguyen
"""

#Set up
import numpy as np
import matplotlib.pyplot as plt
#Function to calculate and return Euclidean distance
def euclidean_distance(u:np.ndarray,v:np.ndarray):
    result = np.sum((u-v)**2)
    return result**(1/2)

#Function to calculate and return Manhattan disctance
def manhattan_distance(u:np.ndarray, v:np.ndarray):
    diff = np.absolute(u-v)
    return np.sum(diff)

#Function to calculate and return accuracy and generalization error
def accuracy_generalized_error(u,v):
    accuracy = np.mean(u==v)
    err = 1-accuracy
    return accuracy, err

def compute_functions(u,v):
    precision = 0
    recall = 0
    f1_score = 0
    true_pos = 0
    false_pos = 0
    true_neg = 0
    false_neg = 0
    for i,j in zip(u,v):
        if i == j and i == 1:
            true_pos+=1
        if i == j and i == 0:
            true_neg+=1
        if i == 0 and j == 1:
            false_pos+=1
        if i == 1 and j == 0:
            false_neg+=1
    
    precision = true_pos/(true_pos+false_pos)
    recall = true_pos/(true_pos+false_neg)
    f1_score = 2*true_pos/(2*true_pos + false_pos+false_neg)
    confusion_matrix = np.array([[true_pos,false_pos],
                                 [true_neg, false_neg]])
    return precision, recall, f1_score, confusion_matrix

def roc_curve(y_true, y_score, pos_label=1):
    
    #y_score can have many duplicates. Only get indices of disticnt value 
    distinct_val, distinct_val_ind = np.unique(y_score, return_index = True)
    threshold_idx = np.r_[distinct_val_ind, y_true.size-1]
    threshold_idx = distinct_val_ind

    #sort y_score then flip it for composing thresholds
    threshold = np.flip(np.sort(y_score[threshold_idx]))
    threshold = np.insert(threshold, 0, (max(y_score)+1))
    #get positive and negative label indices to ease process of computing tps and fps
    pos_indices = np.where(y_true == pos_label)[0]
    neg_indices = np.where(y_true != pos_label)[0]
    
    tps = np.zeros(len(threshold))
    fps = np.zeros(len(threshold))
    fns = np.zeros(len(threshold))
    tns = np.zeros(len(threshold))
    
    for i in range(len(threshold)):
        pos_score = y_score[pos_indices]
        neg_score = y_score[neg_indices]
        for j in range(len(pos_score)):
            if pos_score[j] >= threshold[i]:
                 tps[i] = tps[i]+1
            elif pos_score[j] < threshold[i]:
                fns[i] += 1
        for k in range(len(neg_score)):
            if neg_score[k] >= threshold[i]:
                fps[i] += 1
            elif neg_score[k] < threshold[i]:
                tns[i] += 1
        
    tpr = tps/(tps+fns)
    fpr = fps/(fps+tns)

    plt.plot(fpr,tpr, label = 'roc_curve')
    plt.xlabel('fpr')
    plt.ylabel('tpr')
    plt.legend(loc = 4)
    plt.show
    return fpr,tpr,threshold

def compute_AUC(y_true, y_score, pos_label = 1 ):
    fpr,tpr,threshold = roc_curve(y_true,y_score, pos_label)
    auc = np.trapz(tpr,fpr)
    return auc

def partition(X,y,t):
   mark = int(t*(len(X)-1))
   X_train = X[:-mark,:]
   X_test = X[len(X)-mark:,:]
   y_train = y[:-mark]
   y_test = y[len(X)-mark:]
   
   return X_train, X_test, y_train, y_test

def standardize_data(X):
    for i in range(X.shape[1]):
        mean = np.mean(X[:,i])
        X[:,i] = np.array(X[:,i])
        std = np.std(X[:,i])
        X[:,i] = (X[:,i] - mean) / std
        
    return X


def sFold(folds, data, labels, model, model_agrs, error_function):
    partition = np.array_split(data, folds)
    # print(partition)
    partition_idx = np.asarray(range(len(partition)))
    metrics = model_agrs['metrics']
    weights = model_agrs['weights']
    n = model_agrs['n_neighbors']
    error = []
    accuracy = []
    report = {
        'param':{},
        'expected': [],
        'predicted': [],
        'avg_err': 0,
        'accuracy' :0
        }
    params = {'metric':metrics}
    for i in range(folds):
        test_idx = folds-(i+1)
        train_idx = np.setdiff1d(partition_idx, np.asarray(test_idx))
        partition = np.asarray(partition)
        train_data = np.concatenate(partition[train_idx])
        test_data = partition[test_idx]
        X_test = test_data[:,:-1]
        y_test = test_data[:,-1]
        model.fit(train_data[:,:-1], train_data[:,-1],n,weights,**params)
        y_pred = model.predict(X_test)
        report['expected'].append(y_test)
        report['predicted'].append(y_pred)
        _,_,f1,_ = compute_functions(y_test,y_pred)
        error.append(f1)
        accuracy.append(np.mean(y_test == y_pred))
        
    report['param'] = {'n_neighbors': n, 'weights': weights, 'metrics': metrics}
    report['avg_err'] = np.mean(error)
    report['accuracy']= np.mean(accuracy)
    
    return report
        
        

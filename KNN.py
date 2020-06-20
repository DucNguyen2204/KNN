# -*- coding: utf-8 -*-
"""
Created on Tue Jun  17 01:10:05 2020

@author: Duc Nguyen

"""
import support_functions as sf
import numpy as np
# import numpy as np
class KNN():
    
    def __init__(self):
        pass
    
    def fit(self, X, Y, n, weights = 'uniform', **kwargs):
        self.X = X
        self.Y = Y
        self.n = n
        self.weights = weights
        for key, val in kwargs.items():
            if key == 'metric':
                self.metric = val
                
    def predict(self, X):
        y_pred = np.zeros(len(X))
        for i in range(len(X)):
            distance = []
            for j in range(len(self.X)):
                if self.metric == 'euclidean':
                    distance.append((sf.euclidean_distance(X[i], self.X[j]), self.Y[j]))
                else:
                    distance.append((sf.manhattan_distance(X[i], self.X[j]), self.Y[j]))
            
            distance.sort(key = lambda x:x[0])            
            distances = distance[:self.n]
            k_nearest = [d[1] for d in distances]
            unique, counts = np.unique(k_nearest, return_counts=True)
            y_pred[i] = unique[np.argmax(counts)]
            
        return y_pred
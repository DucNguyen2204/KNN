# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 16:08:06 2020

@author: Duc Nguyen

"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats
import KNN
#read in data
pd.options.display.max_columns = None
df = pd.read_csv('winequality-white.csv', sep = ';')
df.loc[df.quality <= 5, 'quality'] = 0
df.loc[df.quality > 5, 'quality'] = 1
df.info()
print('\nDimension of the data: ', df.shape)

n_row = df.shape[0]
n_col = df.shape[1]

print('\nNo of row: ', n_row)
print('No of col: ', n_col)

# df.hist(bins = 50, figsize = (20,15))

print(df.describe())

df = df.sample(frac = 1)

#Calculate correlation coefficient
def corrfunc(x, y, **kws):
    r, _ = stats.pearsonr(x, y)
    ax = plt.gca()
    ax.annotate("r = {:.2f}".format(r),
                xy=(.1, .6), xycoords=ax.transAxes,
                size = 24)
    
cmap = sns.cubehelix_palette(light=1, dark = 0.1,
                              hue = 0.5, as_cmap=True)

sns.set_context(font_scale=2)

# Pair grid set up
g = sns.PairGrid(df)

# Scatter plot on the upper triangle
g.map_upper(plt.scatter, s=10, color = 'red')

# Distribution on the diagonal
g.map_diag(sns.distplot, kde=False, color = 'red')

# Density Plot and Correlation coefficients on the lower triangle
g.map_lower(sns.kdeplot, cmap = cmap)
g.map_lower(corrfunc);

# =============================================================================
# Looking at the pair plots, citrid acid and free sulfur dioxide are redundant features
# Drop redundant features
# =============================================================================

df = df.drop(['citric acid', 'free sulfur dioxide', 'sulphates', 'pH', 'residual sugar'], axis = 1)

X = df.iloc[:,:-1]
Y = df.iloc[:,-1].values.reshape((df.shape[0],1))
X = X.to_numpy()

knn_clf = KNN.KNN()

X = KNN.sf.standardize_data(X)
X_train_std,X_test_std, y_train, y_test = KNN.sf.partition( X, Y, 0.2)

knn_clf.fit(X_train_std, y_train, n = 9, metric = 'euclidean')
y_pred_std = knn_clf.predict(X_test_std)

accuracy_std, error_std = KNN.sf.accuracy_generalized_error(y_test, y_pred_std)

print('Standardized data accuracy: ', accuracy_std)
print('\nGeneralized Error: ' , error_std)

precision_std, recall_std, f1_std, confusion_matrix_std = KNN.sf.compute_functions(y_test, y_pred_std)

print('\nStandardized Precision: ', precision_std)
print('\nStandardized Recall ', recall_std)
print('\nStardardized f1 score:', f1_std)
print('\nStandardized Confusion Matrix: \n', confusion_matrix_std)

report = KNN.sf.sFold(5, np.concatenate((X,Y),axis = 1), Y, knn_clf,{'n_neighbors': 5, 'weights': 'uniform', 'metrics': 'euclidean'}, 'f1')
print('Average f1 error function: ' , report['avg_err'])
print('Accuracy: ', report['accuracy'])


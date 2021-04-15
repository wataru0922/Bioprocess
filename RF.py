#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 15:26:25 2020

@author: biodeep
"""
from __future__ import print_function

import pandas as pd
#from pandas import Series,DataFrame

import os.path
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf

from tensorflow import keras
from keras.datasets import fashion_mnist
from keras.models import Sequential, Model
from keras import metrics

from keras.layers import Input, Dense, Dropout
from keras.optimizers import RMSprop
from keras.optimizers import Adam
from keras import optimizers
#from keras.callbacks import ModelCheckpoint


import keras.callbacks
from keras.callbacks import CSVLogger
from keras.callbacks import EarlyStopping
from keras import backend as K
from keras import optimizers
#from keras.utils.visualize_util import plot
from keras.utils import plot_model
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
import scipy.stats
import keras.callbacks
import keras.backend.tensorflow_backend
from keras.models import model_from_yaml
from keras.models import model_from_json

import matplotlib.backends.backend_pdf
dirpath = os.getcwd()
dirname = "RF/Growth_standardized/18h-Exp1"
path = os.path.join(dirpath,dirname)
if not os.path.exists(path):
    os.makedirs(path)

def PDF_save (Fig, FilePath):
    pdf = matplotlib.backends.backend_pdf.PdfPages(FilePath)
    pdf.savefig( Fig )
    pdf.close()
    return
#CSVファイルの読み込み
DataSet = pd.read_csv ("DataMatrix201027.csv", header = 0)
Z = DataSet.loc [:,"x001":"x078"]
Z_st=scipy.stats.zscore(Z)
Z_st=np.nan_to_num(Z_st)
Z_st=pd.DataFrame(Z_st)
Z = pd.DataFrame (Z)
DataSet_y = DataSet.loc [:,"y2"]


ValIndexStart = 0
ValIndexEnd = 2
Data = DataSet.drop(index = DataSet.index [ValIndexStart:ValIndexEnd+1])
Val_data = DataSet.loc [ValIndexStart:ValIndexEnd,:]
x =  Z_st.drop(index = DataSet.index [ValIndexStart:ValIndexEnd+1])
x = x.astype(np.float)
xv = Z_st.loc [ValIndexStart:ValIndexEnd,:]

x = Data.loc [:,"x001":"x078"].astype(np.float)
y = Data.loc [:,"y1"].astype(np.float)
#xv = Xv.loc[:,"X001":"X205"].astype(np.float)
yv = Val_data.loc[:,"y1"].astype(np.float)
#print (DataSetVal)
x_train, x_test, y_train, y_test = train_test_split(x, y, 
test_size = 0.15)

from sklearn.ensemble import RandomForestRegressor
#import mglearn
#from sklearn.ensemble import RandomForestClassifer
forest = RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=10,
           max_features=6, max_leaf_nodes=None,
           min_impurity_decrease=0.0, min_impurity_split=None,
           min_samples_leaf=1, min_samples_split=3,
           min_weight_fraction_leaf=0.0, n_estimators=300, n_jobs=1,
           oob_score=False, random_state=2525, verbose=0, warm_start=False)
Fit = forest.fit (x_train, y_train)

#fig, axes = plt.subplots (3, 4, figsize = (12,12))
#for i, (ax, tree) in enumerate (zip(axes.ravel(),forest.estimators_)):
#    ax.set_title ("Tree{}".format(i))
#    mglearn.plots.plot_tree_partition (x_train, y_train, tree, ax = ax)

#mglearn.plots.plot_2d_separator (forest, x_train, fill = True, ax = axes [-1,-1], alpha = .4)
#axes [-1,-1].set_title ("Random Forest")
#mglearn.discreate_scatter (x_train [:,0], x_train [:,1], y_train)

# 予測値を計算
y_train_pred = forest.predict(x_train)
y_test_pred = forest.predict(x_test)
y_val = forest.predict(xv)
# MSEの計算
from sklearn.metrics import mean_squared_error
print('MSE train : %.5f, test : %.5f, validation:%.5f' % (mean_squared_error(y_train, y_train_pred), mean_squared_error(y_test, y_test_pred),mean_squared_error(yv, y_val)) )
# R^2の計算
from sklearn.metrics import r2_score
print('R2 train : %.5f, test : %.5f, validation:%.5f' % (r2_score(y_train, y_train_pred), r2_score(y_test, y_test_pred), r2_score(yv, y_val)) )

# Calculate scores for calibration and cross-validation
score_c = r2_score(y_train, y_train_pred, multioutput='variance_weighted')
score_cv = r2_score(y_test, y_test_pred, multioutput='variance_weighted')
score_val = r2_score (yv, y_val, multioutput='variance_weighted')

# Calculate mean square error for calibration and cross validation
mse_c = mean_squared_error(y_train, y_train_pred)
mse_cv = mean_squared_error(y_test, y_test_pred)
mse_val = mean_squared_error (yv, y_val)

scores = mse_c, mse_cv, mse_val, score_c, score_cv, score_val
scores = pd.DataFrame (scores, index=['mse_train','mse_test', 'mse_val', 'r2_train', 'r2_test','r2_val'])
scores.to_csv (path + "/scores.csv")

#scores.loc [:,'mse_train'] = mse_c
#scores.loc [:,'mse_test'] = mse_cv
#scores.loc [:,'mse_val'] = mse_val
#scores.loc [:,'r2_train'] = score_c
#scores.loc [:,'r2_test'] = score_cv
#scores.loc [:,'r2_val'] = score_val

fti = Fit.feature_importances_ 


with plt.style.context(('seaborn-colorblind')):
    fig, ax = plt.subplots(figsize=(4, 4))
    plt.gca().yaxis.set_tick_params(which='both', direction='in',bottom=True, top=True, left=True, right=True)
    plt.gca().xaxis.set_tick_params(which='both', direction='in',bottom=True, top=True, left=True, right=True)
    plt.scatter (y_train_pred, y_train, c = 'yellow', edgecolors = 'k', label = "Training Data")
    plt.scatter(y_test_pred, y_test, c='red', edgecolors='k', label = "Test Data")
    plt.scatter (y_val, yv, c = 'blue', edgecolors = 'k', label = "Validation Data")
    #ax.plot(z[1]+z[0]*y, y_test, c='red', linewidth=1)
    #ax.plot(z2[1]+z2[0]*yv, yv, c='blue', linewidth=1)
    #ax.plot(y, y, color='green', linewidth=1, label = "")
    plt.title('RSME (val): %.5f '% (mse_val))
    plt.xlabel('Predicted Value')
    plt.ylabel('Measured Value') 
    plt.legend(loc = 'upper left')
    #ax.plot(z2[1]+z2[0]*yv, yv, c='blue', linewidth=1)
    plt.show()
    PDF_save (fig, path + "/fig1.pdf")

#fti = pd.DataFrame (fti)
#y_position = np.arange(len(fti))
x_feature = pd.DataFrame (Z.columns.values)
labels = DataSet.loc [:,"Unnamed: 0"] 

result = pd.DataFrame (columns=['x_feature','fti'])

result.loc [:,'x_feature'] = x_feature [0] 

result.loc[:,'fti'] = fti
fti_Max = max (fti)

result = result.sort_values('fti')
result.to_csv (path + '/SigScore.csv')
result_filtered = result.iloc [len(result)-20:len(result),:]
#result.loc[:,'y_position'] = y_position
y_position = np.array ([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19])
with plt.style.context(( 'seaborn-muted')):
    fig, ax = plt.subplots (figsize = (5,5))
    plt.gca().yaxis.set_tick_params(which='both', direction='in',bottom=True, top=True, left=True, right=True)
    plt.gca().xaxis.set_tick_params(which='both', direction='in',bottom=True, top=True, left=True, right=True)
    rects = ax.barh(y_position, result_filtered.loc[:,'fti'],
                     align='center',
                     height=0.5,
                     tick_label=result_filtered.loc [:,'x_feature'])
    ax.set_xlim([0, max (fti)])
    ax.xaxis.grid(True, linestyle='--', which='major',
                   color='grey', alpha=.25)
    ax.set_ylabel('features')
    ax.set_xlabel ('Scores')
    PDF_save (fig, path + "/fig2SigFeatures.pdf")
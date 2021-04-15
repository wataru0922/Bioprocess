#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 20:48:43 2020

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
from keras.layers import merge
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
from sklearn import preprocessing

import matplotlib.backends.backend_pdf
dirpath = os.getcwd()
dirname = "DNN/GFP_sigmid3"
path = os.path.join(dirpath,dirname)
if not os.path.exists(path):
    os.makedirs(path)

def PDF_save (Fig, FilePath):
    pdf = matplotlib.backends.backend_pdf.PdfPages(FilePath)
    pdf.savefig( Fig )
    pdf.close()
    return
#CSVファイルの読み込み
keras.backend.clear_session()
DataSet = pd.read_csv ("L81_Deepwell.csv", header = 0, index_col=0)
Z = DataSet.loc [:,"PottasiumPhosphate":"Thiamine"]
Target = "GFP"
Z_st=scipy.stats.zscore(Z)
Z_st=np.nan_to_num(Z_st)
Z_st=preprocessing.minmax_scale(Z_st)
Z_st=pd.DataFrame(Z_st)
Z = pd.DataFrame (Z)
Z_st.columns = Z.columns
Z_st.index = Z.index
DataSet_y = DataSet.loc [:,Target]/1000

DataSet_raw = DataSet_y
DataSet_raw = pd.DataFrame(DataSet_y)
DataSet_raw = pd.DataFrame (DataSet_raw)
DataSet_raw = DataSet_raw.join(Z_st)
DataSet_raw.to_csv (path+"/DataSet.csv")



import scipy.stats as stats

#パラメトリック，ノンパラメトリックな一元配置分散分析を行う関数Auto_ANOVA
def Auto_ANOVA(*data, param = None, message = True, bart_alpha = 0.05, alpha = 0.05):

    #入力引数
    #*data      ： 分析を行うデータ(2つ以上)
    #
    #param      ： 正規性の仮定(ブールで設定．Default = None)
    #      True  -> F検定
    #      False -> Kruskal-Wallis検定
    #      None  -> Bartlett検定によりF検定あるいはKruskal-Wallis検定を判断(メッセージを表示)
    #
    #message    : メッセージの表示(ブールで設定．Default = True)
    #        True  -> 正規性の仮定とBartlett検定の結果が合致しない場合にメッセージを表示．
    #        False -> 分布の仮定にしたがった検定結果のみを出力
    #
    #bart_alpha : Bartlett検定(複数群の分布の正規性を検定)の有意水準．Default = 0.05
    #
    #alpha      : ANOVAの有意水準．Default = 0.05

    #データが2群未満の場合
    if len(data) < 2:
        #エラーを返して終了
        raise ValueError('need at least two array to calculate')
    else:
        #正規性を仮定する場合
        if param == True:
            #メッセージの表示がTrueのとき
            if message == True:
                #Bartlett検定の結果が仮定に沿わない場合
                if stats.bartlett(*data).pvalue < bart_alpha:
                    #メッセージの表示
                    print('Warning : Bartlett test returned a significant difference.')
                    print('          Please consider to practice non-parametric ANOVA.')

            #F検定を行う
            f, p = stats.f_oneway(*data)

        #正規性を仮定しない場合
        elif param == False:
            #メッセージの表示がTrueのとき
            if message == True:
                #Bartlett検定の結果が仮定に沿わない場合
                if stats.bartlett(*data).pvalue >= bart_alpha:
                    #メッセージの表示
                    print('Warning : Bartlett test did not return a significant difference.')
                    print('          Please consider to practice parametric ANOVA.')

            #Kruskal-Wallis検定を行う
            f, p = stats.kruskal(*data)

        #Noneのとき
        else:
            #正規性仮説が棄却される場合
            if stats.bartlett(*data).pvalue < bart_alpha:
                #メッセージを表示
                print('Message : Bartlett test returned a significant difference.')
                print('          This function practices non-parametric ANOVA.')

                #Kruskal-Wallis検定
                f, p = stats.kruskal(*data)

            #正規性仮説が採択される場合
            else:
                #メッセージを表示
                print('Message : Bartlett test did not return a significant difference.')
                print('          This function practices parametric ANOVA.')

                #F検定
                f, p = stats.f_oneway(*data)

        #有意差の有無(ブール)
        significance = (p < alpha)

        #出力引数
        #f　　　　　　　　　　:　F値
        #p　　　　　　　　　　:　p値
        #significance　:　有意差の有無
        return f, p, significance


def Anova (data):
    wide, h = Z_st.shape
    k = int (wide /3)

    p_val_add = []
    n = int((h+1)/3)
    for i in range (h):
        p_Data = []
        p_Data = data.iloc[:,i]
        p_Data = np.asarray (p_Data)
        p_Data = p_Data. reshape (k, 3)
        p_val = [] 
        p_val = Auto_ANOVA (*[p_Data [j] for j in range (k)], param = True) 
        p_val_add.append(p_val)
    
    return p_val_add

p_values = Anova (Z_st)
p_values = pd.DataFrame (p_values)
p_values = p_values.rename(columns={0: 'F_value'})
p_values = p_values.rename(columns={1: 'p_value'})
p_values = p_values.rename(columns={2: 'Significance'})
p_values.to_csv  (path + "/p_values.csv")

DataSet_raw_t = DataSet_raw.T
DataSet_raw_t = DataSet_raw_t.iloc [1:,:]
DataSet_raw_t = DataSet_raw_t.reset_index(drop=True)
DataSet_raw_t.columns = Z_st.index
DataSet_raw_t = DataSet_raw_t.join (p_values)
DataSet_raw_t.index = DataSet_raw.columns[1:]
DataSet_raw_t.to_csv (path + "/anova_result.csv")
DataSet_raw_t = DataSet_raw_t [DataSet_raw_t.Significance == True]

X_t = DataSet_raw_t.iloc [:,:(len (DataSet_raw_t.columns)-3)]
#X_t = X_t.iloc [1:,:]
x = X_t.T
Data = DataSet

DataSet_Z_st = DataSet_y
DataSet_Z_st = pd.DataFrame (DataSet_Z_st)
DataSet_Z_st = DataSet_Z_st.join(Z_st)

x = x.astype(np.float)
y = Data.loc [:,Target].astype(np.float)/1000


x_train, x_test, y_train, y_test = train_test_split(x, y, 
test_size = 0.05)

inputs = Input(shape=(len(x.columns),))

x1= Dense (256, activation='sigmoid',kernel_initializer='he_normal',
                input_shape = (len(x.columns),),
                name = 'dense1-1')(inputs)

x2= Dense(256, activation='sigmoid', kernel_initializer='he_normal',
                input_shape = (256,),
                name = 'dense2-1')(x1)
x3= Dense(128, activation='sigmoid', kernel_initializer='he_normal',
                input_shape = (256,),
                name = 'dense3-1')(x2)
x4 = Dense(64, activation='sigmoid', kernel_initializer='he_normal',
                input_shape = (128,),
                name = 'dense4-1')(x3)
prediction = Dense(1, activation='relu', kernel_initializer='he_normal',
                input_shape = (8,), name = 'dense6')(x4)
model = Model(input=[inputs], output=[prediction])
model.summary()
print("\n")

plot_model(model, to_file=path +'/model_plot.png', show_shapes=True, show_layer_names=True)

#ニューラルネットワークの実装②
#model.compile(optimizer='adam', loss='mse')
model.compile(loss='mean_squared_error',optimizer=Adam(lr=0.01, beta_1=0.9, beta_2=0.999,
                                                       epsilon=1e-08, decay=0.0, amsgrad=False),
              metrics=['mae', 'accuracy'])

#visualize the model
plot_model(model, to_file= path + '/model.png', show_shapes = True)

callbacks = []
# CSVLogger
callbacks.append(CSVLogger(path + "/log.csv"))

from keras.callbacks import ModelCheckpoint
checkpointer = ModelCheckpoint(filepath=path + "/best_weight.h5", monitor="val_loss", save_best_only=True, save_weights_only=True)
callbacks = [checkpointer]
# fitting
history = model.fit(x_train, y_train,batch_size=243,epochs=10000,verbose=1,validation_data=(x_test, y_test),  callbacks = callbacks)

#ニューラルネットワークの推論
score = model.evaluate(x_test,y_test,verbose=1)
print("\n")
print("Test loss:",score[0])
print("Test accuracy:",score[1])

sample = x_test
sample0 = x_train
sample = np.array(sample)
sample0 = np.array(sample0)

print('save the architecture of a model')
model.save((path +'/model.h5'), include_optimizer=False)
yaml_string = model.to_yaml()
open(os.path.join(path +'/model.yaml'), 'w').write(yaml_string)
json_string = model.to_json()
open(os.path.join(path +'/model.json'), 'w').write(json_string)
print('save weights')
model.save_weights(os.path.join(path +'/model_weights.hdf5'))

#print (sample)
print("\n")
predict = model.predict_on_batch(sample)
print("\n")
print("--予測値--")
print(predict)
print("\n")

def plot_history(history):
    print(history.history.keys())

    # 精度の履歴をプロット
    with plt.style.context(('seaborn-colorblind')):
        fig, ax = plt.subplots(figsize=(4, 4))
        plt.gca().yaxis.set_tick_params(which='both', direction='in',bottom=True, top=True, left=True, right=True)
        plt.gca().xaxis.set_tick_params(which='both', direction='in',bottom=True, top=True, left=True, right=True)
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('model accuracy')
        plt.xlabel('epoch')
        plt.ylabel('accuracy')
        plt.legend(['acc', 'val_acc'], loc='lower right')
        PDF_save (fig, path + "/accuracy.pdf")
        plt.show()

    # 損失の履歴をプロット
    with plt.style.context(('seaborn-colorblind')):
        fig, ax = plt.subplots(figsize=(4, 4))
        plt.gca().yaxis.set_tick_params(which='both', direction='in',bottom=True, top=True, left=True, right=True)
        plt.gca().xaxis.set_tick_params(which='both', direction='in',bottom=True, top=True, left=True, right=True)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend(['loss', 'val_loss'], loc='lower right')
        PDF_save (fig, path + "/loss.pdf")
        plt.show()

# 学習履歴をプロット
plot_history(history)
keras.backend.clear_session()
json_string=open(path +'/model.json').read()
model2=model_from_json(json_string)
model2.load_weights(path +'/best_weight.h5')
sample = np.array(sample)

model2.compile(loss='mean_squared_error',optimizer=RMSprop(),metrics=['mae', 'accuracy'])
#history = model2.fit(x_train, y_train,batch_size=1,epochs=1,verbose=1,validation_data=(x_test, y_test),  callbacks = callbacks)
score2 = model2.evaluate(x_test,y_test,verbose=1)
print("\n")
print("Best loss:",score2[0])
print("Best accuracy:",score2[1])
Score2 = pd.DataFrame (score2)
Score2.to_csv (path +"/BestScore.csv")
predict1 = model2.predict_on_batch(sample)

#predictをcsvに書き出し
predict1 = pd.DataFrame (predict1)
predict1.to_csv(path +"/predict_test.csv")

predict0 = model2.predict_on_batch(sample0)
#predictをcsvに書き出し
predict0 = pd.DataFrame (predict0)
predict0.to_csv(path +"/predict_train.csv")
predict0.to_csv(path +"/predict_cv.csv")

from sklearn.metrics import mean_squared_error, r2_score
# Calculate scores for calibration and cross-validation
score_c = r2_score(y_train, predict0, multioutput='variance_weighted')
score_cv = r2_score(y_test, predict1, multioutput='variance_weighted')
mse_c = mean_squared_error(y_train, predict0)
mse_cv = mean_squared_error(y_test, predict1)
print('R2 calib: %5.3f'  % score_c)
print('R2 test: %5.3f'  % score_cv)
print('MSE calib: %5.3f' % mse_c)
print('MSE test: %5.3f' % mse_cv)
MSE_R2 = score_c, score_cv, mse_c, mse_cv
MSE_R2 = pd.DataFrame (MSE_R2)
MSE_R2.index = ["R2_train", "R2_test", "MSE_train", "MSE_test"]
MSE_R2.to_csv(path +"/MSE_R2.csv")

with plt.style.context(('seaborn-colorblind')):
    fig, ax = plt.subplots(figsize=(4, 4))
    plt.gca().yaxis.set_tick_params(which='both', direction='in',bottom=True, top=True, left=True, right=True)
    plt.gca().xaxis.set_tick_params(which='both', direction='in',bottom=True, top=True, left=True, right=True)
    plt.scatter (predict0, y_train, c = 'yellow', edgecolors = 'k', label = "Training Data")
    plt.scatter(predict1, y_test, c='red', edgecolors='k', label = "Test Data")

    plt.title('MSE (CV): '+str(score2[0]))
    plt.xlabel('Predicted Value')
    plt.ylabel('Measured Value') 
    plt.legend(loc = 'upper left')
    plt.show()
    PDF_save (fig, path + "/fig1.pdf")

#visualize_cam(model2, 3, filter_indices, seed_input, penultimate_layer_idx=None, \
#    backprop_modifier=None, grad_modifier=None)
w_list = [1,2,3,4]
for i in w_list:
    with plt.style.context(('seaborn-colorblind')):
        fig, ax = plt.subplots(figsize=(4, 4))
        plt.gca().yaxis.set_tick_params(which='both', direction='in',bottom=True, top=True, left=True, right=True)
        plt.gca().xaxis.set_tick_params(which='both', direction='in',bottom=True, top=True, left=True, right=True)
        w1 = model2.layers[i].get_weights()[0]
        plt.imshow(w1, cmap='coolwarm', interpolation='nearest')
        plt.colorbar()
        plt.figure()
        FileName = "/Weight%i.pdf" %(i)
        PDF_save (fig, path + FileName)
    with plt.style.context(('seaborn-colorblind')):
        fig, ax = plt.subplots(figsize=(4, 4))
        plt.gca().yaxis.set_tick_params(which='both', direction='in',bottom=True, top=True, left=True, right=True)
        plt.gca().xaxis.set_tick_params(which='both', direction='in',bottom=True, top=True, left=True, right=True)
        w1 = model2.layers[i].get_weights()[0]
        plt.plot((w1**2).mean(axis=1), 'o-')
        ax.set_ylabel('Sum Weitght')
        ax.set_xlabel ('Units')
        plt.show()
        FileName = "/WeightSum%i.pdf" %(i)
        PDF_save (fig, path + FileName)

#Caluculate significant score (mean increased loss)
Z_st = x
SumSigScore = []
NewDataSetX = []
Shape = Z_st.shape [1]
Length = Z_st.shape [0]
for i in range (Shape):

    for j in range (Length):
        NewDataSetX =  Z_st.copy()
        NewDataSetX = pd.DataFrame (NewDataSetX)       
        X_value =Z_st.iloc [j,i] 
        NewDataSetX.iloc[:,i] = X_value
        NewDataSetX = NewDataSetX.astype (np.float)
        DataSet_y = DataSet_y.astype (np.float)
        SigScore = model2.evaluate (NewDataSetX, DataSet_y, verbose=0)
        SumSigScore.append(SigScore[0])

SumSigScore = np.array (SumSigScore)
SumSigScore = SumSigScore.reshape (Shape,Length)

from statistics import mean, median,variance,stdev
AveSumSigScore = []
for i in range (Shape):
    AveSumSigScore.append (mean (SumSigScore[i]))

AveSumSigScore = pd.DataFrame (AveSumSigScore)
AveSumSigScore.to_csv (path + "/SignificantScores.csv")

x_feature = x.columns.values
labels = x.index

result = pd.DataFrame (columns=['x_feature','Score'])

result.loc [:,'x_feature'] = x_feature

result.loc[:,'Score'] = AveSumSigScore
Score_Max = max (AveSumSigScore[0])
result = result.sort_values('Score')

result_filtered = result.iloc [len(result)-20:len(result),:]
#result.loc[:,'y_position'] = y_position
y_position = np.array ([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19])
with plt.style.context(( 'seaborn-muted')):
    fig, ax = plt.subplots (figsize = (5,5))
    rects = ax.barh(y_position, result_filtered.loc[:,'Score'],
                     align='center',
                     height=0.5,
                     tick_label=result_filtered.loc [:,'x_feature'])
    ax.set_xlim([0, Score_Max])
    ax.xaxis.grid(True, linestyle='--', which='major',
                   color='grey', alpha=.5)
    ax.set_ylabel('features')
    ax.set_xlabel ('Scores')
    PDF_save (fig, path + "/figSigScores.pdf")

#Caluculate averaged sensitivity
Sample = 0
SampleN = len(Z_st.index)
Sensitivity_sum = []
#def SensitivityAnalysis (SampleN, Data):
for index, Data in enumerate(Z_st.columns):
    predj_sum = []
    
    i = 0
    j = 0
    for j in range (SampleN):
        DataSetSense = []
        for i in range (100):
            DataSetSense.append(Z_st.iloc [j,:])
        DataSetSense = pd.DataFrame (DataSetSense)
        sense = np.arange(min(Z_st.loc[:,Data]),max(Z_st.loc[:,Data]),(max(Z_st.loc[:,Data])-min(Z_st.loc[:,Data]))/100)
        DataSetSense.loc [:,Data] = sense [0:100]
        pred1 = model2.predict (DataSetSense)
        predj = pred1[:,0]
        predj_sum.append (predj)
    predj_sum_pd = pd.DataFrame(predj_sum)
    Sensitivity = np.average (predj_sum_pd.iloc[:,99]-predj_sum_pd.iloc[:,0])
    Sensitivity_sum.append (Sensitivity.astype(np.float))
Sensitivity_sum = pd.DataFrame (Sensitivity_sum)
Sensitivity_sum.index = Z_st.columns
Sensitivity_sum.to_csv (path + "/Sensitivity.csv")

y_position = Z_st.columns
with plt.style.context(( 'seaborn-muted')):
    fig, ax = plt.subplots (figsize = (5,10))
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    rects = ax.barh(y_position, Sensitivity_sum[0],
                     align='center',
                     height=0.5,
                     tick_label=Sensitivity_sum.index)
    ax.set_xlim([min(Sensitivity_sum[0]), max(Sensitivity_sum[0])])
    ax.xaxis.grid(True, linestyle='--', which='major',
                   color='grey', alpha=.5)
    ax.set_ylabel('features')
    ax.set_xlabel ('Averaged sensitivity')
    PDF_save (fig, path + "/FigSensitivity.pdf")

#Caluculate sensitivity (each components)
def sensitivity (data):
    SampleN = len(x.index)
    predj_sum = []
    i = 0
    j = 0
    for j in range (SampleN):
        DataSetSense = []
        for i in range (100):
            DataSetSense.append(x.iloc [j,:])
        DataSetSense = pd.DataFrame (DataSetSense)
        sense = np.arange(min(x.loc[:,Data]),max(x.loc[:,Data]),(max(x.loc[:,Data])-min(x.loc[:,Data]))/100)
        DataSetSense.loc [:,Data] = sense [0:100]
    
        pred1 = model2.predict (DataSetSense)
        predj = pred1[:,0]
        predj_sum.append (predj)
    predj_sum = pd.DataFrame(predj_sum)
    
    colorlist = ["b","b","g","g","r","r","c","c","m","m","y","y","k","k",
                 "b","b","g","g","r","r","c","c","m","m","y","y","k","k",
                 "b","b","g","g","r","r","c","c","m","m","y","y","k","k"]
    linelist = ["solid","solid","solid","solid","solid","solid","solid",
                "solid","solid","solid","solid","solid","solid","solid",
                "dashed","dashed","dashed","dashed","dashed","dashed","dashed",
                "dashed","dashed","dashed","dashed","dashed","dashed","dashed",
                "dashdot","dashdot","dashdot","dashdot","dashdot","dashdot","dashdot",
                "dashdot","dashdot","dashdot","dashdot","dashdot","dashdot","dashdot",
                "dotted","dotted","dotted","dotted","dotted","dotted","dotted",
                "dotted","dotted","dotted","dotted","dotted","dotted","dotted"]
    fig, ax = plt.subplots (figsize = (5,5))
    for j in range (len(x.index)):
        plt.plot (sense, predj_sum.iloc [j,:])#, color = colorlist [j], linestyle = linelist [j])
    plt.title (Data)
    plt.xlabel('Concentrations')
    plt.ylabel('GFP')
    plt.ylim (0,)
    dirpath = os.getcwd()
    dirname = path + "/sensitivity"
    path2 = os.path.join(dirpath,dirname)
    if not os.path.exists(path2):
        os.makedirs(path2)
    FileName = Data 
    plt.savefig(path2 + "/%sg.eps" %FileName)
    plt.show()
    predj_sum_pd.to_csv (path2 + "/%s.csv" %FileName)

for index, Data in enumerate (Z_st.columns):
    sensitivity (Data)
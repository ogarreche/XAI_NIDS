

print('---------------------------------------------------------------------------------')
print('SVM')
print('---------------------------------------------------------------------------------')


print('---------------------------------------------------------------------------------')
print('Importing Libraries')
print('---------------------------------------------------------------------------------')

import tensorflow as tf
import os
from matplotlib import pyplot as plt
import numpy as np
# import pafy
import pandas as pd
#import csv
import math
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import *
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
from keras.preprocessing import sequence
from keras.preprocessing.sequence import TimeseriesGenerator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from collections import Counter
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import auc
from sklearn import svm, datasets
import sklearn.model_selection as model_selection
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import time
import shap
np.random.seed(0)

from sklearn.calibration import CalibratedClassifierCV
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDClassifier



print('---------------------------------------------------------------------------------')
print('Defining metrics')
print('---------------------------------------------------------------------------------')

def ACC(TP,TN,FP,FN):
    Acc = (TP+TN)/(TP+FP+FN+TN)
    return Acc

def PRECISION(TP,FP):
    Precision = TP/(TP+FP)
    return Precision
def RECALL(TP,FN):
    Recall = TP/(TP+FN)
    return Recall
def F1(Recall, Precision):
    F1 = 2 * Recall * Precision / (Recall + Precision)
    return F1
def BACC(TP,TN,FP,FN):
    BACC =(TP/(TP+FN)+ TN/(TN+FP))*0.5
    return BACC
def MCC(TP,TN,FP,FN):
    MCC = (TN*TP-FN*FP)/(((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))**.5)
    return MCC
def AUC_ROC(y_test_bin,y_score):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    auc_avg = 0
    counting = 0
    for i in range(n_classes):
      fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
      auc_avg += auc(fpr[i], tpr[i])
      counting = i+1
    return auc_avg/counting

print('---------------------------------------------------------------------------------')
print('Defining features')
print('---------------------------------------------------------------------------------')


req_cols = ['FLOW_DURATION_MILLISECONDS','FIRST_SWITCHED',
            'TOTAL_FLOWS_EXP','TCP_WIN_MSS_IN','LAST_SWITCHED',
            'TCP_WIN_MAX_IN','TCP_WIN_MIN_IN','TCP_WIN_MIN_OUT',
           'PROTOCOL','TCP_WIN_MAX_OUT','TCP_FLAGS',
            'TCP_WIN_SCALE_OUT','TCP_WIN_SCALE_IN','SRC_TOS',
            'DST_TOS','FLOW_ID','L4_SRC_PORT','L4_DST_PORT',
           'MIN_IP_PKT_LEN','MAX_IP_PKT_LEN','TOTAL_PKTS_EXP',
           'TOTAL_BYTES_EXP','IN_BYTES','IN_PKTS','OUT_BYTES','OUT_PKTS',
            'ALERT']
#----------------------------------------
#Load Databases from csv file
print('---------------------------------------------------------------------------------')
print('Loading Databases')
print('---------------------------------------------------------------------------------')
#Denial of Service
df0 = pd.read_csv ('sensor_db/dos-03-15-2022-15-44-32.csv', usecols=req_cols)
df1 = pd.read_csv ('sensor_db/dos-03-16-2022-13-45-18.csv', usecols=req_cols)
df2 = pd.read_csv ('sensor_db/dos-03-17-2022-16-22-53.csv', usecols=req_cols)
df3 = pd.read_csv ('sensor_db/dos-03-18-2022-19-27-05.csv', usecols=req_cols)
df4 = pd.read_csv ('sensor_db/dos-03-19-2022-20-01-53.csv', usecols=req_cols)
df5 = pd.read_csv ('sensor_db/dos-03-20-2022-14-27-54.csv', usecols=req_cols) 

#Normal
df7 = pd.read_csv  ('sensor_db/normal-03-15-2022-15-43-44.csv', usecols=req_cols)
df8 = pd.read_csv  ('sensor_db/normal-03-16-2022-13-44-27.csv', usecols=req_cols)
df9 = pd.read_csv  ('sensor_db/normal-03-17-2022-16-21-30.csv', usecols=req_cols)
df10 = pd.read_csv ('sensor_db/normal-03-18-2022-19-17-31.csv', usecols=req_cols)
df11 = pd.read_csv ('sensor_db/normal-03-18-2022-19-25-48.csv', usecols=req_cols)
df12 = pd.read_csv ('sensor_db/normal-03-19-2022-20-01-16.csv', usecols=req_cols) 
df13 = pd.read_csv ('sensor_db/normal-03-20-2022-14-27-30.csv', usecols=req_cols) 

#PortScanning

df14 = pd.read_csv  ('sensor_db/portscanning-03-15-2022-15-44-06.csv', usecols=req_cols)
df15 = pd.read_csv  ('sensor_db/portscanning-03-16-2022-13-44-50.csv', usecols=req_cols)
df16 = pd.read_csv  ('sensor_db/portscanning-03-17-2022-16-22-53.csv', usecols=req_cols)
df17 = pd.read_csv  ('sensor_db/portscanning-03-18-2022-19-27-05.csv', usecols=req_cols)
df18 = pd.read_csv  ('sensor_db/portscanning-03-19-2022-20-01-45.csv', usecols=req_cols)
df19 = pd.read_csv  ('sensor_db/portscanning-03-20-2022-14-27-49.csv', usecols=req_cols) 

frames = [df0, df1, df2, df3, df4, df5, df7, df8, df9, df10, df11, df12, df13, df14, df15, df16, df17, df18, df19]
df = pd.concat(frames,ignore_index=True)
df = df.sample(frac = 1)
print('---------------------------------------------------------------------------------')
print('Normalizing')
print('---------------------------------------------------------------------------------')
df_max_scaled = df.copy()
y = df_max_scaled.pop('ALERT')
df_max_scaled
for col in df_max_scaled.columns:
    t = abs(df_max_scaled[col].max())
    df_max_scaled[col] = df_max_scaled[col]/t
df_max_scaled
df = df_max_scaled.assign(ALERT = y)
print(df)
df = df.fillna(0)
print('---------------------------------------------------------------------------------')
print('---------------------------------------------------------------------------------')
print('---------------------------------------------------------------------------------')
print('---------------------------------------------------------------------------------')
print('Spliting Train and Test')
print('---------------------------------------------------------------------------------')

df['is_train'] = np.random.uniform(0, 1, len(df)) <= .60

train, test = df[df['is_train']==True], df[df['is_train']==False]

features = df.columns[:len(req_cols)-1]

y_train, label = pd.factorize(train['ALERT'])
y_test, label = pd.factorize(test['ALERT'])
#---------------------------------------------------------------------
X_train = train[features]
X_test = test[features]

#----------------------------------------

print('---------------------------------------------------------------------------------')
print('Model training')
print('---------------------------------------------------------------------------------')

start = time.time()

rbf_feature = RBFSampler(gamma=1, random_state=1)
X_features = rbf_feature.fit_transform(X_train)
clf = SGDClassifier(max_iter=20,loss='hinge')
clf.fit(X_features, y_train)

clf.score(X_features, y_train)
end = time.time()


print('---------------------------------------------------------------------------------')
print('ELAPSE TIME MODEL TRAINING: ',(end - start)/60, 'min')
print('---------------------------------------------------------------------------------')

print('---------------------------------------------------------------------------------')
print('Model Prediction')
print('---------------------------------------------------------------------------------')

start = time.time()
X_test_ = rbf_feature.fit_transform(X_test)
rbf_pred = clf.predict(X_test_)
end = time.time()

print('---------------------------------------------------------------------------------')
print('ELAPSE TIME MODEL PREDICTION: ',(end - start)/60, 'min')
print('---------------------------------------------------------------------------------')
ypred = rbf_pred
pred_label = label[ypred]
print('---------------------------------------------------------------------------------')
print('Confusion Matrix')
print('---------------------------------------------------------------------------------')

confusion_matrix = pd.crosstab(test['ALERT'], pred_label, rownames=['Actual ALERT'], colnames = ['Predicted ALERT'])
print(confusion_matrix)

FP = confusion_matrix.sum(axis=0) - np.diag(confusion_matrix)
FN = confusion_matrix.sum(axis=1) - np.diag(confusion_matrix)
TP = np.diag(confusion_matrix)
TN = confusion_matrix.values.sum() - (FP + FN + TP)
TP_total = sum(TP)
TN_total = sum(TN)
FP_total = sum(FP)
FN_total = sum(FN)

TP_total = np.array(TP_total,dtype=np.float64)
TN_total = np.array(TN_total,dtype=np.float64)
FP_total = np.array(FP_total,dtype=np.float64)
FN_total = np.array(FN_total,dtype=np.float64)
#---------------------------------------------------------------
#----------------------------------------------------------------
print('---------------------------------------------------------------------------------')
print('Metrics')
print('---------------------------------------------------------------------------------')
Acc = ACC(TP_total,TN_total, FP_total, FN_total)
print('Accuracy total: ', Acc)
Precision = PRECISION(TP_total, FP_total)
print('Precision total: ', Precision )
Recall = RECALL(TP_total, FN_total)
print('Recall total: ', Recall )
F_1 = F1(Recall,Precision)
print('F1 total: ', F_1 )
BACC = BACC(TP_total,TN_total, FP_total, FN_total)
print('BACC total: ', BACC)
MCC = MCC(TP_total,TN_total, FP_total, FN_total)
print('MCC total: ', MCC)
#----------------------------------------------------------------
model = CalibratedClassifierCV(clf)
model.fit(X_train, y_train)
ypred = model.predict_proba(X_test)
#----------------------------------------------------------------
classes_n = []
for i in range(len(label)): classes_n.append(i)
y_test_bin = label_binarize(y_test,classes = classes_n)
n_classes = y_test_bin.shape[1]
print('rocauc is ',roc_auc_score(y_test_bin,ypred, multi_class='ovr'))
for i in range(0,len(TP)):
    Acc = ACC(TP[i],TN[i], FP[i], FN[i])
    print('Accuracy: ', label[i] ,' - ' , Acc)
test.pop('ALERT')
test.pop('is_train')
start_index = 0
end_index = 500
explainer = shap.KernelExplainer(model.predict_proba, test[start_index:end_index])
shap_values = explainer.shap_values(test[start_index:end_index])

shap.summary_plot(shap_values = shap_values,
                  features = test[start_index:end_index],
                 class_names=[label[0],label[1],label[2]],show=False)
plt.savefig('SVM_Shap_Summary_global.png')
plt.clf()
shap.summary_plot(shap_values = shap_values[0],
                 features = test[start_index:end_index],
                  class_names=[label[0],label[1],label[2]],show=False)
plt.savefig('SVM_Shap_Summary_Beeswarms.png')
plt.clf()
#----------------------------------------------------------------

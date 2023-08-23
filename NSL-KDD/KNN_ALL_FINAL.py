#!/usr/bin/env python
# coding: utf-8

# In[1]:


# importing required libraries
import numpy as np
import pandas as pd
import pickle # saving and loading trained model
from os import path


# importing required libraries for normalizing data
from sklearn import preprocessing
from sklearn.preprocessing import (StandardScaler, OrdinalEncoder,LabelEncoder, MinMaxScaler, OneHotEncoder)
from sklearn.preprocessing import Normalizer, MaxAbsScaler , RobustScaler, PowerTransformer

# importing library for plotting
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import metrics
from sklearn.metrics import accuracy_score # for calculating accuracy of model
from sklearn.model_selection import train_test_split # for splitting the dataset for training and testing
from sklearn.metrics import classification_report # for generating a classification report of model

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc

import tensorflow as tf
from tensorflow.keras.utils import to_categorical

from keras.layers import Dense # importing dense layer

from keras.layers import Input
from keras.models import Model
# representation of model layers
from keras.utils import plot_model
from sklearn.metrics import confusion_matrix
import shap




# In[2]:


#Defining metric functions
def ACC(TP,TN,FP,FN):
    Acc = (TP+TN)/(TP+FP+FN+TN)
    return Acc
def ACC_2 (TP, FN):
    ac = (TP/(TP+FN))
    return ac
def PRECISION(TP,FP):
    eps = 1e-7
    Precision = TP/(TP+FP+eps)
    

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
    eps = 1e-7
    MCC = (TN*TP-FN*FP)/(((TP+FP+eps)*(TP+FN+eps)*(TN+FP+eps)*(TN+FN+eps))**.5)
    return MCC
def AUC_ROC(y_test_bin,y_score):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    auc_avg = 0
    counting = 0
    for i in range(n_classes):
      fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
     # plt.plot(fpr[i], tpr[i], color='darkorange', lw=2)
      #print('AUC for Class {}: {}'.format(i+1, auc(fpr[i], tpr[i])))
      auc_avg += auc(fpr[i], tpr[i])
      counting = i+1
    return auc_avg/counting


# In[3]:


# attach the column names to the dataset
feature=["duration","protocol_type","service","flag","src_bytes","dst_bytes","land","wrong_fragment","urgent","hot",
          "num_failed_logins","logged_in","num_compromised","root_shell","su_attempted","num_root","num_file_creations","num_shells",
          "num_access_files","num_outbound_cmds","is_host_login","is_guest_login","count","srv_count","serror_rate","srv_serror_rate",
          "rerror_rate","srv_rerror_rate","same_srv_rate","diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count", 
          "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate","dst_host_srv_diff_host_rate","dst_host_serror_rate",
          "dst_host_srv_serror_rate","dst_host_rerror_rate","dst_host_srv_rerror_rate","label","difficulty"]
# KDDTrain+_2.csv & KDDTest+_2.csv are the datafiles without the last column about the difficulty score
# these have already been removed.

train='KDDTrain+.txt'
test='KDDTest+.txt'

df=pd.read_csv(train,names=feature)
df_test=pd.read_csv(test,names=feature)

# shape, this gives the dimensions of the dataset
print('Dimensions of the Training set:',df.shape)
print('Dimensions of the Test set:',df_test.shape)


# In[4]:


df.drop(['difficulty'],axis=1,inplace=True)
df_test.drop(['difficulty'],axis=1,inplace=True)


# In[5]:


print('Label distribution Training set:')
print(df['label'].value_counts())
print()
print('Label distribution Test set:')
print(df_test['label'].value_counts())


# In[6]:


# colums that are categorical and not binary yet: protocol_type (column 2), service (column 3), flag (column 4).
# explore categorical features
print('Training set:')
for col_name in df.columns:
    if df[col_name].dtypes == 'object' :
        unique_cat = len(df[col_name].unique())
        print("Feature '{col_name}' has {unique_cat} categories".format(col_name=col_name, unique_cat=unique_cat))

#see how distributed the feature service is, it is evenly distributed and therefore we need to make dummies for all.
print()
print('Distribution of categories in service:')
print(df['service'].value_counts().sort_values(ascending=False).head())


# In[7]:


# Test set
print('Test set:')
for col_name in df_test.columns:
    if df_test[col_name].dtypes == 'object' :
        unique_cat = len(df_test[col_name].unique())
        print("Feature '{col_name}' has {unique_cat} categories".format(col_name=col_name, unique_cat=unique_cat))


# In[8]:


from sklearn.preprocessing import LabelEncoder,OneHotEncoder
categorical_columns=['protocol_type', 'service', 'flag']
# insert code to get a list of categorical columns into a variable, categorical_columns
categorical_columns=['protocol_type', 'service', 'flag'] 
 # Get the categorical values into a 2D numpy array
df_categorical_values = df[categorical_columns]
testdf_categorical_values = df_test[categorical_columns]
df_categorical_values.head()


# In[9]:


# protocol type
unique_protocol=sorted(df.protocol_type.unique())
string1 = 'Protocol_type_'
unique_protocol2=[string1 + x for x in unique_protocol]
# service
unique_service=sorted(df.service.unique())
string2 = 'service_'
unique_service2=[string2 + x for x in unique_service]
# flag
unique_flag=sorted(df.flag.unique())
string3 = 'flag_'
unique_flag2=[string3 + x for x in unique_flag]
# put together
dumcols=unique_protocol2 + unique_service2 + unique_flag2
print(dumcols)

#do same for test set
unique_service_test=sorted(df_test.service.unique())
unique_service2_test=[string2 + x for x in unique_service_test]
testdumcols=unique_protocol2 + unique_service2_test + unique_flag2


# In[10]:


df_categorical_values_enc=df_categorical_values.apply(LabelEncoder().fit_transform)
print(df_categorical_values_enc.head())
# test set
testdf_categorical_values_enc=testdf_categorical_values.apply(LabelEncoder().fit_transform)


# In[11]:


enc = OneHotEncoder()
df_categorical_values_encenc = enc.fit_transform(df_categorical_values_enc)
df_cat_data = pd.DataFrame(df_categorical_values_encenc.toarray(),columns=dumcols)
# test set
testdf_categorical_values_encenc = enc.fit_transform(testdf_categorical_values_enc)
testdf_cat_data = pd.DataFrame(testdf_categorical_values_encenc.toarray(),columns=testdumcols)

df_cat_data.head()


# In[12]:


trainservice=df['service'].tolist()
testservice= df_test['service'].tolist()
difference=list(set(trainservice) - set(testservice))
string = 'service_'
difference=[string + x for x in difference]
difference


# In[13]:


for col in difference:
    testdf_cat_data[col] = 0

testdf_cat_data.shape


# In[14]:


newdf=df.join(df_cat_data)
newdf.drop('flag', axis=1, inplace=True)
newdf.drop('protocol_type', axis=1, inplace=True)
newdf.drop('service', axis=1, inplace=True)
# test data
newdf_test=df_test.join(testdf_cat_data)
newdf_test.drop('flag', axis=1, inplace=True)
newdf_test.drop('protocol_type', axis=1, inplace=True)
newdf_test.drop('service', axis=1, inplace=True)
print(newdf.shape)
print(newdf_test.shape)


# In[15]:


# take label column
labeldf=newdf['label']
labeldf_test=newdf_test['label']
# change the label column
newlabeldf=labeldf.replace({ 'normal' : 0, 'neptune' : 1 ,'back': 1, 'land': 1, 'pod': 1, 'smurf': 1, 'teardrop': 1,'mailbomb': 1, 'apache2': 1, 'processtable': 1, 'udpstorm': 1, 'worm': 1,
                           'ipsweep' : 2,'nmap' : 2,'portsweep' : 2,'satan' : 2,'mscan' : 2,'saint' : 2
                           ,'ftp_write': 3,'guess_passwd': 3,'imap': 3,'multihop': 3,'phf': 3,'spy': 3,'warezclient': 3,'warezmaster': 3,'sendmail': 3,'named': 3,'snmpgetattack': 3,'snmpguess': 3,'xlock': 3,'xsnoop': 3,'httptunnel': 3,
                           'buffer_overflow': 4,'loadmodule': 4,'perl': 4,'rootkit': 4,'ps': 4,'sqlattack': 4,'xterm': 4})
newlabeldf_test=labeldf_test.replace({ 'normal' : 0, 'neptune' : 1 ,'back': 1, 'land': 1, 'pod': 1, 'smurf': 1, 'teardrop': 1,'mailbomb': 1, 'apache2': 1, 'processtable': 1, 'udpstorm': 1, 'worm': 1,
                           'ipsweep' : 2,'nmap' : 2,'portsweep' : 2,'satan' : 2,'mscan' : 2,'saint' : 2
                           ,'ftp_write': 3,'guess_passwd': 3,'imap': 3,'multihop': 3,'phf': 3,'spy': 3,'warezclient': 3,'warezmaster': 3,'sendmail': 3,'named': 3,'snmpgetattack': 3,'snmpguess': 3,'xlock': 3,'xsnoop': 3,'httptunnel': 3,
                           'buffer_overflow': 4,'loadmodule': 4,'perl': 4,'rootkit': 4,'ps': 4,'sqlattack': 4,'xterm': 4})
# put the new label column back
newdf['label'] = newlabeldf
newdf_test['label'] = newlabeldf_test
print(newdf['label'].head())


# In[21]:


# Specify your selected features. Note that you'll need to modify this list according to your final processed dataframe
#Uncomment the below lines to use these top 20 features from shap analysis
#selected_features = ["root_shell","service_telnet","num_shells","service_uucp","dst_host_same_src_port_rate"
#                     ,"dst_host_rerror_rate","dst_host_srv_serror_rate","dst_host_srv_count","service_private","logged_in",
#                    "dst_host_serror_rate","serror_rate","srv_serror_rate","flag_S0","diff_srv_rate","dst_host_srv_diff_host_rate","num_file_creations","flag_RSTR"#,"dst_host_same_srv_rate","service_Idap","label"]
                     

# Select those features from your dataframe
#newdf = newdf[selected_features]
#newdf_test = newdf_test[selected_features]

# Now your dataframe only contains your selected features.

# In[22]:


# creating a dataframe with multi-class labels (Dos,Probe,R2L,U2R,normal)
multi_data = newdf.copy()
multi_label = pd.DataFrame(multi_data.label)

multi_data_test=newdf_test.copy()
multi_label_test = pd.DataFrame(multi_data_test.label)


# In[23]:


# using standard scaler for normalizing
std_scaler = StandardScaler()
def standardization(df,col):
    for i in col:
        arr = df[i]
        arr = np.array(arr)
        df[i] = std_scaler.fit_transform(arr.reshape(len(arr),1))
    return df

numeric_col = multi_data.select_dtypes(include='number').columns
data = standardization(multi_data,numeric_col)
numeric_col_test = multi_data_test.select_dtypes(include='number').columns
data_test = standardization(multi_data_test,numeric_col_test)


# In[24]:


# label encoding (0,1,2,3,4) multi-class labels (Dos,normal,Probe,R2L,U2R)
le2 = preprocessing.LabelEncoder()
le2_test = preprocessing.LabelEncoder()
enc_label = multi_label.apply(le2.fit_transform)
enc_label_test = multi_label_test.apply(le2_test.fit_transform)
multi_data = multi_data.copy()
multi_data_test = multi_data_test.copy()

multi_data['intrusion'] = enc_label
multi_data_test['intrusion'] = enc_label_test

#y_mul = multi_data['intrusion']
multi_data
multi_data_test


# In[25]:


multi_data.drop(labels= [ 'label'], axis=1, inplace=True)
multi_data
multi_data_test.drop(labels= [ 'label'], axis=1, inplace=True)
multi_data_test


# In[26]:


y_train_multi= multi_data[['intrusion']]
X_train_multi= multi_data.drop(labels=['intrusion'], axis=1)

print('X_train has shape:',X_train_multi.shape,'\ny_train has shape:',y_train_multi.shape)

y_test_multi= multi_data_test[['intrusion']]
X_test_multi= multi_data_test.drop(labels=['intrusion'], axis=1)

print('X_test has shape:',X_test_multi.shape,'\ny_test has shape:',y_test_multi.shape)


# In[27]:


from collections import Counter

label_counts = Counter(y_train_multi['intrusion'])
print(label_counts)


# In[28]:


from sklearn.preprocessing import LabelBinarizer

y_train_multi = LabelBinarizer().fit_transform(y_train_multi)
y_train_multi

y_test_multi = LabelBinarizer().fit_transform(y_test_multi)
y_test_multi


# In[29]:


y_train_multi


# In[30]:


Y_train=y_train_multi.copy()
X_train=X_train_multi.copy()

Y_test=y_test_multi.copy()
X_test=X_test_multi.copy()


# In[31]:


Y_train


# In[24]:


from sklearn.feature_selection import SelectKBest, f_classif

# Number of best features you want to select
k = 15

# Initialize a dataframe to store the scores for each feature against each class
feature_scores = pd.DataFrame(index=X_train.columns)

# Loop through each class
for class_index in range(Y_train.shape[1]):
    
    # Get the current class labels
    y_train_current_class = Y_train[:, class_index]
    
    # Select K best features for the current class
    best_features = SelectKBest(score_func=f_classif, k='all')
    fit = best_features.fit(X_train, y_train_current_class)

    # Get the scores
    df_scores = pd.DataFrame(fit.scores_, index=X_train.columns, columns=[f"class_{class_index}"])
    
    # Concatenate the scores to the main dataframe
    feature_scores = pd.concat([feature_scores, df_scores],axis=1)

# Get the sum of the scores for each feature
feature_scores['total'] = feature_scores.sum(axis=1)

# Get the top k features in a list
top_k_features = feature_scores.nlargest(k, 'total').index.tolist()

print(top_k_features)


# In[32]:


from imblearn.over_sampling import RandomOverSampler
from sklearn.datasets import make_classification

# Assuming you have features X and labels Y
# X, Y = make_classification()

ros = RandomOverSampler(sampling_strategy='minority', random_state=100)

X_train, Y_train = ros.fit_resample(X_train, Y_train)


# In[33]:


print(Y_test)


# In[34]:


X_train.values


# In[35]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.multioutput import MultiOutputClassifier
import time
# create KNeighborsClassifier instance
knn = KNeighborsClassifier(n_neighbors=3)  # set the number of neighbors to 3

# Assume 'X_train' is your training data and 'X_test' your test data

# Get feature names from the training set
feature_names = X_train.columns.tolist()

# Reorder the test set to match the training set
X_test = X_test[feature_names]

# Wrap KNeighborsClassifier with MultiOutputClassifier
multi_target_knn = MultiOutputClassifier(knn)
start=time.time()
# Training the model
multi_target_knn.fit(X_train.values, Y_train)
end=time.time()
time_taken = end - start
print(f'Time taken for training: {time_taken} seconds')
# Now you can predict the test set results
start=time.time()
y_pred = multi_target_knn.predict(X_test.values)
end=time.time()
time_taken = end - start
print(f'Time taken for training: {time_taken} seconds')


# In[50]:


# Convert Y_test back to its original format
y_test = np.argmax(Y_test, axis=1)


# In[51]:


pred_labels = np.argmax(y_pred, axis=1)


# In[35]:


correctly_classified_indices = np.where(pred_labels == y_test)[0]
misclassified_indices = np.where(pred_labels != y_test)[0]


# In[36]:


misclassified_indices


# In[37]:


Y_train = Y_train.flatten()


# In[38]:


import lime
import lime.lime_tabular


# In[37]:


class_names = ['Normal', 'DoS', 'Probe', 'R2L', 'U2R']


# In[38]:


X_test = X_test[X_train.columns]


# In[39]:


print(set(X_train.columns) == set(X_test.columns))


# In[49]:


correct_instance_df = X_test.iloc[[correctly_classified_indices[1]]]
misclassified_instance_df = X_test.iloc[[misclassified_indices[0]]]


# In[40]:


# Create a Lime explainer object
explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=X_train.values, 
    feature_names=X_train.columns, 
    class_names=class_names, 
    mode='classification'
)


# In[ ]:





# In[50]:


correct_exp = explainer.explain_instance(
    correct_instance_df.values[0], 
    multi_target_knn.predict,  # use the new function here
    num_features=7, 
    top_labels=1
)

misclassified_exp = explainer.explain_instance(
    misclassified_instance_df.values[0], 
    multi_target_knn.predict,  # and here
    num_features=7, 
    top_labels=1
)


# In[59]:


#correct_exp.show_in_notebook(show_table=True)
print(y_test[misclassified_indices[0]], pred_labels[misclassified_indices[0]])



# In[57]:


misclassified_exp.as_list(label=0)#4


# In[ ]:





# In[36]:


#pred_labels = np.argmax(y_pred, axis=1)
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
import numpy as np

y_pred=np.argmax(y_pred, axis=1)
y_true_multiclass = np.argmax(Y_test, axis=1)
confusion = confusion_matrix(y_true_multiclass, y_pred)

# Binarize the output for AUC
lb = LabelBinarizer()
lb.fit(y_true_multiclass)
y_test_bin = lb.transform(y_true_multiclass)
y_pred_bin = lb.transform(y_pred)

# Iterate through each class and calculate the metrics
class_names = ['Normal','DoS', 'Probe', 'R2L', 'U2R']
for i in range(len(class_names)):
    TP = confusion[i, i]
    FP = confusion[:, i].sum() - TP
    FN = confusion[i, :].sum() - TP
    TN = confusion.sum() - TP - FP - FN
    
    # Call your metrics functions
    Acc = ACC(TP, TN, FP, FN)
    Precision = PRECISION(TP, FP)
    Recall = RECALL(TP, FN)
    F1_score = F1(Recall, Precision)
    Balanced_accuracy = BACC(TP, TN, FP, FN)
    Matthews = MCC(TP, TN, FP, FN)
    
    # AUC_ROC calculation
    AUC_ROC = roc_auc_score(y_test_bin[:, i], y_pred_bin[:, i])
    
    # Print metrics
    print(f'Metrics for: {class_names[i]}')
    print('Accuracy: ', Acc)
    print('Precision: ', Precision)
    print('Recall: ', Recall)
    print('F1: ', F1_score)
    print('BACC: ', Balanced_accuracy)
    print('MCC: ', Matthews)
    print('AUC_ROC: ', AUC_ROC)
    print()

# AUC_ROC total
print('AUC_ROC total: ', roc_auc_score(y_test_bin, y_pred_bin, multi_class='ovr'))
print('---------------------------------------------------------------------------------')



# In[28]:


# Use KernelExplainer for model agnostic
explainer = shap.KernelExplainer(multi_target_knn.predict, shap.sample(X_train,500))

# Calculate Shap values on a small sample of test data
small_X_test = X_test[:500]
shap_values = explainer.shap_values(small_X_test)

# Plot SHAP summary
shap.summary_plot(shap_values, small_X_test, feature_names=multi_data.columns,class_names = ['Normal', 'DoS', 'Probe', 'R2L', 'U2R'])


# In[ ]:


fig = plt.gcf()

# Save the figure
fig.savefig('shap_summary_plot.png')


# In[32]:


import matplotlib.pyplot as plt

# Plot SHAP summary
shap.summary_plot(shap_values, small_X_test, feature_names=multi_data.columns, class_names=['Normal', 'DoS', 'Probe', 'R2L', 'U2R'])

# Get current figure
fig = plt.gcf()

# Save the figure
fig.savefig('shap_summary_plot_KNN.png')


# In[31]:


import matplotlib.pyplot as plt
import matplotlib.pylab as pl

# Create a new matplotlib Figure and Axes
fig, ax = pl.subplots(1,1)

# Plot SHAP summary on the created Axes
shap.summary_plot(shap_values, small_X_test, feature_names=multi_data.columns, plot_type="bar", show=False)

# Save the figure
plt.savefig('shap_summary_plot.png', bbox_inches='tight')
plt.close(fig)


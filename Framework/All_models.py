

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

from keras.layers import Dense # importing dense layer

from keras.layers import Input
from keras.models import Model
# representation of model layers
from keras.utils import plot_model
from sklearn.metrics import confusion_matrix
import shap
from sklearn.preprocessing import LabelBinarizer

# In[35]:
import pickle

# Load the preprocessed datasets
with open('X_train.pkl', 'rb') as file:
    X_train = pickle.load(file)

with open('X_test.pkl', 'rb') as file:
    X_test = pickle.load(file)

with open('Y_train.pkl', 'rb') as file:
    Y_train = pickle.load(file)

with open('Y_test.pkl', 'rb') as file:
    Y_test = pickle.load(file)

# Now X_train, X_test, Y_train, and Y_test are available for use


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


# In[44]:


print(Y_test)

# In[45]:
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
# Interactive model selection
print("Select the model to train and predict:")
print("1. AdaBoost Classifier")
print("2. K-Nearest Neighbors (KNN)")
print("3. Multi-Layer Perceptron (MLP)")
print("4. Random Forest Classifier")
print("5. Deep Neural Network (DNN)")
print("6. LightGBM")
print("7. Support Vector Machine (SVM)")

model_choice = input("Enter your choice (1-7): ")

if model_choice == '1':
    from sklearn.multioutput import MultiOutputClassifier
    from sklearn.ensemble import AdaBoostClassifier
    import time

    abc = AdaBoostClassifier(n_estimators=1, learning_rate=1.0)

    # Assume 'X_train' is your training data and 'X_test' your test data

    # Get feature names from the training set
    feature_names = X_train.columns.tolist()

    # Reorder the test set to match the training set
    X_test = X_test[feature_names]


    # Wrap AdaBoostClassifier with MultiOutputClassifier
    multi_target_abc = MultiOutputClassifier(abc)
    start=time.time()
    # Training the model
    multi_target_abc.fit(X_train.values, Y_train)
    end=time.time()
    time_taken = end - start
    print(f'Time taken for training: {time_taken} seconds')
    start=time.time()

    # Now you can predict the test set results
    y_pred = multi_target_abc.predict(X_test)
    end=time.time()
    time_taken = end - start
    print(f'Time taken for training: {time_taken} seconds')


elif model_choice == '2':
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

elif model_choice == '3':
    from sklearn.neural_network import MLPClassifier
    from sklearn.multioutput import MultiOutputClassifier
    import time

    # create MLPClassifier instance
    mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=200, random_state=1)

    # Assume 'X_train' is your training data and 'X_test' your test data

    # Get feature names from the training set
    feature_names = X_train.columns.tolist()

    # Reorder the test set to match the training set
    X_test = X_test[feature_names]

    # Wrap MLPClassifier with MultiOutputClassifier
    multi_target_mlp = MultiOutputClassifier(mlp)
    start=time.time()
    # Training the model
    multi_target_mlp.fit(X_train.values, Y_train)
    end=time.time()
    time_taken = end - start
    print(f'Time taken for training: {time_taken} seconds')
    start=time.time()
    # Now you can predict the test set results2
    
    y_pred = multi_target_mlp.predict(X_test)
    end=time.time()
    time_taken = end - start
    print(f'Time taken for pred: {time_taken} seconds')

elif model_choice == '4':
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.multioutput import MultiOutputClassifier
    import time
    import numpy as np

    # Assuming 'X_train', 'Y_train', 'X_test', and 'Y_test' are already defined

    # Create RandomForestClassifier instance with specified parameters
    rf = RandomForestClassifier(
    n_estimators=200, 
    max_depth=15, 
    min_samples_split=10, 
    min_samples_leaf=5, 
    class_weight='balanced_subsample', 
    random_state=0,
    n_jobs=-1  # Utilize all processors for training
    )

    # Get feature names from the training set
    feature_names = X_train.columns.tolist()

    # Reorder the test set to match the training set
    X_test = X_test[feature_names]

    # Wrap RandomForestClassifier with MultiOutputClassifier
    multi_target_rf = MultiOutputClassifier(rf)

    # Training the model
    start = time.time()
    multi_target_rf.fit(X_train.values, Y_train)
    end = time.time()
    print(f'Time taken for training: {end - start} seconds')

    # Predict the test set results
    start = time.time()
    preds = multi_target_rf.predict(X_test)
    end = time.time()
    print(f'Time taken for predictions: {end - start} seconds')

elif model_choice == '5':
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense
    import numpy as np
    import time

    model = Sequential()
    model.add(Dense(128, input_dim=X_train.shape[1], activation='relu'))  # Input layer
    model.add(Dense(64, activation='relu'))  # Hidden layer
    model.add(Dense(5, activation='softmax'))  # Output layer

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # summary of model layers
    model.summary()
    # Convert Y_test back to its original format
    y_test = np.argmax(Y_test, axis=1)

    # Start the timer
    start = time.time()

    model.fit(X_train, Y_train, epochs=50, batch_size=32, verbose=0)

    # End the timer
    end = time.time()

    # Calculate the time taken and print it out
    time_taken = end - start
    print(f'Time taken for training: {time_taken} seconds')

    # Predict classes
    preds = model.predict(X_test)


    # In[27]:


    start = time.time()

    preds = model.predict(X_test)


    # End the timer
    end = time.time()
    time_taken = end - start
    print(f'Time taken for pred: {time_taken} seconds')


elif model_choice == '6':
    from lightgbm import LGBMClassifier
    from sklearn.multioutput import MultiOutputClassifier

    # Create LGBMClassifier instance with default parameters
    lgbm = LGBMClassifier(random_state=0)

    # Wrap LGBMClassifier with MultiOutputClassifier
    multi_target_lgbm = MultiOutputClassifier(lgbm)

    # Training the model
    multi_target_lgbm.fit(X_train, Y_train)

    # Now you can predict the test set results
    y_pred = multi_target_lgbm.predict(X_test)


    # In[26]:


    # Convert Y_test back to its original format
    y_test = np.argmax(Y_test, axis=1)


    # In[27]:


    pred_labels = np.argmax(y_pred, axis=1)


elif model_choice == '7':
    from sklearn.multioutput import MultiOutputClassifier
    from sklearn.linear_model import SGDClassifier

    # Instantiate the SGDClassifier
    clf = SGDClassifier(loss='hinge')  # hinge loss gives a linear SVM

    # Wrap SGDClassifier with MultiOutputClassifier
    multi_target_clf = MultiOutputClassifier(clf)

    # Fit the model
    multi_target_clf.fit(X_train, Y_train)

    # Make predictions
    y_pred = multi_target_clf.predict(X_test)


    # In[37]:


    # Convert Y_test back to its original format
    y_test = np.argmax(Y_test, axis=1)


    # In[57]:


    pred_labels = np.argmax(y_pred, axis=1)


else:
    print("Invalid choice.")










from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import LabelBinarizer

# Convert predictions and true labels to label format if they are one-hot encoded
y_pred_labels = np.argmax(y_pred, axis=1)
y_test_labels = np.argmax(Y_test, axis=1)

# Print classification report
print("Classification Report:")
print(classification_report(y_test_labels, y_pred_labels, target_names=['Normal', 'DoS', 'Probe', 'R2L', 'U2R']))

# Confusion Matrix
confusion = confusion_matrix(y_test_labels, y_pred_labels)
print("Confusion Matrix:")
print(confusion)

# Binarize the output for AUC calculation if necessary
lb = LabelBinarizer()
lb.fit(y_test_labels)
y_test_bin = lb.transform(y_test_labels)
y_pred_bin = lb.transform(y_pred_labels)

# Calculate AUC-ROC
if Y_test.shape[1] == 2:  # Binary classification
    roc_auc = roc_auc_score(y_test_bin, y_pred_bin)
elif Y_test.shape[1] > 2:  # Multi-class classification
    roc_auc = roc_auc_score(y_test_bin, y_pred_bin, multi_class='ovr', average='macro')

print(f"AUC-ROC: {roc_auc}")

# Additional Metrics if needed
# Iterate through each class and calculate additional metrics
class_names = ['Normal', 'DoS', 'Probe', 'R2L', 'U2R']
for i, class_name in enumerate(class_names):
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

    # Print metrics
    print(f'Metrics for {class_name}:')
    print(f'  Accuracy: {Acc}')
    print(f'  Precision: {Precision}')
    print(f'  Recall: {Recall}')
    print(f'  F1 Score: {F1_score}')
    print(f'  Balanced Accuracy: {Balanced_accuracy}')
    print(f'  MCC: {Matthews}')
    print()

print('---------------------------------------------------------------------------------')


# In[68]:
'''

correctly_classified_indices = np.where(pred_labels == y_test)[0]
misclassified_indices = np.where(pred_labels != y_test)[0]


# In[69]:


misclassified_indices[:5]


# In[70]:


class_names = ['Normal','DoS', 'Probe', 'R2L', 'U2R']


# In[71]:


import lime
import lime.lime_tabular


# In[79]:


exp = explainer.explain_instance(X_test.iloc[0], multi_target_abc.predict, num_features=5, num_samples=10000)


# In[72]:


# Create a Lime explainer object
explainer = lime.lime_tabular.LimeTabularExplainer(
    X_train.values,
    training_labels=Y_train,
    feature_names=X_train.columns.tolist(),
    class_names=class_names, 
    mode='classification'
)


# In[86]:


# Select a correctly classified instance


correct_instance = X_test.iloc[correctly_classified_indices[0]].values
correct_exp = explainer.explain_instance(
correct_instance, 
multi_target_abc.predict, 
    num_samples=10000,
num_features=7, 
top_labels=1
)
mr=np.random.randint(0,misclassified_indices.shape[0])
misclassified_instance = X_test.iloc[misclassified_indices[2]].values


# Explain this instance with LIME
misclassified_exp = explainer.explain_instance(
misclassified_instance, 
multi_target_abc.predict, 
    num_samples=10000,
num_features=7, 
top_labels=1
)
    


# In[90]:


misclassified_exp.as_list(label=4)#4



# Use KernelExplainer for model agnostic
explainer = shap.KernelExplainer(multi_target_abc.predict, shap.sample(X_train, 20))

# Calculate Shap values on a small sample of test data
small_X_test = X_test[:100]
shap_values = explainer.shap_values(small_X_test)

# Plot SHAP summary
shap.summary_plot(shap_values, small_X_test, feature_names=multi_data.columns)


# In[43]:


fig = plt.gcf()

# Save the figure
fig.savefig('shap_summary_plot.png')


# In[47]:


import matplotlib.pyplot as plt

# Plot SHAP summary
shap.summary_plot(shap_values, small_X_test, feature_names=multi_data.columns, class_names=['Normal', 'DoS', 'Probe', 'R2L', 'U2R'])

# Get current figure
fig = plt.gcf()

# Save the figure
fig.savefig('shap_summary_plot_ADA.png')


# In[51]:


import matplotlib.pyplot as plt
import matplotlib.pylab as pl

# Create a new matplotlib Figure and Axes
fig, ax = pl.subplots(1,1)

# Plot SHAP summary
shap.summary_plot(shap_values, small_X_test, feature_names=multi_data.columns, class_names=['Normal', 'DoS', 'Probe', 'R2L', 'U2R'])

# Show the plot
plt.show()

# Save the figure
plt.savefig('shap_summary_plot_ADA.png', bbox_inches='tight')
plt.close(fig)


# In[51]:


print(y_test[misclassified_indices[3]], pred_labels[misclassified_indices[3]])

'''
import sys

# Path to the directory where XAI_Framework is located
framework_dir = '/Users/tanishrohith/Downloads/XAI_NIDS-main_framework'

# Add the directory to sys.path
sys.path.append(framework_dir)

# Now try importing your module
from XAI_Framework.Shap_Lime import shap_barplots, shap_waterfallplot, shap_beeswarmplot, lime_plot
# Use the imported functions


# Now you can use these functions


# Use the imported functions


# Example of using one of the functions
model = multi_target_abc

#shap_barplots(model, X_train, X_test)
row_index = 0  # Adjust the index as needed
shap_waterfallplot(model, X_train, X_test,2, row_index)
#shap_beeswarmplot(model, X_train, X_test)
lime_plot(model, X_train, X_test,Y_train, row_index,class_names)


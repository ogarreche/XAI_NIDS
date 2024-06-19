# XAI-IDS: Towards Proposing an Explainable Artificial Intelligence Framework for Enhancing Network Intrusion Detection Systems

# Abstract

The exponential growth of different intrusions on networked systems inspires new research directions on developing advanced artificial intelligence (AI) techniques for intrusion detection systems (IDS). There are several challenges for such dependence on AI for IDS including the performance of such AI models, and the lack of explainability of the decisions made by such AI algorithms where its outputs are not understandable by the human security analyst. To close such a research gap, we propose an end-to-end explainable AI (XAI) framework for enhancing understandability of AI models for network intrusion detection tasks. We first benchmark eight black-box AI models on two real-world network intrusion datasets with different characteristics. We then generate local and global explanations using different XAI models. We also generate model-specific and intrusion-specific important features. We furthermore generate the common important features that affect different AI models. Our framework has different levels of explanations that can help the network security analysts make more informed decisions based on such explanations. We release our source codes for the community to access it as a baseline XAI framework and to build on it with new datasets and models.

# Performance

Overall performances for AI models with top 15 features for the RoEduNet-SIMARGL2021 dataset.

![image](https://user-images.githubusercontent.com/55901425/228660881-86554614-d70b-49df-a6ea-3a22a118aca0.png)

Overall performances for AI models with top 15 features for the CICIDS-2017 dataset.

![image](https://user-images.githubusercontent.com/55901425/228660928-d2df3862-6cda-49e2-ab76-6e4e3179ffc1.png)

Low-Level XAI Pipeline Components

![image](https://github.com/ogarreche/XAI_NIDS/blob/main/images/framework.png?raw=true)


  - Loading Intrusion Database: Loading either one of the datasets.
  - Feature Extraction: Selecting 15 top features or all the features.
  - Redundancy Elimination and Randomizing Rows: Eliminate duplicate rows and data shuffle. 
  - Data Balancing: Oversample technique is used.
  - Feature Normalization: All features are normalize in a scale from 0 to 1.
  - Black-box AI Models: The model we are using in each program (SVM, DNN, MLP, KNN,LightGBM, XGBoost, ADA, Random Forest).
  - Black-box AI Evaluation: The metrics used: accuracy (ACC), precision (Prec), recall (Rec), F1-score (F1), Matthews correlation coefficient  (MCC), balanced accuracy (BACC), and the area under ROC curve (AUCROC).
  - XAI Global Explanations: Shap generates Global Summary and Beeswarm Plot.
  - XAI Local Explanations: Shap and LIME generate single sample explanations.
  

# How to use the programs:
## For Global Explanations.
  1. Download one of the datasets.
     
    RoEduNet-SIMARGL2021: https://www.kaggle.com/datasets/7f91274fa3074d53e983f6eb7a7b24ad1dca136ca967ad0ebe48955e246c24ee

    CICIDS-2017: [https://www.kaggle.com/datasets/cicdataset/cicids2017](https://www.kaggle.com/datasets/usmanshuaibumusa/cicids-17)
    
  3. Each program is a standalone program that is aimed to run one form of AI model within a set of features. (i.e. DNN_final.py in the CICIDS-2017 folder will run the      DNN model with 15 features for that given dataset. On the other hand. DNN_all_final.py will run the DNN model for all features for the given dataset).
  4. Each program outputs a confusion matrix, metrics scores (i.e. accuracy (ACC), precision (Prec), recall (Rec), F1-score (F1), Matthews correlation coefficient  (MCC), balanced accuracy (BACC), and the area under ROC curve (AUCROC)), and the Global Summary/Beeswarm Plot.

  5. Extra: there is a standalone example program RF_example.ipynb in the RoEduNet-SIMARGL2021 folder.

## For Local Explanations. 
  1. Download one of the datasets.
     
    RoEduNet-SIMARGL2021: https://www.kaggle.com/datasets/7f91274fa3074d53e983f6eb7a7b24ad1dca136ca967ad0ebe48955e246c24ee

    CICIDS-2017: [https://www.kaggle.com/datasets/cicdataset/cicids2017](https://www.kaggle.com/datasets/usmanshuaibumusa/cicids-17)
    
  4. Run the example python notebook called "RF_LIME_SHAP.ipynb" in a python notebook environment.
  5. The program outputs one Local Waterfall shap explanation and one Local LIME explanation for the same sample using the Random Forest method

# Visualization results  

## Global Summary/Beeswarm plots with SHAP. 

Results example for Random Forest using RoEduNet-SIMARGL2021.

![image](https://user-images.githubusercontent.com/55901425/227805146-0a686613-1428-432a-a8b4-e93221eff1b3.png)
![image](https://user-images.githubusercontent.com/55901425/227805161-1ef31f27-74eb-44ff-a29c-9ca3d723dbfb.png)

## Local Explanation with LIME and SHAP.
 
 Results using SHAP and LIME for the same Random Forest prediction for a normal traffic sample from the CICIDS-2017 dataset.

![image](https://user-images.githubusercontent.com/55901425/227805234-0f0f9ac4-9b90-4c31-af63-61de5063ad29.png)
![image](https://user-images.githubusercontent.com/55901425/227805243-069ddaff-d56c-4805-b53e-b771ae1c5d43.png)


## XAI-Framework.

Preprocessing File:

- To begin using our framework, run the preprocessing file. This file is designed to process your raw data, encompassing steps like normalization, encoding, and feature selection. Upon completion, it will output four key datasets: `X_train`, `X_test`, `Y_train`, and `Y_test`. These datasets are crucial for feeding into the subsequent model training and testing phases.

All_Model File:

- For users looking to test our architecture with their data, the `All_Model` file is your next step. This interactive file allows you to choose from seven different machine learning algorithms, depending on your specific needs or experimental setup. The options range from ensemble methods like AdaBoost and Random Forest to neural networks, KNN, MLP, and SGD. Simply run the `All_Model` file, select your desired algorithm, and it will automatically apply it to the datasets generated from your preprocessing step.

Framework Folder: 

- A pivotal component of our project is the `XAI_Framework` folder, focusing on Explainable AI (XAI). This folder contains tools and functions for analyzing and visualizing the decision-making process of the models. It is designed to be modified or extended as per your requirements, offering insights into how and why specific model predictions are made.
 
Modularity and Flexibility: 

- Our project's architecture is intentionally modular and user-friendly. Each component - preprocessing, model training, and XAI - is independent yet seamlessly integrated. This design ensures that small changes in one part do not impact the overall functionality, offering a flexible and adaptable environment for users. Whether you're conducting academic research or applying it in industry, our framework is equipped to cater to a wide range of applications and user scenarios.


## Extra: Xplique toolbox comparison.

We want to cite the following pages for this section.

- IGTD github: https://github.com/zhuyitan/IGTD
- Xplique github: https://github.com/deel-ai/xplique

To generate the results below, please go to the Xplique folder and:

1) Convert the tabular dataset into images using tabconversion_cic.py
2) Run the metrics_cic.ipynb
3) Run the Attributions_Regression_CIC.ipynb

The feature importance using Xplique for CICIDS-2017 dataset for SHAP
(on the left) and LIME (on the right).

![image](https://github.com/ogarreche/XAI_NIDS/blob/main/images/xpliquefeatures.png?raw=true)

Xplique metrics (Deletion, Insetion, MuFidelity, and Stability) and their corresponding results for SHAP and LIME for our three datasets.

![image](https://github.com/ogarreche/XAI_NIDS/blob/main/images/xpliquemetrics.png?raw=true)


### Citation:

Please cite this work if it was useful to you :)

https://www.mdpi.com/2076-3417/14/10/4170


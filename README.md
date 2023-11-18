# XAI-IDS: Towards Proposing an Explainable Artificial Intelligence Framework for Enhancing Network Intrusion Detection Systems

# Abstract

The exponential growth of different intrusions on networked systems inspires new research directions on developing advanced artificial intelligence (AI) techniques for intrusion detection systems (IDS). There are several challenges for such dependence on AI for IDS including the performance of such AI models, and the lack of explainability of the decisions made by such AI algorithms where its outputs are not understandable by the human security analyst. To close such a research gap, we propose an end-to-end explainable AI (XAI) framework for enhancing understandability of AI models for network intrusion detection tasks. We first benchmark eight black-box AI models on two real-world network intrusion datasets with different characteristics. We then generate local and global explanations using different XAI models. We also generate model-specific and intrusion-specific important features. We furthermore generate the common important features that affect different AI models. Our framework has different levels of explanations that can help the network security analysts make more informed decisions based on such explanations. We release our source codes for the community to access it as a baseline XAI framework and to build on it with new datasets and models.

# Performance

Overall performances for AI models with top 15 features for the RoEduNet-SIMARGL2021 dataset.

![image](https://user-images.githubusercontent.com/55901425/228660881-86554614-d70b-49df-a6ea-3a22a118aca0.png)

Overall performances for AI models with top 15 features for the CICIDS-2017 dataset.

![image](https://user-images.githubusercontent.com/55901425/228660928-d2df3862-6cda-49e2-ab76-6e4e3179ffc1.png)

Low-Level XAI Pipeline Components

![image](https://github.com/ogarreche/XAI_NIDS/assets/55901425/429da6df-95c5-4da5-b3de-f79f80825091)



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
    CICIDS-2017: https://www.kaggle.com/datasets/cicdataset/cicids2017
  2. Each program is a standalone program that is aimed to run one form of AI model within a set of features. (i.e. DNN_final.py in the CICIDS-2017 folder will run the      DNN model with 15 features for that given dataset. On the other hand. DNN_all_final.py will run the DNN model for all features for the given dataset).
  3. Each program outputs a confusion matrix, metrics scores (i.e. accuracy (ACC), precision (Prec), recall (Rec), F1-score (F1), Matthews correlation coefficient  (MCC), balanced accuracy (BACC), and the area under ROC curve (AUCROC)), and the Global Summary/Beeswarm Plot.

## For Local Explanations. 
  1. Download one of the datasets.
    RoEduNet-SIMARGL2021: https://www.kaggle.com/datasets/7f91274fa3074d53e983f6eb7a7b24ad1dca136ca967ad0ebe48955e246c24ee
    CICIDS-2017: https://www.kaggle.com/datasets/cicdataset/cicids2017
  2. Run the example python notebook called "RF_LIME_SHAP.ipyn" in a python notebook environment.
  3. The program outputs one Local Waterfall shap explanation and one Local LIME explanation for the same sample using the Random Forest method

# Visualization results  

## Global Summary/Beeswarm plots with SHAP. 

Results example for Random Forest using RoEduNet-SIMARGL2021.

![image](https://user-images.githubusercontent.com/55901425/227805146-0a686613-1428-432a-a8b4-e93221eff1b3.png)
![image](https://user-images.githubusercontent.com/55901425/227805161-1ef31f27-74eb-44ff-a29c-9ca3d723dbfb.png)

## Local Explanation with LIME and SHAP.
 
 Results using SHAP and LIME for the same Random Forest prediction for a normal traffic sample from the CICIDS-2017 dataset.

![image](https://user-images.githubusercontent.com/55901425/227805234-0f0f9ac4-9b90-4c31-af63-61de5063ad29.png)
![image](https://user-images.githubusercontent.com/55901425/227805243-069ddaff-d56c-4805-b53e-b771ae1c5d43.png)

# Update changes after revision

## Xplique toolbox comparison.

We want to cite the following pages for this section.

- IGTD github: https://github.com/zhuyitan/IGTD
- Xplique github: https://github.com/deel-ai/xplique

To generate the results below, please go to the Xplique folder and:

1) Convert the tabular dataset into images using tabconversion_cic.py
2) Run the metrics_cic.ipynb
3) Run the Attributions_Regression_CIC.ipynb

The feature importance using Xplique for CICIDS-2017 dataset for SHAP
(on the left) and LIME (on the right).

![image](https://github.com/ogarreche/XAI_NIDS/assets/55901425/c5b559a6-d11e-4381-8d2d-520eccda3e41)

Xplique metrics (Deletion, Insetion, MuFidelity, and Stability) and their corresponding results for SHAP and LIME for our three datasets.

![image](https://github.com/ogarreche/XAI_NIDS/assets/55901425/7cc7bfd7-5c4f-4a48-b8dd-181585966c76)




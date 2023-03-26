#Towards an Explainable Artificial Intelligence Framework for Enhancing Explainability in Network Intrusion Detection Systems

#Abstract

The exponential growth of different intrusions on networked systems inspires new research directions on developing advanced artificial intelligence (AI) techniques for intrusion detection systems (IDS). There are several challenges for such dependence on AI for IDS including the performance of such AI models, and the lack of explainability of the decisions made by such AI algorithms where its outputs are not understandable by the human security analyst. To close such a research gap, we propose an end-to-end explainable AI (XAI) framework for enhancing understandability of AI models for network intrusion detection tasks. We first benchmark eight black-box AI models on two real-world network intrusion datasets with different characteristics. We then generate local and global explanations using different XAI models. We also generate model-specific and intrusion-specific important features. We furthermore generate the common important features that affect different AI models. Our framework has different levels of explanations that can help the network security analysts make more informed decisions based on such explanations. We release our source codes for the community to access it as a baseline XAI framework and to build on it with new datasets and models.

#Performance

Overall performances for AI models with top 15 features for the RoEduNet-SIMARGL2021 dataset.
![image](https://user-images.githubusercontent.com/55901425/227804282-299f2b99-dca1-4a7e-a13c-2207cfadd4ee.png)

Overall performances for AI models with top 15 features for the CICIDS-2017 dataset.
![image](https://user-images.githubusercontent.com/55901425/227804311-8affc707-16c4-4109-901a-ffb6f69846ad.png)

Low-Level XAI Pipeline Components
![image](https://user-images.githubusercontent.com/55901425/227802937-a66c9dd3-14da-41fc-9319-e98696bd85dc.png)


  Loading Intrusion Database: Loading either one of the datasets.
  Feature Extraction: Selecting 15 top features or all the features.
  Redundancy Elimination and Randomizing Rows: Eliminate duplicate rows and data shuffle. 
  Data Balancing: Oversample technique is used.
  Feature Normalization: All features are normalize in a scale from 0 to 1.
  Black-box AI Models: The model we are using in each program (SVM, DNN, MLP, KNN,LightGBM, XGBoost, ADA, Random Forest).
  Black-box AI Evaluation: The metrics used: accuracy (ACC), precision (Prec), recall (Rec), F1-score (F1), Matthews correlation coefficient  (MCC), balanced accuracy (BACC), and the area under ROC curve (AUCROC).
  XAI Global Explanations: Shap generates Global Summary and Beeswarm Plot.
  XAI Local Explanations: Shap and LIME generate single sample explanations.
  

#How to use the programs:
  1) Download both datasets
    RoEduNet-SIMARGL2021: https://www.kaggle.com/datasets/7f91274fa3074d53e983f6eb7a7b24ad1dca136ca967ad0ebe48955e246c24ee
    CICIDS-2017: https://www.kaggle.com/datasets/cicdataset/cicids2017
  2) Each program is a standalone program that is aimed to run one form of AI model within a set of features. (i.e. DNN_final.py in the CICIDS-2017 folder will run the      DNN model with 15 features for that given dataset. On the other hand. DNN_all_final.py will run the DNN model for all features for the given dataset).
  3) Each program outputs a confusion matrix, metrics scores (i.e. accuracy (ACC), precision (Prec), recall (Rec), F1-score (F1), Matthews correlation coefficient  (MCC), balanced accuracy (BACC), and the area under ROC curve (AUCROC)), and the Global Summary/Beeswarm Plot.

Towards an Explainable Artificial Intelligence Framework for Enhancing Explainability in Network Intrusion Detection Systems

Abstract

The exponential growth of different intrusions on networked systems inspires new research directions on developing advanced artificial intelligence (AI) techniques for intrusion detection systems (IDS). There are several challenges for such dependence on AI for IDS including the performance of such AI models, and the lack of explainability of the decisions made by such AI algorithms where its outputs are not understandable by the human security analyst. To close such a research gap, we propose an end-to-end explainable AI (XAI) framework for enhancing understandability of AI models for network intrusion detection tasks. We first benchmark eight black-box AI models on two real-world network intrusion datasets with different characteristics. We then generate local and global explanations using different XAI models. We also generate model-specific and intrusion-specific important features. We furthermore generate the common important features that affect different AI models. Our framework has different levels of explanations that can help the network security analysts make more informed decisions based on such explanations. We release our source codes for the community to access it as a baseline XAI framework and to build on it with new datasets and models.

![image](https://user-images.githubusercontent.com/55901425/227802937-a66c9dd3-14da-41fc-9319-e98696bd85dc.png)

Low-Level XAI Pipeline Components
  Loading Intrusion Database:
  Feature Extraction: 
  Redundancy Elimination and Randomizing Rows:
  Data Balancing: 
  Feature Normalization:
  Black-box AI Models:
  Black-box AI Evaluation:
  XAI Global Explanations: 
  XAI Local Explanations:
  Rationale of Choosing SHAP and LIME:
  Feature Explanation:
  
RoEduNet-SIMARGL2021: https://www.kaggle.com/datasets/7f91274fa3074d53e983f6eb7a7b24ad1dca136ca967ad0ebe48955e246c24ee
CICIDS-2017: https://www.kaggle.com/datasets/cicdataset/cicids2017

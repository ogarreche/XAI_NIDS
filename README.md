Towards an Explainable Artificial Intelligence Framework for Enhancing Explainability in Network Intrusion Detection Systems

Abstract

The exponential growth of different intrusions on networked systems inspires new research directions on developing advanced artificial intelligence (AI) techniques for intrusion detection systems (IDS). There are several challenges for such dependence on AI for IDS including the performance of such AI models, and the lack of explainability of the decisions made by such AI algorithms where its outputs are not understandable by the human security analyst. To close such a research gap, we propose an end-to-end explainable AI (XAI) framework for enhancing understandability of AI models for network intrusion detection tasks. We first benchmark eight black-box AI models on two real-world network intrusion datasets with different characteristics. We then generate local and global explanations using different XAI models. We also generate model-specific and intrusion-specific important features. We furthermore generate the common important features that affect different AI models. Our framework has different levels of explanations that can help the network security analysts make more informed decisions based on such explanations. We release our source codes for the community to access it as a baseline XAI framework and to build on it with new datasets and models.

![image](https://user-images.githubusercontent.com/55901425/227800472-c28c8b33-b46a-49ee-96e3-936495970582.png)


![image](https://user-images.githubusercontent.com/55901425/227800487-e8ad014d-4455-49b2-aacb-3b6f8c9006a8.png)

We now explain the low-level components of our XAI pipeline. The
different components (shown in Figure 2) are explained below.
Loading Intrusion Database: The first component in our
pipeline is loading the data from the database as a starting point.
In our work, we leverage the two popular network intrusion
detection datasets which are the RoEduNet-SIMARGL2021 [46 ]
and the CICIDS-2017 [54] datasets.
Feature Extraction: Having loaded the database, the second
component in our framework is to select important features from
such a database. Such feature extraction from the log files stored
in the database is essential for building AI models for intrusion
detection task. The process of feature extraction usually impacts
the outcome of the AI model with the existence of features that are
more representative of the traffic nature than others. We followed
the prior works [ 46, 54 ] for extracting the basic set of features for
each network intrusion dataset.2
Redundancy Elimination and Randomizing Rows: After
loading the database and doing feature selection, the next step is
to eliminate redundancy in the traffic entries. Such redundancy
is a normal characteristic of network traffic [ 38]. Thus, removing
such redundancy is essential for avoiding underperformance of the
intrusion detection AI model. We do that by simply deleting all
rows that are exactly the same. Then, we perform randomization
of the rows as a preprocessing step before training since the data
was organized sequentially based on attack labels.
Data Balancing: One other issue we address in our framework
is data balancing. As shown in Table 3, both of our datasets are
unbalanced where the normal traffic is much larger compared to
the attack traffic. We emphasize that such an issue of unbalanced
network traffic types is a common problem for building robust AI
models in network security [ 8]. Therefore, we balance the dataset
using the random oversampling technique [49 ], applying it to all
Datasets used
labels, thus balancing all the network traffic classes. This simple
method has been proved to be efficient to balance huge datasets [49 ]
which is the case in our datasets because of its low complexity.
Feature Normalization: To avoid having different scales
for different traffic features, we perform the typical feature
normalization step to have all of our features within the same
numerical scale. We discuss normalization’s effect in Appendix C.
Black-box AI Models: Once the preprocessing of the database
is complete, we train the AI model where we perform splitting
of the data with a split of 70% for the training while leaving the
unseen 30% for testing purposes. We emphasize that for this part, we
have built eight state-of-the-art AI classification models including
random forest (RF), deep neural network (DNN), Adaptive Boosting
(AdaBoost), multi-layer perceptron (MLP), k-nearest neighbors
(KNN), support vector machine (SVM), LightGBM, and XGBoost.
For training each model, we set its specific parameters to ensure the
best possible performance (see details of our setups in Section 4).
Black-box AI Evaluation: After training the eight AI models,
we test each of these models on the unseen data. Thus, the next step
in our framework is to analyze the performance of each model on
such unseen test data. For such performance analysis, we first create
the confusion matrix for each model and from its values we derive
the following different metrics for each AI model: accuracy (ACC),
precision (Prec), recall (Rec), F1-score (F1), Matthews correlation
coefficient (MCC), balanced accuracy (BACC), and the area under
ROC curve (AUCROC). Besides generating such metrics, we also
report both the running time to train and test of each AI model.
XAI Global Explanations: Note that such aforementioned
AI models that we build and evaluate are considered black-box
AI models, thus it is essential to provide explanations of such
models and accompanied features (traffic log parameters) and labels
(attack types). Thus, the next step in our framework is having the
XAI step. In the first part of such a step, we generate the global
importance values for each feature where we generate different
graphs in order to analyze the impact of each feature on the AI
model’s decision which can help in generating expectations about
the model’s behavior. We utilize SHAP local graphs for such global
explanations [43 ] which provide the average marginal contribution
of each feature value over all possible coalitions where a coalition
is defined as a combination of features that are used to estimate the
Shapley value of a specific feature.
XAI Local Explanations: Our framework consists of two
local XAI blocks. First, we use the recent well-known Local
Interpretable Model-Agnostic Explanations (LIME) [ 20] for giving
insights of what happens inside an AI algorithm by capturing
feature interactions. We first generate a model that approximates
the original model locally (LIME surrogate model) and then generate
the LIME local explanations. Second, we leverage SHAP [43] via
generating SHAP local graphs (waterfall graphs).
Rationale of Choosing SHAP and LIME: The choice of
these two frameworks for our pipeline is due to the fact that
single-feature explainability methods (such as partial dependence
plots (PDP) [ 25], individual condition expectations plots (ICE) [ 24],
and leave one column out (LOCO) [3]) don’t directly capture
interactions among different features and when the number of
features is high, which makes maintaining local fidelity for such
models becoming increasingly hard. On the other hand, SHAP and
LIME solves such limitations of single-feature models.
Feature Explanation: The final component in our framework
is extracting detailed metrics from the global explanations. In
particular, we extract the model-specific features (i.e., top important
features for each AI model) and intrusion-specific features (i.e., top
features for each intrusion type) for different classes of AI models
that we have and different types (classes) of network intrusions.
RoEduNet-SIMARGL2021: https://www.kaggle.com/datasets/7f91274fa3074d53e983f6eb7a7b24ad1dca136ca967ad0ebe48955e246c24ee
CICIDS-2017: https://www.kaggle.com/datasets/cicdataset/cicids2017

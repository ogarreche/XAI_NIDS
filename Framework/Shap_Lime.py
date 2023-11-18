import shap
import lime
import numpy as np
# Import other necessary libraries such as pandas, numpy, or your specific ML model libraries

def shap_barplots(model, X_train, X_test):
    explainer = shap.KernelExplainer(model.predict, shap.sample(X_train, 20))
    shap_values = explainer.shap_values(X_test[:100])
    shap.summary_plot(shap_values, X_test[:100], plot_type="bar")

def shap_waterfallplot(model, X_train, X_test, instance_index, class_index):
    explainer = shap.KernelExplainer(model.predict, shap.sample(X_train, 20))
    shap_values = explainer.shap_values(X_test[:10])

    explanation = shap.Explanation(
        values=shap_values[class_index][instance_index],
        base_values=explainer.expected_value[class_index],
        data=X_test.iloc[instance_index],
        feature_names=X_test.columns.tolist()
    )

    shap.waterfall_plot(explanation)




def shap_beeswarmplot(model, X_train, X_test):
    explainer = shap.KernelExplainer(model.predict, shap.sample(X_train, 20))
    shap_values = explainer.shap_values(X_test[:10])
    shap.summary_plot(shap_values[0], X_test[:10])


def lime_plot(model, X_train, X_test,Y_train, row_index,class_names):
    import lime
    import lime.lime_tabular
    # Create a Lime explainer object
    explainer = lime.lime_tabular.LimeTabularExplainer(
    X_train.values,
    training_labels=Y_train,
    feature_names=X_train.columns.tolist(),
    class_names=class_names, 
    mode='classification'
    )

    exp = explainer.explain_instance(X_test.iloc[row_index], model.predict_proba, num_features=len(X_train.columns))
    exp.show_in_notebook(show_table=True)

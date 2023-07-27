### Here I will try to do explainability methods, it can come from Run or from main (probably run realistically)
## will do functions for each I think
import shap

### explainability
## this is the fitted model
def explain_func(model,X_train,X_test):

    print(model.summary())
    explainer = shap.DeepExplainer(model, X_train)

    shap_values = explainer.shap_values(X_test)

    return shap_values

#### interpretability
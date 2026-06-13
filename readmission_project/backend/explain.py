import shap
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lime.lime_tabular import LimeTabularExplainer

def get_shap_values(model, patient_df):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(patient_df)
    
    # Handle both list and array outputs
    if isinstance(shap_values, list):
        vals = shap_values[1][0]
        expected = explainer.expected_value[1]
    else:
        vals = shap_values[0]
        expected = explainer.expected_value
    
    return vals, expected


def plot_shap_bar(model, patient_df, features):
    vals, _ = get_shap_values(model, patient_df)
    
    fig, ax = plt.subplots(figsize=(6, 4))
    colors = ['#E24B4A' if v > 0 else '#639922' for v in vals]
    ax.barh(features, vals, color=colors)
    ax.axvline(0, color='gray', linewidth=0.8, linestyle='--')
    ax.set_xlabel("SHAP value (impact on readmission risk)")
    ax.set_title("Why this prediction?")
    plt.tight_layout()
    
    return fig


def get_lime_explanation(model, X_train, patient_df, features):
    explainer = LimeTabularExplainer(
        X_train.values,
        feature_names=features,
        class_names=['No Readmission', 'Readmission'],
        mode='classification'
    )
    
    exp = explainer.explain_instance(
        patient_df.values[0],
        model.predict_proba,
        num_features=5
    )
    
    # Return as list of (feature, weight) tuples
    return exp.as_list()
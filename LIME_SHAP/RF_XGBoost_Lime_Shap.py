# -*- coding: utf-8 -*-
"""
Created on Mon Dec 23 21:33:07 2024

@author: rahul
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    auc,
)
import shap
from lime import lime_tabular
import matplotlib.pyplot as plt

# Load and preprocess the dataset
data = pd.read_excel("Adapted_All_Except_Kidney.xlsx")

# Encode categorical variables
label_encoders = {}
for column in ["Skill", "Test", "Muscle"]:
    label_encoders[column] = LabelEncoder()
    data[column] = label_encoders[column].fit_transform(data[column])

# Features and target
X = data.iloc[:, 4:]
y = data["Skill"]

# Handle missing values
X.fillna(X.mean(), inplace=True)

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, stratify=y, random_state=42
)

# Initialize models
models = {
    "Random Forest": RandomForestClassifier(random_state=42, n_estimators=100),
    "XGBoost": XGBClassifier(
        use_label_encoder=False, eval_metric="mlogloss", random_state=42
    ),
}

# Train and evaluate models
for model_name, model in models.items():
    print(f"--- {model_name} ---")

    # Train model
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)

    # Evaluation metrics
    print("Classification Report:")
    print(
        classification_report(
            y_test, y_pred, target_names=label_encoders["Skill"].classes_
        )
    )
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # ROC-AUC Score
    roc_auc = roc_auc_score(
        y, model.predict_proba(X_scaled), multi_class="ovr"
    )
    print(f"ROC-AUC Score: {roc_auc:.2f}")

    # Plot ROC Curves
    plt.figure(figsize=(10, 8))
    for i in range(y_prob.shape[1]):
        fpr, tpr, _ = roc_curve((y_test == i).astype(int), y_prob[:, i])
        plt.plot(fpr, tpr, label=f"Class {i} (AUC = {auc(fpr, tpr):.2f})")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curves for {model_name}")
    plt.legend()
    plt.grid()
    plt.show()

    # SHAP explanations
    print(f"\nGenerating SHAP explanations for {model_name}...")
    explainer = shap.Explainer(model, X_train)
    shap_values = explainer(X_test)

    # SHAP Summary Plot
    shap.summary_plot(shap_values, X_test, feature_names=data.columns[4:])

    # Feature importance from SHAP
    if len(shap_values.values.shape) == 3:
        shap_values_mean = np.mean(np.abs(shap_values.values), axis=(0, 2))
    else:
        shap_values_mean = np.mean(np.abs(shap_values.values), axis=0)

    important_features = pd.DataFrame(
        {"Feature": data.columns[4:], "Importance": shap_values_mean}
    ).sort_values(by="Importance", ascending=False)

    print("\nTop 10 Important Features:")
    print(important_features.head(10))

    plt.figure(figsize=(10, 8))
    plt.barh(
        important_features["Feature"].head(10),
        important_features["Importance"].head(10),
    )
    plt.xlabel("SHAP Importance")
    plt.ylabel("Feature")
    plt.title(f"Top 10 Important Features for {model_name}")
    plt.gca().invert_yaxis()
    plt.show()

    # LIME explanations
    print(f"\nGenerating LIME explanations for {model_name}...")
    explainer_lime = lime_tabular.LimeTabularExplainer(
        X_train,
        feature_names=data.columns[4:],
        class_names=label_encoders["Skill"].classes_,
        discretize_continuous=True,
    )

    instance_index = 0  # Change as needed
    exp = explainer_lime.explain_instance(
        X_test[instance_index], model.predict_proba, num_features=10
    )
    exp.show_in_notebook(show_table=True, show_all=False)

    # Plot LIME explanation
    plt.figure(figsize=(10, 8))
    lime_weights = dict(exp.as_list())
    lime_features = list(lime_weights.keys())
    lime_importance = list(lime_weights.values())
    plt.barh(lime_features, lime_importance)
    plt.xlabel("LIME Importance")
    plt.ylabel("Feature")
    plt.title(f"LIME Feature Importance for {model_name}")
    plt.gca().invert_yaxis()
    plt.show()

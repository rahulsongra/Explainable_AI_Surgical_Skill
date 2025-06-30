# -*- coding: utf-8 -*-
"""
Created on Sun Dec 22 23:55:21 2024

@author: rahul
"""

from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, label_binarize
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    confusion_matrix,
    roc_auc_score,
    matthews_corrcoef,
    roc_curve,
    auc,
)
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Load and preprocess dataset
df = pd.read_excel("Adapted_All_Except_Kidney.xlsx")

# Encode categorical variables
label_encoders = {}
for column in ["Skill", "Test", "Muscle"]:
    label_encoders[column] = LabelEncoder()
    df[column] = label_encoders[column].fit_transform(df[column])

# Features and target
X = df.iloc[:, 4:]
y = df["Skill"]

# Handle missing values (Impute with column mean)
X.fillna(X.mean(), inplace=True)

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled,
    y_resampled,
    test_size=0.3,
    random_state=42,
    stratify=y_resampled,
)

# Train Random Forest Classifier
rf_model = RandomForestClassifier(random_state=42, n_estimators=100)
rf_model.fit(X_train, y_train)
y_rf_pred = rf_model.predict(X_test)
y_rf_prob = rf_model.predict_proba(X_test)

# Train XGBoost Classifier
xgb_model = XGBClassifier(random_state=42, eval_metric="logloss")
xgb_model.fit(X_train, y_train)
y_xgb_pred = xgb_model.predict(X_test)
y_xgb_prob = xgb_model.predict_proba(X_test)

# Feature importance from Random Forest
importances_rf = pd.Series(rf_model.feature_importances_, index=df.columns[4:])

# Feature importance from XGBoost
importances_xgb = pd.Series(
    xgb_model.feature_importances_, index=df.columns[4:]
)

# Extract task names corresponding to feature columns
test_feature_mapping = X.columns.str.extract(r"(.+)_")[
    0
]  # Adjust regex based on column naming structure

# Group feature importance by task/test for Random Forest
test_importance_rf = importances_rf.groupby(test_feature_mapping).sum()

# Group feature importance by task/test for XGBoost
test_importance_xgb = importances_xgb.groupby(test_feature_mapping).sum()

# Group feature importance by muscle for Random Forest
grouped_muscle_features = X.columns.str.extract(r"^(L_|R_)?(.+)$")[1]
if not grouped_muscle_features.isnull().any():
    muscle_importance_rf = importances_rf.groupby(
        grouped_muscle_features
    ).sum()
    muscle_importance_xgb = importances_xgb.groupby(
        grouped_muscle_features
    ).sum()

    # Plot muscle importance
    if not muscle_importance_rf.empty:
        plt.figure(figsize=(12, 6))
        muscle_importance_rf.sort_values(ascending=False).plot(
            kind="bar", alpha=0.7, label="Random Forest", color="blue"
        )
        muscle_importance_xgb.sort_values(ascending=False).plot(
            kind="bar", alpha=0.7, label="XGBoost", color="orange"
        )
        plt.title("Muscle Importance for Skill Classification")
        plt.ylabel("Importance Score")
        plt.xlabel("Muscle")
        plt.legend()
        plt.show()
    else:
        print("Muscle importance data is empty. No plot generated.")
else:
    print(
        "Error: Grouped muscle features contain NaN values. Check column naming structure."
    )

# Plot test importance
if not test_importance_rf.empty and not test_importance_xgb.empty:
    plt.figure(figsize=(12, 6))
    test_importance_rf.sort_values(ascending=False).plot(
        kind="bar", alpha=0.7, label="Random Forest", color="blue"
    )
    test_importance_xgb.sort_values(ascending=False).plot(
        kind="bar", alpha=0.7, label="XGBoost", color="orange"
    )
    plt.title("Task/Test Importance for Skill Classification")
    plt.ylabel("Importance Score")
    plt.xlabel("Test")
    plt.legend()
    plt.show()
else:
    print("Task importance data is empty. No plot generated.")

# ROC Curve for Random Forest and XGBoost
y_test_bin = label_binarize(y_test, classes=np.unique(y_test))
n_classes = y_test_bin.shape[1]

# Compute ROC for Random Forest
fpr_rf, tpr_rf, roc_auc_rf = {}, {}, {}
for i in range(n_classes):
    fpr_rf[i], tpr_rf[i], _ = roc_curve(y_test_bin[:, i], y_rf_prob[:, i])
    roc_auc_rf[i] = auc(fpr_rf[i], tpr_rf[i])

# Compute ROC for XGBoost
fpr_xgb, tpr_xgb, roc_auc_xgb = {}, {}, {}
for i in range(n_classes):
    fpr_xgb[i], tpr_xgb[i], _ = roc_curve(y_test_bin[:, i], y_xgb_prob[:, i])
    roc_auc_xgb[i] = auc(fpr_xgb[i], tpr_xgb[i])

# Plot ROC Curves
plt.figure(figsize=(10, 8))
for i in range(n_classes):
    plt.plot(
        fpr_rf[i],
        tpr_rf[i],
        label=f"Random Forest Class {i} (AUC = {roc_auc_rf[i]:.2f})",
    )
    plt.plot(
        fpr_xgb[i],
        tpr_xgb[i],
        linestyle="--",
        label=f"XGBoost Class {i} (AUC = {roc_auc_xgb[i]:.2f})",
    )
plt.plot([0, 1], [0, 1], color="gray", linestyle="--")
plt.title("ROC Curve (Random Forest vs XGBoost)")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")
plt.show()

# Display metrics
rf_metrics = {
    "Accuracy": accuracy_score(y_test, y_rf_pred),
    "F1-Score": classification_report(y_test, y_rf_pred, output_dict=True)[
        "macro avg"
    ]["f1-score"],
    "MCC": matthews_corrcoef(y_test, y_rf_pred),
}
xgb_metrics = {
    "Accuracy": accuracy_score(y_test, y_xgb_pred),
    "F1-Score": classification_report(y_test, y_xgb_pred, output_dict=True)[
        "macro avg"
    ]["f1-score"],
    "MCC": matthews_corrcoef(y_test, y_xgb_pred),
}

print("Random Forest Metrics:")
print(rf_metrics)
print("\nClassification Report (Random Forest):")
print(classification_report(y_test, y_rf_pred))

print("\nXGBoost Metrics:")
print(xgb_metrics)
print("\nClassification Report (XGBoost):")
print(classification_report(y_test, y_xgb_pred))

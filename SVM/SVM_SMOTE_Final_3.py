# -*- coding: utf-8 -*-
"""
Created on Sun Dec 22 23:52:11 2024

@author: rahul
"""

# Import necessary libraries
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    confusion_matrix,
    roc_auc_score,
    matthews_corrcoef,
    roc_curve,
    auc,
)
from sklearn.feature_selection import RFE
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


# Function to calculate evaluation metrics
def calculate_metrics(y_test, y_pred, y_prob, n_classes):
    accuracy = accuracy_score(y_test, y_pred)
    f1 = classification_report(y_test, y_pred, output_dict=True)["macro avg"][
        "f1-score"
    ]
    mcc = matthews_corrcoef(y_test, y_pred)

    # Sensitivity (Recall)
    sensitivity = classification_report(y_test, y_pred, output_dict=True)[
        "macro avg"
    ]["recall"]

    # Specificity Calculation
    conf_matrix = confusion_matrix(y_test, y_pred)
    specificity_per_class = []
    for i in range(n_classes):
        tn = conf_matrix.sum() - (
            conf_matrix[i, :].sum()
            + conf_matrix[:, i].sum()
            - conf_matrix[i, i]
        )
        fp = conf_matrix[:, i].sum() - conf_matrix[i, i]
        specificity_per_class.append(tn / (tn + fp))
    specificity = np.mean(specificity_per_class)

    # AUC Calculation
    y_test_bin = label_binarize(y_test, classes=np.unique(y_test))
    auc_score = roc_auc_score(
        y_test_bin, y_prob, average="macro", multi_class="ovr"
    )

    return accuracy, sensitivity, specificity, f1, mcc, auc_score


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

# Recursive Feature Elimination (RFE)
svc_rfe = SVC(kernel="linear", random_state=42)
rfe = RFE(estimator=svc_rfe, n_features_to_select=10, step=1)
X_train_rfe = rfe.fit_transform(X_train, y_train)
X_test_rfe = rfe.transform(X_test)

# Train SVM classifier
svm_model = SVC(kernel="rbf", probability=True, random_state=42)
svm_model.fit(X_train_rfe, y_train)

# Predict the model
y_pred = svm_model.predict(X_test_rfe)
y_prob = svm_model.decision_function(X_test_rfe)

# Calculate metrics
n_classes = len(np.unique(y_test))
accuracy, sensitivity, specificity, f1_score, mcc, auc_score = (
    calculate_metrics(y_test, y_pred, y_prob, n_classes)
)

# Generate a results table
results = pd.DataFrame(
    {
        "Metric": [
            "Accuracy",
            "Sensitivity",
            "Specificity",
            "F1-Score",
            "MCC",
            "AUC",
        ],
        "Score": [
            accuracy,
            sensitivity,
            specificity,
            f1_score,
            mcc,
            auc_score,
        ],
    }
)

# Print Results
print(f"Classification Accuracy: {accuracy:.2f}")
print("\nClassification Report:")
print(
    classification_report(
        y_test, y_pred, target_names=label_encoders["Skill"].classes_
    )
)
print("\nMetrics Summary Table:")
print(results)

# Plot Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
conf_matrix_normalized = (
    conf_matrix.astype("float") / conf_matrix.sum(axis=1)[:, np.newaxis]
)

plt.figure(figsize=(10, 8))
sns.heatmap(
    conf_matrix_normalized,
    annot=True,
    fmt=".2f",
    cmap="Blues",
    xticklabels=label_encoders["Skill"].classes_,
    yticklabels=label_encoders["Skill"].classes_,
)
plt.title("Normalized Confusion Matrix")
plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.show()

# Plot ROC Curves
plt.figure(figsize=(10, 8))
y_test_bin = label_binarize(y_test, classes=np.unique(y_test))
for i in range(n_classes):
    fpr, tpr, _ = roc_curve(
        y_test_bin[:, i], y_prob[:, i] if y_prob.ndim > 1 else y_prob
    )
    plt.plot(fpr, tpr, label=f"Class {i} (AUC = {auc(fpr, tpr):.2f})")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves for SVM Model")
plt.legend()
plt.grid()
plt.show()

# Dimensionality Reduction (PCA and t-SNE)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_resampled)
plt.figure(figsize=(10, 8))
sns.scatterplot(
    x=X_pca[:, 0], y=X_pca[:, 1], hue=y_resampled, palette="viridis"
)
plt.title("PCA Visualization of Feature Space")
plt.show()

tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X_resampled)
plt.figure(figsize=(10, 8))
sns.scatterplot(
    x=X_tsne[:, 0], y=X_tsne[:, 1], hue=y_resampled, palette="viridis"
)
plt.title("t-SNE Visualization of Feature Space")
plt.show()

# Feature importance and task/muscle analysis
feature_names = df.columns[4:]
if hasattr(rfe, "support_") and hasattr(rfe.estimator_, "coef_"):
    selected_features = np.array(feature_names)[rfe.support_]
    feature_importance_values = np.abs(rfe.estimator_.coef_).mean(axis=0)
    important_features_df = pd.DataFrame(
        {"Feature": selected_features, "Importance": feature_importance_values}
    ).sort_values(by="Importance", ascending=False)
else:
    important_features_df = pd.DataFrame(
        {"Feature": feature_names, "Importance": np.zeros(len(feature_names))}
    )

# Aggregate results by muscle and test for selected features
important_muscle_test = (
    df.loc[:, ["Muscle", "Test"] + list(selected_features)]
    .groupby(["Muscle", "Test"])
    .size()
    .reset_index(name="Count")
)

# Plot feature importance
plt.figure(figsize=(12, 8))
plt.barh(
    important_features_df["Feature"].head(10),
    important_features_df["Importance"].head(10),
)
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Top 10 Important Features")
plt.gca().invert_yaxis()
plt.show()

# Plot important muscles and tests
plt.figure(figsize=(12, 8))
sns.barplot(x="Count", y="Muscle", hue="Test", data=important_muscle_test)
plt.title("Important Muscles and Tasks")
plt.xlabel("Count")
plt.ylabel("Muscle")
plt.legend(title="Test")
plt.show()

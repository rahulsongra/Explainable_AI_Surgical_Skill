# -*- coding: utf-8 -*-
"""
Created on Sun Dec 22 23:52:11 2024

@author: rahul
"""
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
)
from sklearn.preprocessing import label_binarize
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

# Train SVM classifier
svm_model = SVC(kernel="rbf", probability=True, random_state=42)
svm_model.fit(X_train, y_train)

# Predict the model
y_pred = svm_model.predict(X_test)
y_prob = svm_model.decision_function(X_test)

# Metrics Calculation
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(
    y_test, y_pred, target_names=label_encoders["Skill"].classes_
)
conf_matrix = confusion_matrix(y_test, y_pred)

# Sensitivity (Recall)
sensitivity = classification_report(y_test, y_pred, output_dict=True)[
    "macro avg"
]["recall"]

# Specificity Calculation
n_classes = len(np.unique(y_test))
specificity_per_class = []
for i in range(n_classes):
    tn = conf_matrix.sum() - (
        conf_matrix[i, :].sum() + conf_matrix[:, i].sum() - conf_matrix[i, i]
    )
    fp = conf_matrix[:, i].sum() - conf_matrix[i, i]
    specificity_per_class.append(tn / (tn + fp))
specificity = np.mean(specificity_per_class)

# F1-Score
f1_score = classification_report(y_test, y_pred, output_dict=True)[
    "macro avg"
]["f1-score"]

# MCC
mcc = matthews_corrcoef(y_test, y_pred)

# AUC
y_test_bin = label_binarize(y_test, classes=np.unique(y_test))
auc = roc_auc_score(y_test_bin, y_prob, average="macro", multi_class="ovr")

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
        "Score": [accuracy, sensitivity, specificity, f1_score, mcc, auc],
    }
)

# Print Results
print(f"Classification Accuracy: {accuracy:.2f}")
print("\nClassification Report:")
print(classification_rep)
print("\nMetrics Summary Table:")
print(results)

# Plot Confusion Matrix
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

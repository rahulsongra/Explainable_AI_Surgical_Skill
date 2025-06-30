# -*- coding: utf-8 -*-
"""
Created on Sun Dec 22 23:46:10 2024

@author: rahul
"""
from sklearn.metrics import roc_auc_score, matthews_corrcoef, roc_curve
from sklearn.preprocessing import label_binarize
import pandas as pd

# Predict probabilities for AUC calculation
y_prob = svm_model.decision_function(X_test)

# Binarize the labels for AUC and calculate metrics for multi-class
y_test_bin = label_binarize(y_test, classes=np.unique(y_test))
n_classes = y_test_bin.shape[1]

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

# Sensitivity (Recall)
sensitivity = classification_report(y_test, y_pred, output_dict=True)[
    "macro avg"
]["recall"]

# Specificity
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

# Display the results table
print(results)

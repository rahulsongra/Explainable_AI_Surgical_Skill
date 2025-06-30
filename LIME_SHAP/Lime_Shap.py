import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
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

# Train Naive Bayes model
model = GaussianNB()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)

# Evaluation
print("Classification Report:")
print(
    classification_report(
        y_test, y_pred, target_names=label_encoders["Skill"].classes_
    )
)
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# ROC-AUC Score
y_test_bin = LabelEncoder().fit_transform(y_test)
roc_auc = roc_auc_score(y_test_bin, y_prob, multi_class="ovr")
print(f"ROC-AUC Score: {roc_auc:.2f}")


# Wrapper for GaussianNB to make it compatible with SHAP
class GaussianNBWrapper:
    def __init__(self, model):
        self.model = model

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)


# Create a wrapper instance
wrapped_model = GaussianNBWrapper(model)

# Generate SHAP explanations
print("\nGenerating SHAP explanations...")
explainer = shap.Explainer(wrapped_model.predict_proba, X_train)
shap_values = explainer(X_test)

# Debugging SHAP values and feature alignment
print(f"SHAP values shape: {shap_values.values.shape}")
print(f"Number of features in the dataset: {len(data.columns[4:])}")

# Ensure SHAP values align with features
if len(shap_values.values.shape) == 3:
    # Multi-class SHAP values: Aggregate across classes and samples
    shap_values_mean = np.mean(np.abs(shap_values.values), axis=(0, 2))
else:
    # Binary classification: Aggregate across samples
    shap_values_mean = np.mean(np.abs(shap_values.values), axis=0)

# Verify alignment
features = data.columns[4:]
if len(features) != len(shap_values_mean):
    raise ValueError(
        f"Mismatch between feature names ({len(features)}) and SHAP values length ({len(shap_values_mean)})."
    )

# Creating the DataFrame for feature importances
important_features = pd.DataFrame(
    {"Feature": features, "Importance": shap_values_mean}
).sort_values(by="Importance", ascending=False)

print("\nTop 10 Important Features:")
print(important_features.head(10))

# Plot Important Features
plt.figure(figsize=(10, 8))
plt.barh(
    important_features["Feature"].head(10),
    important_features["Importance"].head(10),
)
plt.xlabel("SHAP Importance")
plt.ylabel("Feature")
plt.title("Top 10 Important Features")
plt.gca().invert_yaxis()
plt.show()

# LIME Explanation
print("\nGenerating LIME explanations...")
explainer_lime = lime_tabular.LimeTabularExplainer(
    X_train,
    feature_names=data.columns[4:],
    class_names=label_encoders["Skill"].classes_,
    discretize_continuous=True,
)

# Explain a specific prediction
instance_index = 0  # Change as needed
exp = explainer_lime.explain_instance(
    X_test[instance_index], model.predict_proba, num_features=10
)
exp.show_in_notebook(show_table=True, show_all=False)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve, mean_squared_error, auc
)
from sklearn.inspection import PartialDependenceDisplay
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

# Load the data
file_path = "heart-attack-risk-prediction-dataset.csv"
df = pd.read_csv(file_path)

# Remove duplicate response variable
df = df.drop(columns=['Heart Attack Risk (Text)'])

# Encode Gender column: Male = 1, Female = -1
df['Gender'] = df['Gender'].map({'Male': 1, 'Female': -1})

# Impute missing values with median for each column
df = df.apply(lambda col: col.fillna(col.median()) if col.isnull().any() else col)

# Define features and response variable
X = df.drop(columns=['Heart Attack Risk (Binary)'])
y = df['Heart Attack Risk (Binary)']

# First split: 60% train, 40% temp
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.4, random_state=123, stratify=y
)

# Second split: 50% validation, 50% test from temp (20% each of original)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=123, stratify=y_temp
)

# Confirm splits
print(f"Training set size: {X_train.shape}")
print(f"Validation set size: {X_val.shape}")
print(f"Test set size: {X_test.shape}")

##################################
# LDA and QDA ####################
##################################

# Define a helper function to calculate evaluation metrics
def evaluate_model(y_true, y_pred, y_prob, model_name):
    cm = confusion_matrix(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)  # Sensitivity
    spec = cm[0, 0] / (cm[0, 0] + cm[0, 1])  # Specificity
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)
    mse = mean_squared_error(y_true, y_prob)

    print(f"--- {model_name} ---")
    print("Confusion Matrix:\n", cm)
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall (Sensitivity): {rec:.4f}")
    print(f"Specificity: {spec:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"AUC: {auc:.4f}")
    print(f"MSE: {mse:.4f}")

    return {
        'Model': model_name,
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'Specificity': spec,
        'F1': f1,
        'AUC': auc,
        'MSE': mse
    }


# Train and evaluate LDA
lda = LinearDiscriminantAnalysis()
lda.fit(X_train, y_train)

# Validation set evaluation
y_val_pred_lda = lda.predict(X_val)
y_val_prob_lda = lda.predict_proba(X_val)[:, 1]

lda_metrics = evaluate_model(y_val, y_val_pred_lda, y_val_prob_lda, "LDA")

# Train and evaluate QDA
qda = QuadraticDiscriminantAnalysis()
qda.fit(X_train, y_train)

# Validation set evaluation
y_val_pred_qda = qda.predict(X_val)
y_val_prob_qda = qda.predict_proba(X_val)[:, 1]

qda_metrics = evaluate_model(y_val, y_val_pred_qda, y_val_prob_qda, "QDA")

# Plot ROC Curves for LDA and QDA
fpr_lda, tpr_lda, _ = roc_curve(y_val, y_val_prob_lda)
fpr_qda, tpr_qda, _ = roc_curve(y_val, y_val_prob_qda)

plt.figure()
plt.plot(fpr_lda, tpr_lda, label="LDA (AUC = {:.2f})".format(lda_metrics['AUC']))
plt.plot(fpr_qda, tpr_qda, label="QDA (AUC = {:.2f})".format(qda_metrics['AUC']))
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves: LDA vs QDA")
plt.legend()
plt.grid()
plt.show()

# Partial Dependence Plots (Top 2 features for LDA)
features_to_plot = X_train.columns[:2]  # you can manually choose based on importance later
PartialDependenceDisplay.from_estimator(lda, X_val, features_to_plot)
plt.title("Partial Dependence Plots (LDA)")
plt.show()

# Store results to aggregate later
model_results = [lda_metrics, qda_metrics]

##################################
# K-nn Classification ############
##################################

# Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Try a range of k values
k_range = range(1, 51)
k_scores = []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_scaled, y_train)
    y_test_pred = knn.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_test_pred)
    k_scores.append(acc)

# Plot k vs. test accuracy
plt.figure()
plt.plot(k_range, k_scores, marker='o')
plt.xlabel("k (Number of Neighbors)")
plt.ylabel("Test Accuracy")
plt.title("k-NN: k vs Test Accuracy")
plt.grid()
plt.show()

# Print top k values with highest accuracy
best_k_indices = np.argsort(k_scores)[-5:][::-1]  # Top 5
for idx in best_k_indices:
    print(f"k = {k_range[idx]} → Test Accuracy = {k_scores[idx]:.4f}")

# Evaluate the best k with full metrics
best_k = k_range[best_k_indices[0]]
knn_best = KNeighborsClassifier(n_neighbors=best_k)
knn_best.fit(X_train_scaled, y_train)

y_test_pred_knn = knn_best.predict(X_test_scaled)
y_test_prob_knn = knn_best.predict_proba(X_test_scaled)[:, 1]

knn_metrics = evaluate_model(y_test, y_test_pred_knn, y_test_prob_knn, f"k-NN (k={best_k})")
model_results.append(knn_metrics)

# Train k-NN with k=25
knn_25 = KNeighborsClassifier(n_neighbors=25)
knn_25.fit(X_train_scaled, y_train)

# Predict on the test set
y_test_pred_knn25 = knn_25.predict(X_test_scaled)
y_test_prob_knn25 = knn_25.predict_proba(X_test_scaled)[:,1]

# Evaluate the model
knn25_metrics = evaluate_model(y_test, y_test_pred_knn25, y_test_prob_knn25, "k-NN (k=25)")

# Save the results
model_results.append(knn25_metrics)

##################################
# Random Forests #################
##################################

# Train a full Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=123)
rf.fit(X_train, y_train)

# Evaluate full model on test set
y_test_pred_rf = rf.predict(X_test)
y_test_prob_rf = rf.predict_proba(X_test)[:,1]

rf_metrics = evaluate_model(y_test, y_test_pred_rf, y_test_prob_rf, "Random Forest (All Features)")

# Plot Feature Importances
importances = rf.feature_importances_
feature_names = X_train.columns
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10,6))
plt.title("Feature Importances - Random Forest (All Features)")
plt.bar(range(10), importances[indices[:10]], align='center')
plt.xticks(range(10), [feature_names[i] for i in indices[:10]], rotation=45, ha='right')
plt.tight_layout()
plt.show()

# Select top 10 features
top_10_features = feature_names[indices[:10]]
X_train_top10 = X_train[top_10_features]
X_val_top10 = X_val[top_10_features]
X_test_top10 = X_test[top_10_features]

# Train new Random Forest with top 10 features
rf_top10 = RandomForestClassifier(n_estimators=100, random_state=123)
rf_top10.fit(X_train_top10, y_train)

# Evaluate top-10-feature model on test set
y_test_pred_rf_top10 = rf_top10.predict(X_test_top10)
y_test_prob_rf_top10 = rf_top10.predict_proba(X_test_top10)[:,1]

rf_top10_metrics = evaluate_model(y_test, y_test_pred_rf_top10, y_test_prob_rf_top10, "Random Forest (Top 10 Features)")

# Store results
model_results.append(rf_metrics)
model_results.append(rf_top10_metrics)

##################################
# LASSO Logistic Regression ######
##################################

# Standardize features
scaler_lasso = StandardScaler()
X_train_scaled_lasso = scaler_lasso.fit_transform(X_train)
X_val_scaled_lasso = scaler_lasso.transform(X_val)
X_test_scaled_lasso = scaler_lasso.transform(X_test)

# Define range of C values (regularization strength inversely)
Cs = np.logspace(-4, 4, 100)

# Fit Logistic Regression with L1 penalty (LASSO)
lasso_cv = LogisticRegressionCV(
    Cs=Cs, cv=5, penalty='l1', solver='saga',
    scoring='neg_mean_squared_error', random_state=123, max_iter=5000
)
lasso_cv.fit(X_train_scaled_lasso, y_train)

# Calculate mean validation MSE
mean_mse = -lasso_cv.scores_[1].mean(axis=0)  # Class 1 score
log_Cs = np.log(lasso_cv.Cs_)

# Find minimized and 1-SE rule C
min_idx = np.argmin(mean_mse)
min_C = lasso_cv.Cs_[min_idx]

mse_min = mean_mse[min_idx]
se = np.std(-lasso_cv.scores_[1], axis=0)[min_idx] / np.sqrt(5)  # 5-fold CV

candidates = np.where(mean_mse <= (mse_min + se))[0]
se_C = lasso_cv.Cs_[candidates[-1]]  # Largest C under 1-SE rule

print(f"Best C (minimized MSE): {min_C}")
print(f"Best C (1-SE rule): {se_C}")

# Plot log(C) vs Validation MSE with vertical lines
plt.figure(figsize=(8, 6))
plt.plot(log_Cs, mean_mse, marker='o', label='Validation MSE')
plt.axvline(np.log(min_C), color='red', linestyle='--', label=f'Min C (log={np.log(min_C):.2f})')
plt.axvline(np.log(se_C), color='green', linestyle='--', label=f'1-SE C (log={np.log(se_C):.2f})')
plt.xlabel("log(C)")
plt.ylabel("Validation MSE")
plt.title("LASSO: log(C) vs Validation MSE")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()


# Helper function to train LASSO with a given C and threshold
def train_lasso_and_evaluate(C_value, threshold, model_name):
    from sklearn.linear_model import LogisticRegression
    lasso = LogisticRegression(
        C=C_value, penalty='l1', solver='saga',
        random_state=123, max_iter=5000
    )
    lasso.fit(X_train_scaled_lasso, y_train)

    y_test_prob = lasso.predict_proba(X_test_scaled_lasso)[:, 1]
    y_test_pred = (y_test_prob >= threshold).astype(int)

    # Evaluate metrics
    metrics = evaluate_model(y_test, y_test_pred, y_test_prob, model_name)

    # Print nonzero coefficients
    coef = lasso.coef_[0]
    features = X_train.columns
    nonzero_indices = np.where(coef != 0)[0]

    print(f"--- Coefficients for {model_name} ---")
    for idx in nonzero_indices:
        print(f"{features[idx]}: {coef[idx]:.4f}")

    # Plot feature importance
    if len(nonzero_indices) > 0:
        plt.figure(figsize=(8, 5))
        important_features = features[nonzero_indices]
        important_coefs = coef[nonzero_indices]
        plt.barh(important_features, important_coefs)
        plt.xlabel("Coefficient Value")
        plt.title(f"Feature Importance: {model_name}")
        plt.grid()
        plt.tight_layout()
        plt.show()

    return metrics


# Train and evaluate the 4 LASSO models
lasso_min_05 = train_lasso_and_evaluate(min_C, 0.5, "LASSO (Min C, Threshold 0.5)")
lasso_se_05 = train_lasso_and_evaluate(se_C, 0.5, "LASSO (1SE C, Threshold 0.5)")
lasso_min_04 = train_lasso_and_evaluate(min_C, 0.4, "LASSO (Min C, Threshold 0.4)")
lasso_se_04 = train_lasso_and_evaluate(se_C, 0.4, "LASSO (1SE C, Threshold 0.4)")

# Save all LASSO model results
model_results.extend([lasso_min_05, lasso_se_05, lasso_min_04, lasso_se_04])

##################################
# Support Vector Machine #########
##################################

# Train initial basic SVM (RBF kernel)
svm_basic = SVC(kernel='rbf', probability=True, random_state=123)
svm_basic.fit(X_train_scaled, y_train)

# Predict and evaluate on test set
y_test_pred_svm_basic = svm_basic.predict(X_test_scaled)
y_test_prob_svm_basic = svm_basic.predict_proba(X_test_scaled)[:,1]

svm_basic_metrics = evaluate_model(y_test, y_test_pred_svm_basic, y_test_prob_svm_basic, "SVM (Basic RBF)")
model_results.append(svm_basic_metrics)

# Hyperparameter tuning
param_grid = {
    'C': [0.1, 1, 10],  # You can increase later
    'kernel': ['linear', 'rbf']
}

svm_grid = GridSearchCV(
    SVC(probability=True, random_state=123),  # Add class_weight='balanced' optionally
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1  # Shows progress!
)

# Fit the model
svm_grid.fit(X_train_scaled, y_train)

# Retrieve best model
print("Best SVM Parameters:", svm_grid.best_params_)
svm_best = svm_grid.best_estimator_

# Evaluate on test set
y_test_pred_svm_best = svm_best.predict(X_test_scaled)
y_test_prob_svm_best = svm_best.predict_proba(X_test_scaled)[:,1]

svm_best_metrics = evaluate_model(y_test, y_test_pred_svm_best, y_test_prob_svm_best, "SVM (Tuned)")
model_results.append(svm_best_metrics)

# Plot ROC Curve for best SVM

fpr, tpr, _ = roc_curve(y_test, y_test_prob_svm_best)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, label=f"SVM (Tuned) ROC curve (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve: Best Tuned SVM")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

# Final Model Comparison Table

# Create a DataFrame from model results
results_df = pd.DataFrame(model_results)

# Round the numbers for better display
results_df = results_df.round(4)

# Display the final comparison
print("\n=== Final Model Comparison ===")
print(results_df)

# Sort by highest AUC
results_sorted = results_df.sort_values(by='AUC', ascending=False)
print("\n=== Models Sorted by AUC ===")
print(results_sorted)

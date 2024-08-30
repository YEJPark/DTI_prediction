import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, roc_auc_score, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import seaborn as sns

# Load data
df = pd.read_csv("./data/gene_40.csv")

# Data preprocessing
# df['gender'] = df['gender'].map({'F': 1, 'M': 0})

# Set Feature and Target 
X = df.drop(['outcome', 'gender', 'age'], axis=1)
y = df['outcome']

# Split the dataset (Train/Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=1)

# Scale adjustment
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create and train the Logistic Regression model
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# Analyze regression coefficients (after model training)
coef = pd.Series(model.coef_[0], index=X_train.columns)

# Convert to DataFrame
coef_df = pd.DataFrame({
    'Gene_name': coef.index,
    'Coefficient': coef.values,
})

# Add absolute value column
coef_df['Abs_Coefficient'] = coef_df['Coefficient'].abs()

# Sort in descending order based on absolute values
coef_df_sorted = coef_df.sort_values(by='Abs_Coefficient', ascending=False)

# Save sorted data to CSV
coef_df_sorted.to_csv('./result/final_regression_coefficients_sorted_2.csv', index=False)

# Check output
print(coef_df_sorted.head(10))

# Evaluate on the Test dataset
y_test_pred = model.predict(X_test_scaled)
y_test_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

test_accuracy = accuracy_score(y_test, y_test_pred)
test_precision = precision_score(y_test, y_test_pred)
test_recall = recall_score(y_test, y_test_pred)
test_confusion = confusion_matrix(y_test, y_test_pred)
test_auroc = roc_auc_score(y_test, y_test_pred_proba)

# Visualize and save the AUROC Curve (on the Test dataset)
fpr, tpr, _ = roc_curve(y_test, y_test_pred_proba)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'AUROC: {test_auroc:.2f}')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('AUROC Curve - Test Set')
plt.legend(loc='lower right')
plt.savefig('./result/test_auroc_curve.png')
plt.show()

# Visualize and save the Confusion Matrix (on the Test dataset)
plt.figure(figsize=(8, 6))
sns.heatmap(test_confusion, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix - Test Set')
plt.savefig('./result/test_confusion_matrix.png')
plt.show()

# Save evaluation results
results = {
    'Accuracy': test_accuracy,
    'Precision': test_precision,
    'Recall': test_recall,
    'confusion': test_confusion,
    'AUROC': test_auroc
}

results_df = pd.DataFrame([results])
results_df.to_csv('./result/test_results.csv', index=False)

# Output results
print(f'Test Accuracy: {test_accuracy}')
print(f'Test Precision: {test_precision}')
print(f'Test Recall: {test_recall}')
print(f'Test Confusion Matrix:\n{test_confusion}')
print(f'Test AUROC: {test_auroc}')

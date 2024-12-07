import pandas as pd
import numpy as np
import data_preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.inspection import permutation_importance, partial_dependence, PartialDependenceDisplay
from sklearn.feature_selection import RFECV
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import shap
import seaborn as sns
from tqdm import tqdm
import math

original_data = pd.read_csv('rentfaster.csv')
data = data_preprocessing.process_data(original_data)
# all_data = data_preprocessing.process_data(original_data)
# data = all_data.sample(n=100, random_state=42)

X = data.drop('price', axis=1)  # Drop the target column to get the features
y = data['price']  # Select the target column
n_features = X.shape[1]
feature_names = X.columns.tolist()

# Initialize the model
rf = RandomForestRegressor(random_state=42)

# Cross-validation setup
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_results = []

feature_importances = np.zeros(X.shape[1])

# ================== Recursive Feature Elimination ==================
print("[Recursive Feature Elimination] Running...")
rfe_selector = RFECV(rf, step=1, cv=kf, scoring='neg_mean_squared_error')
rfe_selector.fit(X, y)
selected_features = X.columns[rfe_selector.support_]
# Extract the feature importances of the selected features
rfe_feature_importances = rfe_selector.estimator_.feature_importances_

# Create a DataFrame to rank the features
rfe_importance_df = pd.DataFrame({
    'Feature': selected_features,
    'Importance': rfe_feature_importances
})

# Sort features by importance in descending order
rfe_importance_df = rfe_importance_df.sort_values(by='Importance', ascending=False)

#print(f"[Recursive Feature Elimination] Selected Features by RFE: {selected_features}")

n_folds = 5 
n_rows = 2  
n_cols = math.ceil(n_folds / n_rows)  # Calculate the number of columns needed
fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(20, 10), sharey=True)
axes = axes.flatten()  # Flatten axes array for easy indexing
fig.suptitle("Feature Importance Across 5 Folds", fontsize=16)

# Perform cross-validation and collect feature importances
print("[Cross-Validation on Permutation Importance] Running...")
for fold, (train_idx, test_idx) in enumerate((tqdm(kf.split(X), total=kf.get_n_splits(), desc="K-Fold Progress"))):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    cv_results.append(mse)
    
    # Permutation Importance
    perm_importance = permutation_importance(rf, X_test, y_test, random_state=42)
    feature_importances += perm_importance.importances_mean
    
    # Plot feature importance for this fold in the corresponding subplot
    ax = axes[fold]
    ax.bar(X.columns, perm_importance.importances_mean)
    ax.set_title(f"Fold {fold + 1}")
    ax.set_xticklabels(X.columns, rotation=45)
    ax.set_ylabel("Importance")

# Hide unused subplots if any
for i in range(n_folds, len(axes)):
    axes[i].set_visible(False)

plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave space for the main title
plt.show()


# Average feature importance
print("[Average Feature Importance] Running...")
feature_importances /= kf.get_n_splits()
plt.bar(X.columns, feature_importances)
plt.title("Average Feature Importance Across Folds")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Filter for selected features (from RFECV)
selected_features_importances = feature_importances[rfe_selector.support_]

# Create a DataFrame for ranking
selected_features_df = pd.DataFrame({
    'Feature': selected_features,
    'Importance': selected_features_importances
})

# Sort features by importance in descending order
selected_features_df = selected_features_df.sort_values(by='Importance', ascending=False)


# ================== SHAP analysis ==================
print("[SHAP] Running...")
print("[SHAP] Initializing SHAP explainer...")
explainer = shap.TreeExplainer(rf)
print("[SHAP] Computing SHAP values...")
shap_values = explainer.shap_values(X)
print("[SHAP ]Generating summary plot...")
shap.summary_plot(shap_values, X)

# ================== Partial Dependence Plots ==================
print("[Partial Dependence] Running...")
n_features = len(selected_features)
fig, axes = plt.subplots(
    nrows=2, 
    ncols=math.ceil(n_features / 2), 
    figsize=(5 * min(n_features, 2), 5 * math.ceil(n_features / 2)), 
    constrained_layout=True
)

# Flatten axes array for consistent indexing
if n_features == 1:
    axes = [axes]
else:
    axes = axes.ravel()

for i, feature in enumerate(selected_features):
    display = PartialDependenceDisplay.from_estimator(rf, X, [feature], ax=axes[i])
    axes[i].set_title(f"Partial Dependence Plot for {feature}")


for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

plt.tight_layout()
plt.subplots_adjust(
    top=0.967,  
    left = 0.06,
    bottom = 0.055,
    hspace=0.186, 
    wspace=0.288 
)
plt.show()
# print("[Partial Dependence] Running...")
# for feature in selected_features:
#     display = PartialDependenceDisplay.from_estimator(rf, X, [feature])
#     display.figure_.suptitle(f"Partial Dependence Plot for {feature}")
#     plt.tight_layout()
#     plt.show()


# ================== Correlation Analysis ==================
print("[Correlation] Running...")
correlation_matrix = data.corr()
plt.figure(figsize=(12, 10))
plt.title("Correlation Heatmap")
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
plt.tight_layout()
plt.show()

# ================== Numerical Result ====================
print("Important Features after RFE:")
for idx, (_, row) in enumerate(rfe_importance_df.iterrows(), start=1):
    print(f"Rank {idx}: Feature: {row['Feature']}, Importance: {row['Importance']}")

print("\nImportant Features after cross validation:")
for idx, (_, row) in enumerate(selected_features_df.iterrows(), start=1):
    print(f"Rank {idx}: Feature: {row['Feature']}, Importance: {row['Importance']}")



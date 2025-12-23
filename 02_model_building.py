"""
Script 2 - Model Building
==========================
Load learning set (already merged in Script 1), build preprocessing pipeline,
test multiple algorithms with GridSearchCV, select best model.

All data merging done in Script 1. Only modeling here.
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("SCRIPT 2 - MODEL BUILDING")
print("="*80)

# Configuration
RANDOM_STATE = 42
DEBUG = False  # Set True to use subsample
DEBUG_SIZE = 5000

# =============================================================================
# LOAD LEARNING SET (already merged in Script 1)
# =============================================================================

print("\nLoading learning set...")
learning = pd.read_pickle('learning.pkl')
print(f"Shape: {learning.shape}")

# Prepare X and y
X = learning.drop(columns=['target', 'primary_key'])
y = learning['target']

# Debug subsample
if DEBUG:
    X = X.sample(n=DEBUG_SIZE, random_state=RANDOM_STATE)
    y = y.loc[X.index]
    print(f"Using debug subsample: {X.shape}")

# =============================================================================
# IDENTIFY FEATURE TYPES BY DTYPE
# =============================================================================

print("\nIdentifying feature types...")

# Numeric: float64, int64 (excluding indicators which are already 0/1)
numeric_features = X.select_dtypes(include=['float64', 'int64']).columns.tolist()

# Categorical: object, category, bool types
categorical_features = X.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

# Remove known numeric-coded categoricals from numeric list
numeric_coded_categoricals = [
    'Employee_count', 'Employee_count_retired', 'EMPLOYER_TYPE', 'EMPLOYER_TYPE_retired',
    'JOB_CATEGORY', 'JOB_CATEGORY_retired', 'city_type'
]
for col in numeric_coded_categoricals:
    if col in numeric_features:
        numeric_features.remove(col)
        if col in X.columns:
            categorical_features.append(col)

print(f"Numeric: {len(numeric_features)}, Categorical: {len(categorical_features)}")

# =============================================================================
# CREATE PREPROCESSING PIPELINE
# =============================================================================

print("\nBuilding preprocessing pipeline...")

# Numeric: Impute → Scale (scaling kept for flexibility, not harmful for trees)
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Categorical: Impute → OneHotEncode (SPARSE output to avoid memory issues)
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))  # sparse by default
])

# Combine
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ],
    sparse_threshold=0.3  # Keep sparse if >30% sparse
)

# =============================================================================
# BASELINE MODEL - Simple Decision Tree
# =============================================================================

print("\n" + "="*80)
print("BASELINE MODEL")
print("="*80)

baseline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', DecisionTreeRegressor(max_depth=5, random_state=RANDOM_STATE))
])

# Quick 3-fold CV
cv_quick = KFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
baseline_scores = cross_val_score(baseline, X, y, cv=cv_quick,
                                   scoring='neg_root_mean_squared_error', n_jobs=-1)
baseline_rmse = -baseline_scores.mean()

print(f"Baseline RMSE: {baseline_rmse:.4f} ± {baseline_scores.std():.4f}")

# Train and save
baseline.fit(X, y)
joblib.dump(baseline, 'baseline_model.joblib')
print("Saved: baseline_model.joblib")

# =============================================================================
# MODEL 1 - Random Forest (Reference)
# =============================================================================

print("\n" + "="*80)
print("RANDOM FOREST")
print("="*80)

rf_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1))
])

# Smaller grid for faster training (16 combinations instead of 54)
param_grid_rf = {
    'regressor__n_estimators': [100, 300],
    'regressor__max_depth': [20, None],
    'regressor__min_samples_split': [2, 10],
    'regressor__min_samples_leaf': [1, 2]
}

cv_main = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
rf_search = GridSearchCV(rf_pipeline, param_grid_rf, cv=cv_main,
                         scoring='neg_root_mean_squared_error',
                         n_jobs=-1, verbose=1)

print("Training Random Forest...")
rf_search.fit(X, y)

rf_rmse = -rf_search.best_score_
rf_std = rf_search.cv_results_['std_test_score'][rf_search.best_index_]
print(f"\nBest params: {rf_search.best_params_}")
print(f"CV RMSE: {rf_rmse:.4f} ± {rf_std:.4f}")

# =============================================================================
# MODEL 2 - Gradient Boosting
# =============================================================================

print("\n" + "="*80)
print("GRADIENT BOOSTING")
print("="*80)

gb_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', GradientBoostingRegressor(random_state=RANDOM_STATE))
])

# Focused grid for faster training
param_grid_gb = {
    'regressor__n_estimators': [100, 200, 300],
    'regressor__learning_rate': [0.01, 0.05, 0.1],
    'regressor__max_depth': [3, 5, 7],
    'regressor__min_samples_split': [2, 5]
}

gb_search = GridSearchCV(gb_pipeline, param_grid_gb, cv=cv_main,
                         scoring='neg_root_mean_squared_error',
                         n_jobs=-1, verbose=1)

print("Training Gradient Boosting...")
gb_search.fit(X, y)

gb_rmse = -gb_search.best_score_
gb_std = gb_search.cv_results_['std_test_score'][gb_search.best_index_]
print(f"\nBest params: {gb_search.best_params_}")
print(f"CV RMSE: {gb_rmse:.4f} ± {gb_std:.4f}")

# =============================================================================
# SELECT BEST MODEL
# =============================================================================

print("\n" + "="*80)
print("MODEL SELECTION")
print("="*80)

print(f"\nRandom Forest:     {rf_rmse:.4f} ± {rf_std:.4f}")
print(f"Gradient Boosting: {gb_rmse:.4f} ± {gb_std:.4f}")

# Select simplest among best (tolerance-based)
tolerance = 0.005
noise = max(rf_std, gb_std)

if (gb_rmse + max(tolerance, noise)) < rf_rmse:
    final_model = gb_search.best_estimator_
    final_name = "Gradient Boosting"
    final_rmse = gb_rmse
    print(f"\nSelected: {final_name} (significantly better)")
else:
    final_model = rf_search.best_estimator_
    final_name = "Random Forest"
    final_rmse = rf_rmse
    print(f"\nSelected: {final_name} (simpler model, comparable performance)")

# =============================================================================
# SAVE FINAL MODEL
# =============================================================================

joblib.dump(final_model, 'final_model.joblib')

print("\n" + "="*80)
print("COMPLETE")
print("="*80)
print(f"Final model: {final_name}")
print(f"Expected RMSE on new data: {final_rmse:.4f}")
print("Saved: final_model.joblib")
print("="*80)
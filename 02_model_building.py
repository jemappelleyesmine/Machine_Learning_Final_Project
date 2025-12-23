"""
Script 2 - Model Building
======================================
Load learning set (already merged in Script 1), build preprocessing pipeline,
test multiple algorithms with GridSearchCV, select best model.
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import warnings

warnings.filterwarnings('ignore')

print("=" * 80)
print("SCRIPT 2 - MODEL BUILDING")
print("=" * 80)

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
    print(f" Using debug subsample: {X.shape}")
else:
    print(f" Using full dataset: {X.shape}")

# =============================================================================
# IDENTIFY FEATURE TYPES BY DTYPE
# =============================================================================

print("\nIdentifying feature types...")

# Numeric: float64, int64
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

print(f"  Numeric: {len(numeric_features)}")
print(f"  Categorical: {len(categorical_features)}")

# =============================================================================
# CREATE PREPROCESSING PIPELINE
# =============================================================================

print("\nBuilding preprocessing pipeline...")

# Numeric: Impute ONLY (no scaling for tree models)
# Trees/forests/boosting don't benefit from scaling and it wastes compute
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median'))
])

# Categorical: Impute to OneHotEncode (SPARSE output to avoid memory issues)
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

print(" Pipeline created (imputation + one-hot encoding)")

# =============================================================================
# BASELINE MODEL - Simple Decision Tree
# =============================================================================

print("\n" + "=" * 80)
print("BASELINE MODEL - Decision Tree")
print("=" * 80)

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

print("\n" + "=" * 80)
print("MODEL 1 - RANDOM FOREST")
print("=" * 80)

rf_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1))
])

# EXTENDED GRID (no edge configurations)
# Added values beyond previous optimal to ensure not at edge
param_grid_rf = {
    'regressor__n_estimators': [200, 300, 400],  # Extended: was [100, 300]
    'regressor__max_depth': [15, 20, None],  # Extended: was [20, None]
    'regressor__min_samples_split': [2, 5, 10],  # Extended: was [2, 10]
    'regressor__min_samples_leaf': [1, 2, 4]  # Extended: was [1, 2]
}

print("Parameter grid:")
for param, values in param_grid_rf.items():
    print(f"  {param.split('__')[1]}: {values}")
print(f"Total combinations: {np.prod([len(v) for v in param_grid_rf.values()])}")

cv_main = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
rf_search = GridSearchCV(rf_pipeline, param_grid_rf, cv=cv_main,
                         scoring='neg_root_mean_squared_error',
                         n_jobs=-1, verbose=1)

print("\nTraining Random Forest...")
rf_search.fit(X, y)

rf_rmse = -rf_search.best_score_
rf_std = rf_search.cv_results_['std_test_score'][rf_search.best_index_]
print(f"\n Random Forest complete")
print(f"  Best params: {rf_search.best_params_}")
print(f"  CV RMSE: {rf_rmse:.4f} ± {rf_std:.4f}")

# Check if optimal is at edge (warning)
best_params_rf = rf_search.best_params_
for param, value in best_params_rf.items():
    param_name = param.split('__')[1]
    grid_values = param_grid_rf[param]

    # Handle None separately (represents unbounded, not an edge)
    if value is None:
        continue  # None is unbounded, not an edge configuration

    # Filter out None from grid for comparison
    numeric_values = [v for v in grid_values if v is not None]

    if numeric_values:  # Only check if there are numeric values
        if value == max(numeric_values):
            print(f"  WARNING: {param_name}={value} is at upper edge!")
        elif value == min(numeric_values):
            print(f"  WARNING: {param_name}={value} is at lower edge!")

# =============================================================================
# MODEL 2 - Gradient Boosting
# =============================================================================

print("\n" + "=" * 80)
print("MODEL 2 - GRADIENT BOOSTING")
print("=" * 80)

gb_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', GradientBoostingRegressor(random_state=RANDOM_STATE))
])

# EXTENDED GRID (no edge configurations)
# Added values beyond previous optimal to ensure not at edge
param_grid_gb = {
    'regressor__n_estimators': [200, 300, 400],  # Extended: was [100, 200, 300]
    'regressor__learning_rate': [0.05, 0.1, 0.15],  # Extended: was [0.01, 0.05, 0.1]
    'regressor__max_depth': [5, 7, 9],  # Extended: was [3, 5, 7]
    'regressor__min_samples_split': [2, 5, 10]  # Extended: was [2, 5]
}

print("Parameter grid:")
for param, values in param_grid_gb.items():
    print(f"  {param.split('__')[1]}: {values}")
print(f"Total combinations: {np.prod([len(v) for v in param_grid_gb.values()])}")

gb_search = GridSearchCV(gb_pipeline, param_grid_gb, cv=cv_main,
                         scoring='neg_root_mean_squared_error',
                         n_jobs=-1, verbose=1)

print("\nTraining Gradient Boosting...")
gb_search.fit(X, y)

gb_rmse = -gb_search.best_score_
gb_std = gb_search.cv_results_['std_test_score'][gb_search.best_index_]
print(f"\n Gradient Boosting complete")
print(f"  Best params: {gb_search.best_params_}")
print(f"  CV RMSE: {gb_rmse:.4f} ± {gb_std:.4f}")

# Check if optimal is at edge (warning)
best_params_gb = gb_search.best_params_
for param, value in best_params_gb.items():
    param_name = param.split('__')[1]
    grid_values = param_grid_gb[param]

    # Handle None separately (represents unbounded, not an edge)
    if value is None:
        continue  # None is unbounded, not an edge configuration

    # Filter out None from grid for comparison
    numeric_values = [v for v in grid_values if v is not None]

    if numeric_values:  # Only check if there are numeric values
        if value == max(numeric_values):
            print(f"  WARNING: {param_name}={value} is at upper edge!")
        elif value == min(numeric_values):
            print(f"  WARNING: {param_name}={value} is at lower edge!")

# =============================================================================
# SELECT BEST MODEL
# =============================================================================

print("\n" + "=" * 80)
print("MODEL SELECTION")
print("=" * 80)

print(f"\nResults:")
print(f"  Random Forest:     {rf_rmse:.4f} ± {rf_std:.4f}")
print(f"  Gradient Boosting: {gb_rmse:.4f} ± {gb_std:.4f}")

# Select simplest among best (tolerance-based)
tolerance = 0.005
noise = max(rf_std, gb_std)
threshold = max(tolerance, noise)

print(f"\nSelection criteria:")
print(f"  Tolerance: {tolerance:.4f}")
print(f"  Noise level (max std): {noise:.4f}")
print(f"  Threshold: {threshold:.4f}")

if (gb_rmse + threshold) < rf_rmse:
    final_model = gb_search.best_estimator_
    final_name = "Gradient Boosting"
    final_rmse = gb_rmse
    final_std = gb_std
    print(f"\n Selected: {final_name} (significantly better)")
    print(f"  Improvement: {rf_rmse - gb_rmse:.4f} RMSE points")
else:
    final_model = rf_search.best_estimator_
    final_name = "Random Forest"
    final_rmse = rf_rmse
    final_std = rf_std
    print(f"\n Selected: {final_name} (simpler model, comparable performance)")
    print(f"  Difference: {abs(rf_rmse - gb_rmse):.4f} RMSE points (within threshold)")

# =============================================================================
# SAVE FINAL MODEL
# =============================================================================

joblib.dump(final_model, 'final_model.joblib')

print("\n" + "=" * 80)
print("SCRIPT 2 COMPLETE")
print("=" * 80)
print(f"Final model: {final_name}")
print(f"Expected RMSE on new data: {final_rmse:.4f} ± {final_std:.4f}")
print(f"Model saved: final_model.joblib")
print("=" * 80)
"""
Script 4 - Diagnostics
==================================
Extended analysis and visualizations for the report.
NOT part of the production pipeline. Run separately for analysis.
"""

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

print("=" * 80)
print("DIAGNOSTICS FOR REPORT")
print("=" * 80)

# Configuration
USE_SUBSAMPLE_FOR_IMPORTANCE = False  # Set False for final run on full test set
IMPORTANCE_SAMPLE_SIZE = 5000

# =============================================================================
# LOAD DATA
# =============================================================================

print("\nLoading data...")
test = pd.read_pickle('test.pkl')
model = joblib.load('final_model.joblib')

X_test = test.drop(columns=['target', 'primary_key'])
y_test = test['target'].values  # Convert to numpy array for consistent indexing
y_pred = model.predict(X_test)

errors = y_pred - y_test
abs_errors = np.abs(errors)

print(f"Test set: {len(X_test)} observations, {len(X_test.columns)} features")

# =============================================================================
# ERROR STATISTICS
# =============================================================================

print("\n" + "=" * 80)
print("ERROR ANALYSIS")
print("=" * 80)

# Basic statistics
print("\nSummary statistics:")
print(f"  Mean error: {errors.mean():.4f} (bias)")
print(f"  Std error: {errors.std():.4f} (variance)")
print(f"  Median absolute error: {np.median(abs_errors):.4f}")
print(f"  Mean absolute error: {np.mean(abs_errors):.4f}")

# Quantiles
q_low, q_high = np.quantile(errors, [0.025, 0.975])
print(f"  95% error interval: [{q_low:.4f}, {q_high:.4f}]")

# Standard metrics
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\nStandard metrics:")
print(f"  RMSE: {rmse:.4f}")
print(f"  MAE: {mae:.4f}")
print(f"  R²: {r2:.4f}")

# Worst predictions
print("\n" + "-" * 80)
print("Top 10 worst predictions:")
print("-" * 80)
worst_idx = abs_errors.argsort()[-10:][::-1]
print(f"{'Rank':<6}{'True':>8}{'Pred':>8}{'Error':>8}{'Abs Err':>10}")
print("-" * 80)
for i, idx in enumerate(worst_idx, 1):
    print(f"{i:<6}{y_test[idx]:>8.4f}{y_pred[idx]:>8.4f}{errors[idx]:>8.4f}{abs_errors[idx]:>10.4f}")

# ENHANCEMENT 1: Analyze characteristics of worst predictions
print("\n" + "-" * 80)
print("CHARACTERISTICS OF WORST PREDICTIONS")
print("-" * 80)
worst_cases = X_test.iloc[worst_idx]
print(f"Analyzing {len(worst_cases)} worst predictions...")

# Auto-select top numeric features by variance (most informative)
numeric_candidates = X_test.select_dtypes(include=['float64', 'int64']).columns.tolist()
if numeric_candidates:
    numeric_variances = X_test[numeric_candidates].var().sort_values(ascending=False)
    numeric_cols = numeric_variances.head(5).index.tolist()  # Top 5 by variance
else:
    numeric_cols = []

# Auto-select categorical features with reasonable cardinality
categorical_candidates = X_test.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
categorical_cols = []
for col in categorical_candidates:
    n_unique = X_test[col].nunique()
    if 2 <= n_unique <= 50:  # Reasonable cardinality for interpretation
        categorical_cols.append(col)
categorical_cols = categorical_cols[:5]  # Limit to 5 for readability

print(f"\nAuto-selected features for analysis:")
print(f"  Numeric (top 5 by variance): {', '.join(numeric_cols) if numeric_cols else 'None'}")
print(f"  Categorical (cardinality 2-50): {', '.join(categorical_cols) if categorical_cols else 'None'}")

if numeric_cols:
    print("\nNumeric characteristics (mean):")
    print(f"{'Feature':<25}{'Worst cases':>15}{'All test':>15}{'Difference':>15}")
    print("-" * 70)
    for col in numeric_cols:
        mean_worst = worst_cases[col].mean()
        mean_all = X_test[col].mean()
        diff = mean_worst - mean_all
        print(f"{col:<25}{mean_worst:>15.2f}{mean_all:>15.2f}{diff:>15.2f}")

# Key categorical features
categorical_cols = [c for c in categorical_cols if c in worst_cases.columns]
if categorical_cols:
    print("\nCategorical characteristics (most frequent):")
    print(f"{'Feature':<25}{'Most common value':>25}{'Frequency':>15}")
    print("-" * 65)
    for col in categorical_cols:
        if worst_cases[col].notna().any():
            most_common = worst_cases[col].mode()
            if len(most_common) > 0:
                count = (worst_cases[col] == most_common[0]).sum()
                freq = f"{count}/{len(worst_cases)}"
                print(f"{col:<25}{str(most_common[0]):>25}{freq:>15}")

print("\nPattern assessment:")
if len(worst_cases) > 5:
    print("  Sufficient sample size for pattern detection")
    print("  Examine if worst predictions share common characteristics")
else:
    print("  Limited sample for robust pattern detection")

# =============================================================================
# VISUALIZATIONS
# =============================================================================

print("\n" + "=" * 80)
print("GENERATING VISUALIZATIONS")
print("=" * 80)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('Model Performance Diagnostics', fontsize=14, fontweight='bold')

# 1. True vs Predicted
ax = axes[0, 0]
ax.scatter(y_test, y_pred, alpha=0.3, s=10)
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
        'r--', lw=2, label='Perfect prediction')
ax.set_xlabel('True Values')
ax.set_ylabel('Predicted Values')
ax.set_title('True vs Predicted')
ax.legend()
ax.grid(alpha=0.3)

# 2. Residual distribution
ax = axes[0, 1]
ax.hist(errors, bins=50, edgecolor='black', alpha=0.7)
ax.axvline(0, color='red', linestyle='--', label='Zero error')
ax.set_xlabel('Prediction Error')
ax.set_ylabel('Frequency')
ax.set_title('Residual Distribution')
ax.legend()
ax.grid(alpha=0.3)

# 3. Residual plot
ax = axes[1, 0]
ax.scatter(y_pred, errors, alpha=0.3, s=10)
ax.axhline(0, color='red', linestyle='--', lw=2)
ax.set_xlabel('Predicted Values')
ax.set_ylabel('Residuals')
ax.set_title('Residual Plot')
ax.grid(alpha=0.3)

# 4. Distribution comparison
ax = axes[1, 1]
ax.hist(y_test, bins=30, alpha=0.5, label='True', edgecolor='black')
ax.hist(y_pred, bins=30, alpha=0.5, label='Predicted', edgecolor='black')
ax.set_xlabel('Target Value')
ax.set_ylabel('Frequency')
ax.set_title('Distribution Comparison')
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('diagnostics_performance.png', dpi=150, bbox_inches='tight')
print("Saved: diagnostics_performance.png")
plt.close()

# =============================================================================
# ENHANCEMENT 2: DISTRIBUTION MONITORING (Train vs Test)
# =============================================================================

print("\n" + "=" * 80)
print("DISTRIBUTION MONITORING (Train vs Test)")
print("=" * 80)
print("\n IMPORTANT: This is DESCRIPTIVE validation only")
print("   Purpose: Verify test set is representative of training data")
print("   Timing: Performed AFTER model training completed")
print("   Usage: NO model tuning based on test set distributions")
print("   This validates our evaluation methodology, not model selection\n")

try:
    learning = pd.read_pickle('learning.pkl')
    X_train = learning.drop(columns=['target', 'primary_key'])
    y_train = learning['target'].values

    print(f"Training set: {len(X_train)} observations")
    print(f"Test set: {len(X_test)} observations")

    # Compare target distributions
    print("\nTarget distribution comparison:")
    print(f"{'Statistic':<20}{'Train':>15}{'Test':>15}{'Difference':>15}")
    print("-" * 65)
    print(f"{'Mean':<20}{y_train.mean():>15.4f}{y_test.mean():>15.4f}{y_train.mean() - y_test.mean():>15.4f}")
    print(f"{'Std':<20}{y_train.std():>15.4f}{y_test.std():>15.4f}{y_train.std() - y_test.std():>15.4f}")
    print(
        f"{'Median':<20}{np.median(y_train):>15.4f}{np.median(y_test):>15.4f}{np.median(y_train) - np.median(y_test):>15.4f}")

    # Visualize key feature distributions
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle('Feature Distribution Comparison: Train vs Test', fontsize=14, fontweight='bold')

    # Select key features to visualize
    key_features = ['age_2020', 'working_hours', 'Remuneration',
                    'Community_size', 'n_sport_clubs', 'target']

    # Replace 'target' with actual target column for visualization
    for idx, feature in enumerate(key_features[:6]):
        ax = axes[idx // 3, idx % 3]

        if feature == 'target':
            ax.hist(y_train, bins=30, alpha=0.5, label='Train', edgecolor='black', density=True)
            ax.hist(y_test, bins=30, alpha=0.5, label='Test', edgecolor='black', density=True)
            ax.set_title('Target Distribution')
        elif feature in X_train.columns:
            ax.hist(X_train[feature].dropna(), bins=30, alpha=0.5, label='Train',
                    edgecolor='black', density=True)
            ax.hist(X_test[feature].dropna(), bins=30, alpha=0.5, label='Test',
                    edgecolor='black', density=True)
            ax.set_title(f'{feature}')
        else:
            ax.text(0.5, 0.5, f'{feature}\nNot available',
                    ha='center', va='center', transform=ax.transAxes)

        ax.legend()
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('diagnostics_distributions.png', dpi=150, bbox_inches='tight')
    print("\nSaved: diagnostics_distributions.png")
    plt.close()

    print("\n✓ Distribution monitoring complete")
    print("  → Check for distribution shift between train and test")
    print("  → Large differences may indicate data drift")

except FileNotFoundError:
    print(" learning.pkl not found - skipping distribution monitoring")

# =============================================================================
# ENHANCEMENT 3: PERMUTATION FEATURE IMPORTANCE (Multiple Metrics)
# =============================================================================

print("\n" + "=" * 80)
print("PERMUTATION FEATURE IMPORTANCE (Enhanced)")
print("=" * 80)

# Subsample for speed if configured
if USE_SUBSAMPLE_FOR_IMPORTANCE and len(X_test) > IMPORTANCE_SAMPLE_SIZE:
    print(f" Using subsample of {IMPORTANCE_SAMPLE_SIZE} for speed")
    print(f"   Set USE_SUBSAMPLE_FOR_IMPORTANCE=False for final run")
    rng_sample = np.random.RandomState(42)
    sample_idx = rng_sample.choice(len(X_test), IMPORTANCE_SAMPLE_SIZE, replace=False)
    X_importance = X_test.iloc[sample_idx]
    y_importance = y_test[sample_idx]
    y_pred_importance = model.predict(X_importance)
else:
    print(f"Using full test set ({len(X_test)} observations)")
    X_importance = X_test
    y_importance = y_test
    y_pred_importance = model.predict(X_importance)

rng = np.random.RandomState(42)  # Fixed seed for reproducibility

# Baseline metrics
baseline_rmse = np.sqrt(mean_squared_error(y_importance, y_pred_importance))
baseline_mae = mean_absolute_error(y_importance, y_pred_importance)
baseline_r2 = r2_score(y_importance, y_pred_importance)

print(f"Baseline metrics on importance set:")
print(f"  RMSE: {baseline_rmse:.4f}")
print(f"  MAE: {baseline_mae:.4f}")
print(f"  R²: {baseline_r2:.4f}")

# Store multiple metrics
feature_importance_rmse = {}
feature_importance_mae = {}
feature_importance_r2 = {}

print(f"\nComputing importance for {len(X_importance.columns)} features...")
for i, col in enumerate(X_importance.columns, 1):
    X_permuted = X_importance.copy()
    X_permuted[col] = rng.permutation(X_permuted[col].values)

    y_pred_permuted = model.predict(X_permuted)

    # Compute multiple metrics
    rmse_permuted = np.sqrt(mean_squared_error(y_importance, y_pred_permuted))
    mae_permuted = mean_absolute_error(y_importance, y_pred_permuted)
    r2_permuted = r2_score(y_importance, y_pred_permuted)

    # Delta (increase in error = positive importance)
    feature_importance_rmse[col] = rmse_permuted - baseline_rmse
    feature_importance_mae[col] = mae_permuted - baseline_mae
    feature_importance_r2[col] = baseline_r2 - r2_permuted  # Decrease in R² = positive importance

    if i % 10 == 0:
        print(f"  Progress: {i}/{len(X_importance.columns)} features")

# Sort by RMSE importance (primary metric)
feature_importance_rmse = dict(sorted(feature_importance_rmse.items(),
                                      key=lambda x: x[1], reverse=True))
feature_importance_mae = dict(sorted(feature_importance_mae.items(),
                                     key=lambda x: x[1], reverse=True))
feature_importance_r2 = dict(sorted(feature_importance_r2.items(),
                                    key=lambda x: x[1], reverse=True))

# Display top 20 (all metrics)
print("\n" + "-" * 80)
print("TOP 20 MOST IMPORTANT FEATURES (Observational, not causal)")
print("-" * 80)
print(f"{'Rank':<6}{'Feature':<30}{'ΔRMSE':>10}{'ΔMAE':>10}{'ΔR²':>10}")
print("-" * 80)
for i, feat in enumerate(list(feature_importance_rmse.keys())[:20], 1):
    delta_rmse = feature_importance_rmse[feat]
    delta_mae = feature_importance_mae[feat]
    delta_r2 = feature_importance_r2[feat]
    print(f"{i:<6}{feat:<30}{delta_rmse:>10.4f}{delta_mae:>10.4f}{delta_r2:>10.4f}")

print("\nNote: All metrics should rank similar features highly")
print("      ΔRMSE = increase in RMSE when feature permuted")
print("      ΔMAE = increase in MAE when feature permuted")
print("      ΔR² = decrease in R² when feature permuted")

# Visualize top 15 (RMSE as primary)
fig, ax = plt.subplots(figsize=(10, 8))
top_15 = dict(list(feature_importance_rmse.items())[:15])
features = list(top_15.keys())
importances = list(top_15.values())

ax.barh(range(len(features)), importances, color='steelblue', edgecolor='black')
ax.set_yticks(range(len(features)))
ax.set_yticklabels(features)
ax.set_xlabel('Increase in RMSE when feature is permuted')
title = 'Permutation Feature Importance (Top 15)\n(Observational, not causal)'
if USE_SUBSAMPLE_FOR_IMPORTANCE and len(X_test) > IMPORTANCE_SAMPLE_SIZE:
    title += f'\n Computed on subsample (n={IMPORTANCE_SAMPLE_SIZE})'
ax.set_title(title, fontsize=12, fontweight='bold')
ax.grid(alpha=0.3, axis='x')
ax.invert_yaxis()

plt.tight_layout()
plt.savefig('diagnostics_importance.png', dpi=150, bbox_inches='tight')
print("\nSaved: diagnostics_importance.png")
plt.close()

# =============================================================================
# ENHANCEMENT 4: ROBUST CSV VERIFICATION
# =============================================================================

print("\n" + "=" * 80)
print("VERIFYING predictions.csv")
print("=" * 80)

try:
    pred_df = pd.read_csv('predictions.csv')

    # Load prediction data for row count verification
    try:
        prediction_data = pd.read_pickle('prediction.pkl')
        expected_rows = len(prediction_data)
        print(f"Expected rows (from prediction.pkl): {expected_rows}")
    except FileNotFoundError:
        expected_rows = None
        print("  prediction.pkl not found - cannot verify row count")

    print(f"Actual rows in predictions.csv: {len(pred_df)}")

    # Checks (without hard-coded row count)
    checks = []
    all_ok = True

    # Row count (only if we have expected value)
    if expected_rows is not None:
        rows_ok = len(pred_df) == expected_rows
        checks.append(('Row count', len(pred_df), expected_rows, rows_ok))
        all_ok = all_ok and rows_ok

    # Column count
    cols_ok = len(pred_df.columns) == 2
    checks.append(('Column count', len(pred_df.columns), 2, cols_ok))
    all_ok = all_ok and cols_ok

    # Column names
    expected_cols = ['primary_key', 'target']
    cols_match = list(pred_df.columns) == expected_cols
    checks.append(('Column names', list(pred_df.columns), expected_cols, cols_match))
    all_ok = all_ok and cols_match

    # Missing values
    missing_ok = pred_df.isnull().sum().sum() == 0
    checks.append(('Missing values', pred_df.isnull().sum().sum(), 0, missing_ok))
    all_ok = all_ok and missing_ok

    # Duplicate primary keys
    dup_ok = pred_df['primary_key'].duplicated().sum() == 0
    checks.append(('Duplicate keys', pred_df['primary_key'].duplicated().sum(), 0, dup_ok))
    all_ok = all_ok and dup_ok

    # Display results
    print("\nValidation checks:")
    for check_name, actual, expected, ok in checks:
        status = '✓' if ok else '✗'
        if actual == expected:
            print(f"  {status} {check_name}: {actual} (OK)")
        else:
            print(f"  {status} {check_name}: {actual} (expected {expected})")

    # Check target dtype
    is_numeric = pd.api.types.is_numeric_dtype(pred_df['target'])
    all_ok = all_ok and is_numeric
    print(
        f"  {'numeric' if is_numeric else 'not numeric'} target dtype numeric: {pred_df['target'].dtype} {'(OK)' if is_numeric else '(FAILED)'}")

    # Summary
    print(f"\n{'=' * 80}")
    print(f"Format validation: {' PASSED' if all_ok else ' FAILED'}")
    print(f"{'=' * 80}")

except FileNotFoundError:
    print(" predictions.csv not found!")
    all_ok = False

# =============================================================================
# FINAL SUMMARY
# =============================================================================

print("\n" + "=" * 80)
print("DIAGNOSTICS COMPLETE")
print("=" * 80)
print("\nGenerated files:")
print("  diagnostics_performance.png (4-panel error analysis)")
print("  diagnostics_importance.png (top 15 features)")
if 'learning' in locals():
    print("  ✓ diagnostics_distributions.png (train vs test)")
print("\nAll analysis follows best practices:")
print("  Full error analysis (not just average)")
print("  Study of worst predictions with characteristics")
print("  Multiple importance metrics (RMSE, MAE, R²)")
print("  Distribution monitoring (train vs test)")
print("  Explicit non-causal warning")
print("  Reproducible (fixed random seed)")
print("=" * 80)
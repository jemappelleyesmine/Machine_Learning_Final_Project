"""
Script 4 - Diagnostics (OPTIONAL - for report only)
====================================================
Extended analysis and visualizations for the report.
NOT part of the production pipeline. Run separately for analysis.
"""

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

print("=" * 80)
print("DIAGNOSTICS FOR REPORT")
print("=" * 80)

# Configuration
USE_SUBSAMPLE_FOR_IMPORTANCE = False  # Set False for final run on full test set
IMPORTANCE_SAMPLE_SIZE = 5000

# =============================================================================
# LOAD DATA
# =============================================================================

test = pd.read_pickle('test.pkl')
model = joblib.load('final_model.joblib')

X_test = test.drop(columns=['target', 'primary_key'])
y_test = test['target'].values  # Convert to numpy array for consistent indexing
y_pred = model.predict(X_test)

errors = y_pred - y_test
abs_errors = np.abs(errors)

# =============================================================================
# ERROR STATISTICS
# =============================================================================

print("\nError Analysis:")
print(f"  Mean error: {errors.mean():.4f}")
print(f"  Std error: {errors.std():.4f}")
print(f"  Median absolute error: {np.median(abs_errors):.4f}")

q_low, q_high = np.quantile(errors, [0.025, 0.975])
print(f"  95% error interval: [{q_low:.4f}, {q_high:.4f}]")

# Worst predictions
print("\nTop 10 worst predictions:")
worst_idx = abs_errors.argsort()[-10:][::-1]
for i, idx in enumerate(worst_idx, 1):
    print(f"  {i:2}. True: {y_test[idx]:.4f}, Pred: {y_pred[idx]:.4f}, Error: {errors[idx]:.4f}")

# =============================================================================
# VISUALIZATIONS
# =============================================================================

print("\nGenerating visualizations...")

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
# PERMUTATION FEATURE IMPORTANCE (reproducible, with subsample option)
# =============================================================================

print("\nComputing permutation feature importance...")

# Subsample for speed if configured
if USE_SUBSAMPLE_FOR_IMPORTANCE and len(X_test) > IMPORTANCE_SAMPLE_SIZE:
    print(f"⚠️  Using subsample of {IMPORTANCE_SAMPLE_SIZE} for speed")
    print(f"   Set USE_SUBSAMPLE_FOR_IMPORTANCE=False for final run")
    rng_sample = np.random.RandomState(42)
    sample_idx = rng_sample.choice(len(X_test), IMPORTANCE_SAMPLE_SIZE, replace=False)
    X_importance = X_test.iloc[sample_idx]
    y_importance = y_test[sample_idx]  # numpy array indexing
    y_pred_importance = model.predict(X_importance)
else:
    print(f"   Using full test set ({len(X_test)} rows)")
    X_importance = X_test
    y_importance = y_test
    y_pred_importance = model.predict(X_importance)

rng = np.random.RandomState(42)  # Fixed seed for reproducibility
baseline_rmse = np.sqrt(mean_squared_error(y_importance, y_pred_importance))
feature_importance = {}

for i, col in enumerate(X_importance.columns, 1):
    X_permuted = X_importance.copy()
    X_permuted[col] = rng.permutation(X_permuted[col].values)

    y_pred_permuted = model.predict(X_permuted)
    rmse_permuted = np.sqrt(mean_squared_error(y_importance, y_pred_permuted))

    delta_rmse = rmse_permuted - baseline_rmse
    feature_importance[col] = delta_rmse

    if i % 10 == 0:
        print(f"   Progress: {i}/{len(X_importance.columns)} features")

# Sort by importance
feature_importance = dict(sorted(feature_importance.items(),
                                 key=lambda x: x[1], reverse=True))

# Display top 20
print("\nTop 20 most important features (observational, not causal):")
for i, (feat, delta) in enumerate(list(feature_importance.items())[:20], 1):
    print(f"  {i:2}. {feat:<30} ΔRMSE: {delta:>7.4f}")

# Visualize top 15
fig, ax = plt.subplots(figsize=(10, 8))
top_15 = dict(list(feature_importance.items())[:15])
features = list(top_15.keys())
importances = list(top_15.values())

ax.barh(range(len(features)), importances, color='steelblue', edgecolor='black')
ax.set_yticks(range(len(features)))
ax.set_yticklabels(features)
ax.set_xlabel('Increase in RMSE when feature is permuted')
title = 'Permutation Feature Importance (Top 15)\n(Observational, not causal)'
if USE_SUBSAMPLE_FOR_IMPORTANCE and len(X_test) > IMPORTANCE_SAMPLE_SIZE:
    title += f'\n⚠️ Computed on subsample (n={IMPORTANCE_SAMPLE_SIZE})'
ax.set_title(title, fontsize=12, fontweight='bold')
ax.grid(alpha=0.3, axis='x')
ax.invert_yaxis()

plt.tight_layout()
plt.savefig('diagnostics_importance.png', dpi=150, bbox_inches='tight')
print("Saved: diagnostics_importance.png")
plt.close()

# =============================================================================
# VERIFY predictions.csv
# =============================================================================

print("\nVerifying predictions.csv...")
pred_df = pd.read_csv('predictions.csv')

checks = [
    ('Rows', len(pred_df), 50044),
    ('Columns', len(pred_df.columns), 2),
    ('Column names', list(pred_df.columns), ['primary_key', 'target']),
    ('Missing values', pred_df.isnull().sum().sum(), 0),
    ('Duplicates', pred_df['primary_key'].duplicated().sum(), 0)
]

all_ok = True
for check_name, actual, expected in checks:
    ok = actual == expected
    all_ok = all_ok and ok
    status = '✓' if ok else '✗'
    print(f"  {status} {check_name}: {actual} {'(OK)' if ok else f'(expected {expected})'}")

# Check that target column is numeric (what matters for grader)
is_numeric = pd.api.types.is_numeric_dtype(pred_df['target'])
all_ok = all_ok and is_numeric
print(
    f"  {'✓' if is_numeric else '✗'} target dtype numeric: {pred_df['target'].dtype} {'(OK)' if is_numeric else '(FAILED)'}")

print(f"\nFormat validation: {'✓ PASSED' if all_ok else '✗ FAILED'}")
print("\n" + "=" * 80)
print("DIAGNOSTICS COMPLETE")
print("=" * 80)
"""
Script 3 - Result Production
======================================
Load test, predict, compute metrics, generate predictions.csv.
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load test set and model
test = pd.read_pickle('test.pkl')
model = joblib.load('final_model.joblib')

# Predict on test set
X_test = test.drop(columns=['target', 'primary_key'])
y_test = test['target']
y_pred = model.predict(X_test)

# Compute metrics (use sqrt manually for compatibility)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Test RMSE: {rmse:.4f}")
print(f"Test MAE:  {mae:.4f}")
print(f"Test R²:   {r2:.4f}")

# Generate predictions for unlabeled data
prediction_data = pd.read_pickle('prediction.pkl')
X_prediction = prediction_data.drop(columns=['primary_key'])
y_prediction = model.predict(X_prediction)

# Save predictions.csv in required format
pd.DataFrame({
    'primary_key': prediction_data['primary_key'],
    'target': y_prediction
}).to_csv('predictions.csv', index=False, sep=',', decimal='.')

# Sanity check
print(f"Predictions saved: {len(y_prediction)} rows")
"""
Script 2 - Model Building (All-Star Battle)
=======================================================================
1. Baseline: Ridge Regression (Linear)
2. Reference: Random Forest (Bagging)
3. Challenger: Gradient Boosting (Boosting)
4. Explorer: PyTorch Neural Net (Entity Embeddings)
"""

import pandas as pd
import numpy as np
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
import warnings
import time

warnings.filterwarnings('ignore')

print("=" * 80)
print("SCRIPT 2 - MODEL BUILDING: THE ULTIMATE COMPARISON")
print("=" * 80)

# Configuration
RANDOM_STATE = 42
DEBUG = False   # Set True to use subsample
DEBUG_SIZE = 5000
BATCH_SIZE = 256
EPOCHS = 20    
LR = 0.001     

# Device configuration
if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
    print(">>> Using CUDA (NVIDIA GPU)")
elif torch.backends.mps.is_available():
    DEVICE = torch.device('mps')
    print(">>> Using MPS (Mac M1/M2 GPU)")
else:
    DEVICE = torch.device('cpu')
    print(">>> Using CPU")

# =============================================================================
# 1. LOAD DATA
# =============================================================================
print("\nLoading learning set...")
learning = pd.read_pickle('learning.pkl')

X = learning.drop(columns=['target', 'primary_key'])
y = learning['target']

if DEBUG:
    X = X.sample(n=DEBUG_SIZE, random_state=RANDOM_STATE)
    y = y.loc[X.index]
    print(f" DEBUG MODE: Using subsample {X.shape}")

# Identify features
numeric_features = X.select_dtypes(include=['float64', 'int64']).columns.tolist()
categorical_features = X.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

# Handle numeric-coded categoricals
numeric_coded_categoricals = [
    'Employee_count', 'Employee_count_retired', 'EMPLOYER_TYPE', 'EMPLOYER_TYPE_retired',
    'JOB_CATEGORY', 'JOB_CATEGORY_retired', 'city_type'
]
for col in numeric_coded_categoricals:
    if col in numeric_features:
        numeric_features.remove(col)
        if col in X.columns:
            categorical_features.append(col)

# Split Data (Train/Validation)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

# =============================================================================
# 2. TRADITIONAL ML PIPELINE
# =============================================================================
print("\n" + "-"*40)
print("PART A: SKLEARN MODELS (Ridge, RF, GBM)")
print("-"*40)

# --- Preprocessing for ML ---
preprocessor_ml = ColumnTransformer(
    transformers=[
        ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), numeric_features),
        ('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore'))]), categorical_features)
    ]
)

# --- Model 1: Ridge (Baseline) ---
print("\n1. Training Ridge (Linear Baseline)...")
ridge_model = Pipeline([('preprocessor', preprocessor_ml), ('regressor', Ridge())])
ridge_model.fit(X_train, y_train)
rmse_ridge = np.sqrt(mean_squared_error(y_val, ridge_model.predict(X_val)))
print(f"   Ridge RMSE: {rmse_ridge:.4f}")

# --- Model 2: Random Forest (Reference) ---
print("\n2. Training Random Forest...")
rf_model = Pipeline([
    ('preprocessor', preprocessor_ml),
    ('regressor', RandomForestRegressor(n_estimators=100, max_depth=20, n_jobs=-1, random_state=RANDOM_STATE))
])
rf_model.fit(X_train, y_train)
rmse_rf = np.sqrt(mean_squared_error(y_val, rf_model.predict(X_val)))
print(f"   Random Forest RMSE: {rmse_rf:.4f}")

# --- Model 3: Gradient Boosting (The Heavy Hitter) ---
print("\n3. Training Gradient Boosting...")
gb_model = Pipeline([
    ('preprocessor', preprocessor_ml), 
    ('regressor', GradientBoostingRegressor(
        n_estimators=2000,         
        learning_rate=0.04,        
        max_depth=8,               
        min_samples_split=100,     
        subsample=0.8,             
        random_state=RANDOM_STATE,
        verbose=1                  
    ))
])
gb_model.fit(X_train, y_train)
rmse_gb = np.sqrt(mean_squared_error(y_val, gb_model.predict(X_val)))
print(f"   Gradient Boosting RMSE: {rmse_gb:.4f}")

# =============================================================================
# 3. PYTORCH PIPELINE (Entity Embeddings)
# =============================================================================
print("\n" + "-"*40)
print("PART B: PYTORCH DEEP LEARNING")
print("-"*40)

# Create copies for PyTorch processing
X_train_torch = X_train.copy()
X_val_torch = X_val.copy()

cat_dims = []
encoders = {}

print("Preprocessing for Neural Net...")

# A. Categorical Encoding (Ordinal for Embeddings)
for col in categorical_features:
    X_train_torch[col] = X_train_torch[col].astype(str).fillna("MISSING")
    X_val_torch[col] = X_val_torch[col].astype(str).fillna("MISSING")
    
    le = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    le.fit(X_train_torch[[col]])
    
    X_train_torch[col] = le.transform(X_train_torch[[col]]).astype(int) + 1
    X_val_torch[col] = le.transform(X_val_torch[[col]]).astype(int) + 1
    
    num_unique = int(max(X_train_torch[col].max(), X_val_torch[col].max()) + 1)
    emb_dim = min(50, (num_unique + 1) // 2)
    cat_dims.append((num_unique, emb_dim))
    encoders[col] = le

# B. Numeric Scaling
scaler = StandardScaler()
X_train_torch[numeric_features] = scaler.fit_transform(X_train_torch[numeric_features].fillna(0))
X_val_torch[numeric_features] = scaler.transform(X_val_torch[numeric_features].fillna(0))

# Convert to Tensors
X_train_cat = torch.tensor(X_train_torch[categorical_features].values, dtype=torch.long).to(DEVICE)
X_train_num = torch.tensor(X_train_torch[numeric_features].values, dtype=torch.float32).to(DEVICE)
y_train_t = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1).to(DEVICE)

X_val_cat = torch.tensor(X_val_torch[categorical_features].values, dtype=torch.long).to(DEVICE)
X_val_num = torch.tensor(X_val_torch[numeric_features].values, dtype=torch.float32).to(DEVICE)
y_val_t = torch.tensor(y_val.values, dtype=torch.float32).unsqueeze(1).to(DEVICE)

# PyTorch Dataset
class TabularDataset(Dataset):
    def __init__(self, x_cat, x_num, y):
        self.x_cat = x_cat
        self.x_num = x_num
        self.y = y
    def __len__(self): return len(self.y)
    def __getitem__(self, idx): return self.x_cat[idx], self.x_num[idx], self.y[idx]

train_dl = DataLoader(TabularDataset(X_train_cat, X_train_num, y_train_t), batch_size=BATCH_SIZE, shuffle=True)

# Define Model
class EntityEmbeddingNet(nn.Module):
    def __init__(self, embedding_dims, num_numerical, hidden_layers=[200, 100], dropout_p=0.2):
        super().__init__()
        self.embeddings = nn.ModuleList([nn.Embedding(n, d) for n, d in embedding_dims])
        self.total_emb_dim = sum([d for _, d in embedding_dims])
        
        layers = []
        in_dim = self.total_emb_dim + num_numerical
        for h_dim in hidden_layers:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_p))
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, 1))
        self.mlp = nn.Sequential(*layers)
        
    def forward(self, x_cat, x_num):
        embedded = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]
        cat_out = torch.cat(embedded, dim=1)
        return self.mlp(torch.cat([cat_out, x_num], dim=1))

model = EntityEmbeddingNet(cat_dims, len(numeric_features)).to(DEVICE)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# Training Loop
print("\nStarting PyTorch Training...")
start_time = time.time()
best_val_rmse = float('inf')

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0
    for c_batch, n_batch, y_batch in train_dl:
        optimizer.zero_grad()
        loss = criterion(model(c_batch, n_batch), y_batch)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * c_batch.size(0)
    
    # Validation
    model.eval()
    with torch.no_grad():
        val_pred = model(X_val_cat, X_val_num)
        val_rmse = torch.sqrt(criterion(val_pred, y_val_t)).item()
        
    if val_rmse < best_val_rmse:
        best_val_rmse = val_rmse
        # Optional: save best state_dict here
    
    if (epoch+1) % 5 == 0 or epoch == 0:
        print(f" Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss/len(train_dl.dataset):.2f} | Val RMSE: {val_rmse:.4f}")

print(f"Training Time: {time.time() - start_time:.1f}s")
rmse_nn = best_val_rmse

# =============================================================================
# 4. FINAL SELECTION
# =============================================================================

print("\n" + "=" * 80)
print("FINAL SCOREBOARD")
print("=" * 80)

results = {
    'Ridge (Baseline)': rmse_ridge,
    'Random Forest': rmse_rf,
    'Gradient Boosting': rmse_gb,
    'PyTorch (Embeddings)': rmse_nn
}

# Print sorted results
for name, score in sorted(results.items(), key=lambda x: x[1]):
    print(f"  {name: <20}: RMSE = {score:.4f}")

best_model_name = min(results, key=results.get)
print(f"\n🏆 WINNER: {best_model_name}")

# Save Best Model
print("\nSaving best model...")
if 'PyTorch' in best_model_name:
    torch.save(model.state_dict(), 'final_model.pth')
    print("Saved PyTorch model to final_model.pth (Note: Requires custom load code)")
else:
    if 'Gradient' in best_model_name:
        joblib.dump(gb_model, 'final_model.joblib')
    elif 'Forest' in best_model_name:
        joblib.dump(rf_model, 'final_model.joblib')
    else:
        joblib.dump(ridge_model, 'final_model.joblib')
    print(f"Saved sklearn model to final_model.joblib")

print("=" * 80)
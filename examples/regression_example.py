"""
Basic Regression Example with DKGP
"""
import numpy as np
from dkgp import fit_dkgp, predict

# Generate synthetic data
np.random.seed(42)
X_train = np.random.randn(200, 10)
y_train = np.sum(X_train[:, :3], axis=1) + 0.1 * np.random.randn(200)

X_test = np.random.randn(50, 10)
y_test = np.sum(X_test[:, :3], axis=1) + 0.1 * np.random.randn(50)

print("="*60)
print("DKGP Regression Example")
print("="*60)

# Train with default feature extractor (FCBN)
print("\n1. Training with default extractor...")
mll, gp, dkl, losses = fit_dkgp(
    X_train, y_train,
    feature_dim=16,
    num_epochs=500,
    verbose=True
)

# Predict
print("\n2. Making predictions...")
mean, std = predict(dkl, X_test, return_std=True)

# Evaluate
from dkgp.utils import compute_metrics
metrics = compute_metrics(y_test, mean, std)

print("\n3. Results:")
print(f"  MSE:  {metrics['mse']:.4f}")
print(f"  RMSE: {metrics['rmse']:.4f}")
print(f"  MAE:  {metrics['mae']:.4f}")
print(f"  R²:   {metrics['r2']:.4f}")

# Try different feature extractors
print("\n4. Comparing feature extractors...")
extractors = ['fc', 'fcbn', 'resnet']
results = {}

for ext_type in extractors:
    print(f"\n   Testing {ext_type}...")
    mll, gp, dkl, losses = fit_dkgp(
        X_train, y_train,
        extractor_type=ext_type,
        feature_dim=16,
        num_epochs=500,
        verbose=False
    )
    
    mean, std = predict(dkl, X_test, return_std=True)
    metrics = compute_metrics(y_test, mean, std)
    results[ext_type] = metrics['r2']
    print(f"   {ext_type}: R² = {metrics['r2']:.4f}")

# Best extractor
best = max(results.items(), key=lambda x: x[1])
print(f"\n✓ Best extractor: {best[0]} (R² = {best[1]:.4f})")
print("="*60)

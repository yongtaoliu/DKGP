"""
Feature Extractor Comparison Example
"""
import numpy as np
import time
from dkgp import fit_dkgp, predict
from dkgp.utils import compute_metrics

# Generate synthetic data
np.random.seed(42)
X_train = np.random.randn(200, 50)
y_train = np.sum(X_train[:, :5], axis=1) + 0.1 * np.random.randn(200)

X_test = np.random.randn(50, 50)
y_test = np.sum(X_test[:, :5], axis=1) + 0.1 * np.random.randn(50)

print("="*70)
print("Feature Extractor Comparison")
print("="*70)
print(f"Training samples: {len(X_train)}")
print(f"Input dimension: {X_train.shape[1]}")
print(f"Test samples: {len(X_test)}")
print("="*70)

# Define extractors to test
extractors = [
    {'type': 'fc', 'name': 'Simple FC', 'kwargs': {}},
    {'type': 'fcbn', 'name': 'FC + BatchNorm', 'kwargs': {}},
    {'type': 'resnet', 'name': 'ResNet', 'kwargs': {'hidden_dim': 128, 'num_blocks': 2}},
    {'type': 'attention', 'name': 'Attention', 'kwargs': {'hidden_dim': 128, 'num_heads': 4}},
    {'type': 'wide_deep', 'name': 'Wide & Deep', 'kwargs': {'deep_dims': [128, 64]}},
]

results = []

for config in extractors:
    print(f"\n{'='*70}")
    print(f"Testing: {config['name']}")
    print('='*70)
    
    start_time = time.time()
    
    try:
        mll, gp, dkl, losses = fit_dkgp(
            X_train, y_train,
            feature_dim=16,
            extractor_type=config['type'],
            extractor_kwargs=config['kwargs'],
            num_epochs=500,
            verbose=False
        )
        
        train_time = time.time() - start_time
        
        # Predict
        mean, std = predict(dkl, X_test, return_std=True)
        
        # Evaluate
        metrics = compute_metrics(y_test, mean, std)
        
        results.append({
            'name': config['name'],
            'type': config['type'],
            'mse': metrics['mse'],
            'rmse': metrics['rmse'],
            'mae': metrics['mae'],
            'r2': metrics['r2'],
            'nll': metrics.get('nll', np.nan),
            'train_time': train_time,
            'final_loss': losses[-1]
        })
        
        print(f"✓ Training completed in {train_time:.2f}s")
        print(f"  MSE:  {metrics['mse']:.4f}")
        print(f"  RMSE: {metrics['rmse']:.4f}")
        print(f"  MAE:  {metrics['mae']:.4f}")
        print(f"  R²:   {metrics['r2']:.4f}")
        
    except Exception as e:
        print(f"✗ Failed: {e}")
        results.append({
            'name': config['name'],
            'type': config['type'],
            'error': str(e)
        })

# Summary table
print("\n" + "="*70)
print("SUMMARY TABLE")
print("="*70)
print(f"{'Extractor':<20} {'MSE':<10} {'R²':<10} {'Time (s)':<12}")
print("-"*70)

valid_results = [r for r in results if 'error' not in r]
for res in sorted(valid_results, key=lambda x: x['r2'], reverse=True):
    print(f"{res['name']:<20} {res['mse']:<10.4f} {res['r2']:<10.4f} {res['train_time']:<12.2f}")

print("="*70)

# Best extractor
if valid_results:
    best = max(valid_results, key=lambda x: x['r2'])
    print(f"\n✓ Best extractor: {best['name']} (R² = {best['r2']:.4f})")
    
    fastest = min(valid_results, key=lambda x: x['train_time'])
    print(f"✓ Fastest extractor: {fastest['name']} ({fastest['train_time']:.2f}s)")

print("="*70)

# Recommendations
print("\nRecommendations:")
print("  • For speed: Use 'fc' (simple FC)")
print("  • For accuracy: Use 'fcbn' (FC + BatchNorm)")
print("  • For deep networks: Use 'resnet' (skip connections)")
print("  • For feature interactions: Use 'attention'")
print("  • For mixed features: Use 'wide_deep'")
print("="*70)

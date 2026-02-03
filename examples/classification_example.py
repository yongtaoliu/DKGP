"""
Basic Classification Example with DKGP
"""
import numpy as np
from dkgp import fit_dkgp_classifier, predict_classifier
from sklearn.metrics import accuracy_score, classification_report

# Generate synthetic data
np.random.seed(42)
X_train = np.random.randn(300, 20)
y_train = np.random.randint(0, 3, 300)  # 3 classes

X_test = np.random.randn(100, 20)
y_test = np.random.randint(0, 3, 100)

print("="*60)
print("DKGP Classification Example")
print("="*60)

# Train classifier
print("\n1. Training classifier...")
model, losses = fit_dkgp_classifier(
    X_train, y_train,
    num_classes=3,
    feature_dim=16,
    num_epochs=500,
    verbose=True
)

# Predict
print("\n2. Making predictions...")
y_pred = predict_classifier(model, X_test)
y_proba = predict_classifier(model, X_test, return_proba=True)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
print(f"\n3. Results:")
print(f"  Accuracy: {accuracy:.2%}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Example with confidence weights
print("\n4. Training with confidence weights...")
# Simulate noisy labels
noisy_indices = np.random.choice(300, 60, replace=False)
y_train_noisy = y_train.copy()
y_train_noisy[noisy_indices] = np.random.randint(0, 3, 60)

# Assign lower confidence to noisy samples
confidence_weights = np.ones(300)
confidence_weights[noisy_indices] = 0.3

model_weighted, losses = fit_dkgp_classifier(
    X_train, y_train_noisy,
    num_classes=3,
    confidence_weights=confidence_weights,
    feature_dim=16,
    num_epochs=500,
    verbose=False
)

y_pred_weighted = predict_classifier(model_weighted, X_test)
accuracy_weighted = accuracy_score(y_test, y_pred_weighted)

print(f"\n5. Comparison:")
print(f"  Without confidence weights: {accuracy:.2%}")
print(f"  With confidence weights:    {accuracy_weighted:.2%}")
print(f"  Improvement: {(accuracy_weighted - accuracy)*100:.1f}%")
print("="*60)

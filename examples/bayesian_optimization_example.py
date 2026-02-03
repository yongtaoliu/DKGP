"""
Bayesian Optimization Example with DKGP
"""
import numpy as np
from dkgp import fit_dkgp, predict
from dkgp.acquisition import (
    expected_improvement,
    upper_confidence_bound,
    probability_of_improvement
)

# Define a test function to optimize
def objective_function(x):
    """Simple 2D test function with global maximum"""
    return -(x[0]**2 + x[1]**2) + 10 * np.exp(-((x[0]-2)**2 + (x[1]-2)**2))

print("="*60)
print("Bayesian Optimization Example")
print("="*60)

# Initial random samples
np.random.seed(42)
bounds = np.array([[-5, 5], [-5, 5]])
n_initial = 10

X_train = np.random.uniform(bounds[:, 0], bounds[:, 1], size=(n_initial, 2))
y_train = np.array([objective_function(x) for x in X_train])

print(f"\nInitial samples: {n_initial}")
print(f"Initial best: {y_train.max():.4f} at {X_train[y_train.argmax()]}")

# Bayesian Optimization loop
n_iterations = 10
for iteration in range(n_iterations):
    print(f"\n--- Iteration {iteration+1}/{n_iterations} ---")
    
    # Train GP model
    mll, gp, dkl, losses = fit_dkgp(
        X_train, y_train,
        feature_dim=8,
        num_epochs=300,
        verbose=False
    )
    
    # Generate candidate points
    n_candidates = 1000
    candidates = np.random.uniform(
        bounds[:, 0], 
        bounds[:, 1], 
        size=(n_candidates, 2)
    )
    
    # Compute acquisition functions
    best_f = y_train.max()
    
    ei = expected_improvement(dkl, candidates, best_f, maximize=True)
    ucb = upper_confidence_bound(dkl, candidates, beta=2.0, maximize=True)
    pi = probability_of_improvement(dkl, candidates, best_f, maximize=True)
    
    # Select next point using EI
    next_idx = np.argmax(ei)
    next_x = candidates[next_idx]
    next_y = objective_function(next_x)
    
    # Update dataset
    X_train = np.vstack([X_train, next_x])
    y_train = np.append(y_train, next_y)
    
    print(f"  Next point: {next_x}")
    print(f"  Value: {next_y:.4f}")
    print(f"  Current best: {y_train.max():.4f}")
    print(f"  EI: {ei[next_idx]:.4f}, UCB: {ucb[next_idx]:.4f}, PI: {pi[next_idx]:.4f}")

print("\n" + "="*60)
print("Optimization Complete!")
print("="*60)
print(f"Final best value: {y_train.max():.4f}")
print(f"Best location: {X_train[y_train.argmax()]}")
print(f"True optimum: ~10.0 at [2, 2]")
print(f"Samples used: {len(y_train)}")
print("="*60)

# Visualize results (optional)
try:
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: Convergence
    axes[0].plot(np.maximum.accumulate(y_train), 'o-', linewidth=2)
    axes[0].axhline(y=10, color='r', linestyle='--', label='True optimum')
    axes[0].set_xlabel('Iteration', fontsize=12)
    axes[0].set_ylabel('Best Value Found', fontsize=12)
    axes[0].set_title('Convergence Plot', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Sampled points
    axes[1].scatter(X_train[:n_initial, 0], X_train[:n_initial, 1], 
                   c='blue', label='Initial', s=100, alpha=0.6)
    axes[1].scatter(X_train[n_initial:, 0], X_train[n_initial:, 1], 
                   c='red', label='BO samples', s=100, alpha=0.6)
    axes[1].scatter(2, 2, c='green', marker='*', s=300, label='True optimum')
    axes[1].set_xlabel('x₁', fontsize=12)
    axes[1].set_ylabel('x₂', fontsize=12)
    axes[1].set_title('Sampled Points', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('bayesian_optimization_results.png', dpi=150, bbox_inches='tight')
    print("\n✓ Results saved to 'bayesian_optimization_results.png'")
    
except ImportError:
    print("\n(Install matplotlib to visualize results)")

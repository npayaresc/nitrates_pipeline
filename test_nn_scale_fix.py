#!/usr/bin/env python3
"""Test neural network with corrected ppm scale."""

import sys
sys.path.insert(0, '/home/payanico/pipeline_nitrates')

import torch
import numpy as np
from src.models.neural_network import NitrateNN, NeuralNetworkRegressor

# Test forward pass with ppm-scale values
print("=" * 60)
print("Testing Neural Network with PPM Scale")
print("=" * 60)

# Create test input
batch_size = 10
input_dim = 72
X = torch.randn(batch_size, input_dim)

print(f"\nTest input shape: {X.shape}")

# Create network
nn = NitrateNN(input_dim=input_dim, dropout_rate=0.3)
nn.eval()

# Forward pass
output = nn(X)
print(f"\nOutput shape: {output.shape}")
print(f"Output range: [{output.min().item():.2f}, {output.max().item():.2f}] ppm")
print(f"Expected range: [0, 45000] ppm")

# Test with realistic training
print("\n" + "=" * 60)
print("Testing Training with Realistic PPM Values")
print("=" * 60)

# Create synthetic data in ppm range (similar to real data)
np.random.seed(42)
n_samples = 100
n_features = 72
X_train = np.random.randn(n_samples, n_features)
# Generate targets in realistic ppm range (131-36741)
y_train = np.random.uniform(5000, 25000, n_samples)

print(f"\nTraining data:")
print(f"  X shape: {X_train.shape}")
print(f"  y range: [{y_train.min():.0f}, {y_train.max():.0f}] ppm")
print(f"  y mean: {y_train.mean():.0f} ppm")

# Train small network
regressor = NeuralNetworkRegressor(
    model_type='full',
    epochs=50,
    batch_size=32,
    learning_rate=0.001,
    verbose=True
)

print("\nTraining neural network...")
regressor.fit(X_train, y_train)

# Make predictions
predictions = regressor.predict(X_train[:10])
print(f"\nPredictions:")
print(f"  Shape: {predictions.shape}")
print(f"  Range: [{predictions.min():.0f}, {predictions.max():.0f}] ppm")
print(f"  Sample predictions: {predictions[:5]}")
print(f"  Actual targets: {y_train[:5]}")

print("\n" + "=" * 60)
print("✓ Scale test completed successfully!")
print("=" * 60)

#!/usr/bin/env python3
"""Quick test script to validate adaptive neural network architecture."""

import sys
sys.path.insert(0, '/home/payanico/nitrates_pipeline')

import torch
import numpy as np
from src.models.neural_network import (
    ArchitectureSelector,
    NitrateNN,
    LightNitrateNN,
    AutoGluonOptimizedNN,
    NeuralNetworkRegressor
)

def test_architecture_selector():
    """Test the architecture selector for different feature counts."""
    print("=" * 60)
    print("Testing ArchitectureSelector")
    print("=" * 60)

    test_dims = [20, 50, 72, 150, 300, 495]

    for dim in test_dims:
        print(f"\n--- Testing with {dim} features ---")
        arch_type = ArchitectureSelector.select_architecture_type(dim)
        layer_sizes = ArchitectureSelector.compute_layer_sizes(dim, arch_type)
        dropout_schedule = ArchitectureSelector.get_dropout_schedule(arch_type, 0.3)

        print(f"Architecture type: {arch_type}")
        print(f"Layer sizes: {layer_sizes}")
        print(f"Dropout schedule: {dropout_schedule}")

def test_network_creation():
    """Test creating networks with different input dimensions."""
    print("\n" + "=" * 60)
    print("Testing Network Creation")
    print("=" * 60)

    test_dims = [20, 72, 150, 495]

    for dim in test_dims:
        print(f"\n--- Creating networks for {dim} features ---")

        # Test NitrateNN
        print("\n1. NitrateNN:")
        nn_full = NitrateNN(input_dim=dim, dropout_rate=0.3)

        # Test LightNitrateNN
        print("\n2. LightNitrateNN:")
        nn_light = LightNitrateNN(input_dim=dim, dropout_rate=0.3)

        # Test AutoGluonOptimizedNN
        print("\n3. AutoGluonOptimizedNN:")
        nn_autogluon = AutoGluonOptimizedNN(input_dim=dim, dropout_rate=0.237)

def test_forward_pass():
    """Test forward passes through the networks."""
    print("\n" + "=" * 60)
    print("Testing Forward Pass")
    print("=" * 60)

    # Create test input
    batch_size = 10
    input_dim = 72
    X = torch.randn(batch_size, input_dim)

    print(f"\nTest input shape: {X.shape}")

    # Test NitrateNN
    nn_full = NitrateNN(input_dim=input_dim, dropout_rate=0.3)
    nn_full.eval()
    output = nn_full(X)
    print(f"\nNitrateNN output shape: {output.shape}")
    print(f"Output range: [{output.min().item():.4f}, {output.max().item():.4f}]")

    # Test LightNitrateNN
    nn_light = LightNitrateNN(input_dim=input_dim, dropout_rate=0.3)
    nn_light.eval()
    output = nn_light(X)
    print(f"\nLightNitrateNN output shape: {output.shape}")
    print(f"Output range: [{output.min().item():.4f}, {output.max().item():.4f}]")

def test_regressor_wrapper():
    """Test the NeuralNetworkRegressor wrapper."""
    print("\n" + "=" * 60)
    print("Testing NeuralNetworkRegressor Wrapper")
    print("=" * 60)

    # Create synthetic data
    np.random.seed(42)
    n_samples = 100
    n_features = 72
    X = np.random.randn(n_samples, n_features)
    y = np.random.rand(n_samples) * 0.5  # Nitrate range 0-0.5%

    print(f"\nTraining data shape: X={X.shape}, y={y.shape}")

    # Test with full model
    print("\nTesting NeuralNetworkRegressor with model_type='full'")
    regressor = NeuralNetworkRegressor(
        model_type='full',
        epochs=10,
        batch_size=32,
        verbose=False
    )

    # Fit the model
    regressor.fit(X, y)

    # Make predictions
    predictions = regressor.predict(X[:10])
    print(f"Predictions shape: {predictions.shape}")
    print(f"Sample predictions: {predictions[:5]}")

    print("\n✓ All tests passed!")

if __name__ == "__main__":
    try:
        test_architecture_selector()
        test_network_creation()
        test_forward_pass()
        test_regressor_wrapper()

        print("\n" + "=" * 60)
        print("SUCCESS: All adaptive neural network tests passed!")
        print("=" * 60)

    except Exception as e:
        print("\n" + "=" * 60)
        print(f"ERROR: Test failed with exception:")
        print(f"{type(e).__name__}: {e}")
        print("=" * 60)
        import traceback
        traceback.print_exc()
        sys.exit(1)

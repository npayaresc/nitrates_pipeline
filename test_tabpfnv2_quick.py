#!/usr/bin/env python3
"""Quick test to check if TabPFNv2 can initialize without config errors."""

import pandas as pd
import numpy as np

print("Testing TabPFNv2 initialization...")

try:
    from autogluon.tabular.models.tabpfnv2.tabpfnv2_model import TabPFNV2Model
    print("[OK] TabPFNv2Model imported successfully")

    # Create minimal test data
    np.random.seed(42)
    n_samples = 50
    X = pd.DataFrame({
        'feature1': np.random.randn(n_samples),
        'feature2': np.random.randn(n_samples),
        'feature3': np.random.randn(n_samples),
    })
    y = pd.Series(np.random.randn(n_samples), name='target')

    print(f"[OK] Created test data: {X.shape[0]} samples, {X.shape[1]} features")

    # Try to initialize the model
    print("Attempting to initialize TabPFNv2Model...")
    model = TabPFNV2Model(path='/tmp/test_tabpfnv2', name='test_tabpfnv2')
    print("[OK] TabPFNv2Model initialized")

    # Try to fit (this is where the error typically occurs)
    print("Attempting to fit model...")
    model.fit(X=X, y=y)
    print("[OK] Model fitted successfully!")

    # Try to predict
    predictions = model.predict(X)
    print(f"[OK] Predictions generated: shape {predictions.shape}")

    print("\n[SUCCESS] TabPFNv2 is working correctly with tabpfn 2.1.0!")

except Exception as e:
    print(f"\n[ERROR] TabPFNv2 failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

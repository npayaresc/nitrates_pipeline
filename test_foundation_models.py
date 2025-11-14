#!/usr/bin/env python3
"""
Test script to verify that tabular foundation models are available in AutoGluon.
Tests: TabPFNv2, TabICL, Mitra, TabM, RealMLP
"""

import sys

def test_foundation_models():
    """Test if foundation models are available in AutoGluon."""
    print("=" * 80)
    print("Testing AutoGluon Tabular Foundation Models")
    print("=" * 80)

    # Test 1: Check AutoGluon version
    print("\n1. Checking AutoGluon version...")
    try:
        import autogluon.core as ag
        from autogluon.tabular import TabularPredictor
        version = ag.__version__
        print(f"   [OK] AutoGluon version: {version}")
        # Parse version and check if >= 1.4.0
        major, minor = map(int, version.split('.')[:2])
        if (major, minor) < (1, 4):
            print(f"   [ERROR] AutoGluon {version} is too old. Need >= 1.4.0")
            return False
    except ImportError as e:
        print(f"   [ERROR] Cannot import AutoGluon: {e}")
        return False

    # Test 2: Check for foundation model imports
    print("\n2. Checking foundation model availability...")
    models_to_check = {
        'TabPFNv2': 'autogluon.tabular.models.tabpfnv2.tabpfnv2_model',
        'TabICL': 'autogluon.tabular.models.tabicl.tabicl_model',
        'Mitra': 'autogluon.tabular.models.mitra.mitra_model',
        'TabM': 'autogluon.tabular.models.tabm.tabm_model',
        'RealMLP': 'autogluon.tabular.models.realmlp.realmlp_model',
    }

    available_models = []
    missing_models = []

    for model_name, module_path in models_to_check.items():
        try:
            # Try to import the model module
            parts = module_path.split('.')
            module = __import__(module_path, fromlist=[parts[-1]])
            print(f"   [OK] {model_name}: Available")
            available_models.append(model_name)
        except ImportError as e:
            print(f"   [ERROR] {model_name}: Not available ({e})")
            missing_models.append(model_name)

    # Test 3: Check configuration
    print("\n3. Checking pipeline configuration...")
    try:
        from src.config.pipeline_config import config

        # Check if foundation models are in hyperparameters
        foundation_keys = ['TABPFNV2', 'TABICL', 'MITRA', 'TABM', 'REALMLP']
        configured_models = []

        for key in foundation_keys:
            if key in config.autogluon.hyperparameters:
                print(f"   [OK] {key}: Configured in hyperparameters")
                configured_models.append(key)
            else:
                print(f"   [ERROR] {key}: Not configured in hyperparameters")

        # Check preset
        preset = config.autogluon.presets
        print(f"\n   Current preset: {preset}")
        if preset == "extreme":
            print("   [OK] Using 'extreme' preset (foundation models enabled)")
        else:
            print(f"   [INFO] Using '{preset}' preset (foundation models available but not auto-enabled)")
            print("   [INFO] To enable foundation models, set presets='extreme' in pipeline_config.py")

    except Exception as e:
        print(f"   [ERROR] Cannot check configuration: {e}")
        return False

    # Test 4: Check optional dependencies
    print("\n4. Checking optional dependencies...")
    optional_deps = {
        'tabpfn': 'TabPFNv2',
        'tabicl': 'TabICL',
    }

    for pkg, model in optional_deps.items():
        try:
            __import__(pkg)
            print(f"   [OK] {pkg}: Installed (required for {model})")
        except ImportError:
            print(f"   [ERROR] {pkg}: Not installed (needed for {model})")
            print(f"      Install with: uv pip install 'autogluon.tabular[{pkg}]'")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Available models: {len(available_models)}/{len(models_to_check)}")
    print(f"  Available: {', '.join(available_models) if available_models else 'None'}")
    if missing_models:
        print(f"  Missing: {', '.join(missing_models)}")

    print(f"\nConfigured models: {len(configured_models)}/{len(foundation_keys)}")
    print(f"  Configured: {', '.join(configured_models) if configured_models else 'None'}")

    # Final verdict
    print("\n" + "=" * 80)
    if len(available_models) >= 3:  # At least 3 models should be available
        print("[SUCCESS] Foundation models are available and ready to use!")
        print("\nNext steps:")
        print("1. Set presets='extreme' in src/config/pipeline_config.py")
        print("2. Run: python main.py autogluon")
        return True
    else:
        print("[INCOMPLETE] Some foundation models are missing")
        print("\nTroubleshooting:")
        print("1. Install missing dependencies:")
        print("   uv pip install 'autogluon.tabular[mitra,tabicl,tabpfn]'")
        print("2. Check AutoGluon version: must be >= 1.4.0")
        return False

if __name__ == "__main__":
    try:
        success = test_foundation_models()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n[FATAL ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

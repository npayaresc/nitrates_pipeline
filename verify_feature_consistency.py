#!/usr/bin/env python3
"""
Verify Feature Name Consistency

This script verifies that the feature names stored in .feature_names.json files
match the features actually used by the models after all filtering configured
in pipeline_config.py.

It checks:
1. Feature selection filtering (use_feature_selection, feature_selection_method)
2. SHAP-based feature selection (use_shap_feature_selection)
3. Feature strategy filtering (N_only, simple_only, full_context)
4. Dimension reduction effects
"""

import json
import joblib  # Models are saved with joblib.dump in model_trainer.py
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.config.pipeline_config import config


def load_model_and_feature_names(model_path: Path) -> Tuple[Optional[object], Optional[Dict], Optional[List[str]]]:
    """
    Load a model and its associated feature names file.

    Returns:
        Tuple of (model, feature_info_dict, feature_names_list)
    """
    # Load model using joblib (models are saved with joblib.dump in model_trainer.py)
    model = None
    load_error = None

    try:
        model = joblib.load(model_path)
        print(f"[OK] Loaded model: {model_path.name}")
    except Exception as e:
        load_error = e
        print(f"[ERROR] Failed to load model {model_path.name}: {load_error}")
        print(f"  -> Skipping model verification, checking feature_names.json only")
        # Continue to load feature names anyway

    # Load feature names file
    feature_names_path = model_path.with_suffix('.feature_names.json')
    if not feature_names_path.exists():
        print(f"[ERROR] Feature names file not found: {feature_names_path.name}")
        return model, None, None

    try:
        with open(feature_names_path, 'r') as f:
            feature_info = json.load(f)
        print(f"[OK] Loaded feature info: {feature_names_path.name}")
    except Exception as e:
        print(f"[ERROR] Failed to load feature names {feature_names_path.name}: {e}")
        return model, None, None

    feature_names = feature_info.get('feature_names', [])
    print(f"  - Stored feature count: {feature_info.get('feature_count', 'N/A')}")
    print(f"  - Strategy: {feature_info.get('strategy', 'unknown')}")
    print(f"  - Pipeline steps: {feature_info.get('pipeline_steps', [])}")

    transformations = feature_info.get('transformations', {})
    if transformations.get('feature_selection'):
        fs_info = transformations['feature_selection']
        print(f"  - Feature selection: {fs_info.get('method', 'unknown')} ({fs_info.get('n_features_selected', 'N/A')} selected)")

    if transformations.get('dimension_reduction'):
        dr_info = transformations['dimension_reduction']
        print(f"  - Dimension reduction: {dr_info.get('method', 'unknown')} ({dr_info.get('n_components', 'N/A')} components)")

    return model, feature_info, feature_names


def extract_model_feature_count(model: object) -> Optional[int]:
    """
    Extract the number of features the model expects from various model types.
    """
    # Try to get from pipeline
    if hasattr(model, 'steps'):
        # It's a pipeline
        for step_name, transformer in reversed(model.steps):
            if step_name == 'model' and hasattr(transformer, 'steps'):
                # Nested pipeline
                for nested_name, nested_transformer in reversed(transformer.steps):
                    if nested_name == 'regressor':
                        return _get_feature_count_from_estimator(nested_transformer)
            elif step_name == 'regressor':
                return _get_feature_count_from_estimator(transformer)
    else:
        # Direct estimator
        return _get_feature_count_from_estimator(model)

    return None


def _get_feature_count_from_estimator(estimator) -> Optional[int]:
    """Get feature count from a fitted estimator."""
    # Try different attributes depending on model type
    if hasattr(estimator, 'n_features_in_'):
        return estimator.n_features_in_
    elif hasattr(estimator, 'feature_importances_'):
        return len(estimator.feature_importances_)
    elif hasattr(estimator, 'coef_'):
        coef = estimator.coef_
        if len(coef.shape) == 1:
            return len(coef)
        else:
            return coef.shape[1]

    return None


def verify_model(model_path: Path) -> Dict:
    """
    Verify a single model's feature consistency.

    Returns:
        Dictionary with verification results
    """
    print("\n" + "="*80)
    print(f"VERIFYING MODEL: {model_path.name}")
    print("="*80)

    model, feature_info, stored_feature_names = load_model_and_feature_names(model_path)

    if feature_info is None:
        return {'status': 'error', 'message': 'Failed to load feature info - no .feature_names.json file'}

    # Extract expected feature count from model (if model loaded successfully)
    model_feature_count = None
    if model is not None:
        model_feature_count = extract_model_feature_count(model)
    else:
        print("[WARNING] Model could not be loaded, will verify feature_names.json file only")

    if model_feature_count is None:
        print("[WARNING] Could not extract feature count from model")
    else:
        print(f"\n[INFO] Model expects: {model_feature_count} features")

    # Extract stored feature count
    stored_count = feature_info.get('feature_count')
    print(f"[INFO] Stored count: {stored_count} features")
    print(f"[INFO] Stored names: {len(stored_feature_names)} feature names")

    # Verify consistency
    results = {
        'model_path': str(model_path),
        'model_name': model_path.stem,
        'strategy': feature_info.get('strategy', 'unknown'),
        'stored_count': stored_count,
        'stored_names_count': len(stored_feature_names),
        'model_expected_count': model_feature_count,
        'consistent': False,
        'issues': []
    }

    # Check 1: Stored count vs stored names count
    if stored_count != len(stored_feature_names):
        issue = f"Mismatch: stored count ({stored_count}) != stored names count ({len(stored_feature_names)})"
        results['issues'].append(issue)
        print(f"[FAIL] {issue}")
    else:
        print(f"[PASS] Stored count matches stored names count")

    # Check 2: Model expected count vs stored count
    if model_feature_count is not None:
        if model_feature_count != stored_count:
            issue = f"Mismatch: model expects ({model_feature_count}) != stored count ({stored_count})"
            results['issues'].append(issue)
            print(f"[FAIL] {issue}")
        else:
            print(f"[PASS] Model expected count matches stored count")

    # Check 3: Model expected count vs stored names count
    if model_feature_count is not None:
        if model_feature_count != len(stored_feature_names):
            issue = f"Mismatch: model expects ({model_feature_count}) != stored names count ({len(stored_feature_names)})"
            results['issues'].append(issue)
            print(f"[FAIL] {issue}")
        else:
            print(f"[PASS] Model expected count matches stored names count")

    # Overall consistency
    if not results['issues'] and model_feature_count is not None:
        results['consistent'] = True
        results['status'] = 'pass'
        print(f"\n[PASS] All feature counts are consistent")
    elif not results['issues']:
        results['status'] = 'warning'
        results['issues'].append('Could not verify against model (feature count extraction failed)')
        print(f"\n[WARNING] Could not fully verify (model feature count unavailable)")
    else:
        results['status'] = 'fail'
        print(f"\n[FAIL] Feature count inconsistencies detected")

    # Show first 10 feature names as sample
    if stored_feature_names:
        print(f"\nFirst 10 stored feature names:")
        for i, name in enumerate(stored_feature_names[:10], 1):
            print(f"   {i:2d}. {name}")
        if len(stored_feature_names) > 10:
            print(f"   ... and {len(stored_feature_names) - 10} more")

    return results


def main():
    """Main verification function."""
    print("\n" + "="*80)
    print("FEATURE CONSISTENCY VERIFICATION")
    print("="*80)
    print(f"\nPipeline Config Settings:")
    print(f"  - Feature strategies: {config.feature_strategies}")
    print(f"  - Use feature selection: {config.use_feature_selection}")
    if config.use_feature_selection:
        print(f"    - Method: {config.feature_selection_method}")
        print(f"    - N features: {config.n_features_to_select}")
    print(f"  - Use SHAP selection: {config.use_shap_feature_selection}")
    if config.use_shap_feature_selection:
        print(f"    - SHAP file: {config.shap_importance_file}")
        print(f"    - Top N: {config.shap_top_n_features}")
    print(f"  - Use dimension reduction: {config.use_dimension_reduction}")
    if config.use_dimension_reduction:
        print(f"    - Method: {config.dimension_reduction.method}")
        print(f"    - N components: {config.dimension_reduction.n_components}")

    # Find models to verify
    models_dir = Path("/home/payanico/nitrates_pipeline/models")

    # Get all .pkl files
    model_files = list(models_dir.glob("*.pkl"))

    # Filter to only models with .feature_names.json files
    models_to_verify = [m for m in model_files if m.with_suffix('.feature_names.json').exists()]

    print(f"\nFound {len(models_to_verify)} models with feature names files")

    if not models_to_verify:
        print("\n[WARNING] No models found with feature names files!")
        print("Train a model first using: python main.py train")
        return

    # Ask user which models to verify
    print("\nSelect models to verify:")
    print("  1. All models")
    print("  2. Latest N_only model")
    print("  3. Latest full_context model")
    print("  4. Latest simple_only model")
    print("  5. Specific model (enter pattern)")

    choice = input("\nEnter choice (1-5) [default: 2]: ").strip() or "2"

    selected_models = []

    if choice == "1":
        selected_models = models_to_verify
    elif choice in ["2", "3", "4"]:
        strategy_map = {"2": "N_only", "3": "full_context", "4": "simple_only"}
        strategy = strategy_map[choice]
        strategy_models = [m for m in models_to_verify if strategy in m.name]
        if strategy_models:
            # Sort by modification time, get latest
            latest = max(strategy_models, key=lambda p: p.stat().st_mtime)
            selected_models = [latest]
        else:
            print(f"[WARNING] No {strategy} models found")
            return
    elif choice == "5":
        pattern = input("Enter model name pattern: ").strip()
        selected_models = [m for m in models_to_verify if pattern in m.name]
        if not selected_models:
            print(f"[WARNING] No models matching '{pattern}' found")
            return

    # Verify selected models
    all_results = []
    for model_path in selected_models:
        result = verify_model(model_path)
        all_results.append(result)

    # Summary
    print("\n" + "="*80)
    print("VERIFICATION SUMMARY")
    print("="*80)

    passed = sum(1 for r in all_results if r.get('status') == 'pass')
    failed = sum(1 for r in all_results if r.get('status') == 'fail')
    warnings = sum(1 for r in all_results if r.get('status') == 'warning')
    errors = sum(1 for r in all_results if r.get('status') == 'error')

    print(f"\nTotal models verified: {len(all_results)}")
    print(f"  [PASS] Passed: {passed}")
    print(f"  [FAIL] Failed: {failed}")
    print(f"  [WARNING] Warnings: {warnings}")
    print(f"  [ERROR] Errors: {errors}")

    if failed > 0:
        print("\nFailed models:")
        for r in all_results:
            if r.get('status') == 'fail':
                print(f"\n  {r['model_name']}")
                for issue in r['issues']:
                    print(f"    - {issue}")

    print("\n" + "="*80)

    if failed == 0 and errors == 0:
        print("SUCCESS: ALL VERIFICATIONS PASSED")
    elif warnings > 0 and failed == 0:
        print("WARNING: VERIFICATION COMPLETED WITH WARNINGS")
    else:
        print("FAILURE: VERIFICATION FAILED")

    print("="*80 + "\n")


if __name__ == "__main__":
    main()

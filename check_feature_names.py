#!/usr/bin/env python3
"""
Quick Feature Names Checker

Verifies .feature_names.json files for consistency without loading models.
This is faster and more reliable than loading pickled models.
"""

import json
from pathlib import Path
from typing import Dict, List


def check_feature_names_file(json_path: Path) -> Dict:
    """Check a single feature_names.json file."""
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        return {
            'path': str(json_path),
            'status': 'error',
            'message': f'Failed to load: {e}'
        }

    # Extract info
    model_name = json_path.stem.replace('.feature_names', '')
    feature_count = data.get('feature_count')
    feature_names = data.get('feature_names', [])
    actual_count = len(feature_names) if feature_names else 0
    strategy = data.get('strategy', 'unknown')
    transformations = data.get('transformations', {})

    # Check consistency
    issues = []

    if feature_count != actual_count:
        issues.append(f"Count mismatch: declared={feature_count}, actual={actual_count}")

    if feature_names is None:
        issues.append("No feature_names array in file")
    elif len(feature_names) == 0:
        issues.append("Empty feature_names array")

    # Extract transformation info
    fs_info = transformations.get('feature_selection')
    dr_info = transformations.get('dimension_reduction')

    result = {
        'model_name': model_name,
        'strategy': strategy,
        'declared_count': feature_count,
        'actual_count': actual_count,
        'consistent': len(issues) == 0,
        'issues': issues,
        'has_feature_selection': fs_info is not None,
        'has_dimension_reduction': dr_info is not None,
        'pipeline_steps': data.get('pipeline_steps', []),
        'first_features': feature_names[:10] if feature_names else []
    }

    if fs_info:
        result['feature_selection'] = {
            'method': fs_info.get('method', 'unknown'),
            'n_selected': fs_info.get('n_features_selected')
        }

    if dr_info:
        result['dimension_reduction'] = {
            'method': dr_info.get('method', 'unknown'),
            'n_components': dr_info.get('n_components')
        }

    return result


def main():
    """Main function."""
    models_dir = Path("/home/payanico/nitrates_pipeline/models")

    # Find all .feature_names.json files
    json_files = sorted(models_dir.glob("*.feature_names.json"))

    if not json_files:
        print("WARNING: No .feature_names.json files found in models directory!")
        return

    print("="*80)
    print("FEATURE NAMES FILE CHECKER")
    print("="*80)
    print(f"\nFound {len(json_files)} feature names files\n")

    # Ask user to filter
    print("Filter by strategy:")
    print("  1. All files")
    print("  2. N_only")
    print("  3. simple_only")
    print("  4. full_context")
    print("  5. Custom pattern")

    choice = input("\nEnter choice (1-5) [default: 1]: ").strip() or "1"

    if choice == "2":
        json_files = [f for f in json_files if "N_only" in f.name]
    elif choice == "3":
        json_files = [f for f in json_files if "simple_only" in f.name]
    elif choice == "4":
        json_files = [f for f in json_files if "full_context" in f.name]
    elif choice == "5":
        pattern = input("Enter pattern: ").strip()
        json_files = [f for f in json_files if pattern in f.name]

    if not json_files:
        print("WARNING: No files match the selected filter")
        return

    # Get latest N files
    if len(json_files) > 10:
        show_latest = input(f"\nFound {len(json_files)} files. Show only latest 10? (y/n) [y]: ").strip().lower()
        if show_latest != 'n':
            json_files = sorted(json_files, key=lambda p: p.stat().st_mtime, reverse=True)[:10]

    print(f"\nChecking {len(json_files)} files...\n")

    # Check each file
    results = []
    for json_path in json_files:
        result = check_feature_names_file(json_path)
        results.append(result)

    # Display results
    print("="*80)
    print("RESULTS")
    print("="*80)

    for i, r in enumerate(results, 1):
        status_text = "[PASS]" if r.get('consistent') else "[FAIL]"
        print(f"\n{i}. {status_text} {r['model_name']}")
        print(f"   Strategy: {r['strategy']}")
        print(f"   Feature count: {r['declared_count']} (actual: {r['actual_count']})")

        if r.get('has_feature_selection'):
            fs = r['feature_selection']
            print(f"   Feature selection: {fs['method']} → {fs['n_selected']} features")

        if r.get('has_dimension_reduction'):
            dr = r['dimension_reduction']
            print(f"   Dimension reduction: {dr['method']} → {dr['n_components']} components")

        if r.get('pipeline_steps'):
            print(f"   Pipeline: {' → '.join(r['pipeline_steps'])}")

        if r.get('issues'):
            for issue in r['issues']:
                print(f"   WARNING: {issue}")

        if r.get('first_features') and r['consistent']:
            print(f"   First 5 features: {', '.join(r['first_features'][:5])}")

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    total = len(results)
    consistent = sum(1 for r in results if r.get('consistent'))
    inconsistent = total - consistent

    print(f"\nTotal files checked: {total}")
    print(f"  [PASS] Consistent: {consistent}")
    print(f"  [FAIL] Inconsistent: {inconsistent}")

    # Group by strategy
    strategies = {}
    for r in results:
        strat = r['strategy']
        if strat not in strategies:
            strategies[strat] = {'consistent': 0, 'total': 0, 'avg_features': []}
        strategies[strat]['total'] += 1
        if r.get('consistent'):
            strategies[strat]['consistent'] += 1
        if r['actual_count']:
            strategies[strat]['avg_features'].append(r['actual_count'])

    if len(strategies) > 1:
        print("\nBy strategy:")
        for strat, stats in strategies.items():
            avg_feat = sum(stats['avg_features']) / len(stats['avg_features']) if stats['avg_features'] else 0
            print(f"  {strat}: {stats['consistent']}/{stats['total']} consistent, avg {avg_feat:.0f} features")

    # Feature selection stats
    with_fs = sum(1 for r in results if r.get('has_feature_selection'))
    with_dr = sum(1 for r in results if r.get('has_dimension_reduction'))

    if with_fs > 0 or with_dr > 0:
        print("\nTransformations applied:")
        if with_fs > 0:
            print(f"  Feature selection: {with_fs}/{total} models")
        if with_dr > 0:
            print(f"  Dimension reduction: {with_dr}/{total} models")

    print("\n" + "="*80)

    if inconsistent == 0:
        print("SUCCESS: ALL FILES ARE CONSISTENT")
    else:
        print("WARNING: SOME FILES HAVE ISSUES")

    print("="*80 + "\n")


if __name__ == "__main__":
    main()

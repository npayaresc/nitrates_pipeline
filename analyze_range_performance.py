#!/usr/bin/env python3
"""
Analyze model performance across different NO3 concentration ranges.
Identifies where models perform best during optimization.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Dict, Tuple
import argparse
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)


def calculate_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Mean Absolute Percentage Error."""
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def analyze_range_performance(
    predictions_df: pd.DataFrame,
    ranges: List[Tuple[float, float, str]]
) -> pd.DataFrame:
    """
    Analyze model performance across different concentration ranges.

    Args:
        predictions_df: DataFrame with ElementValue and PredictedValue columns
        ranges: List of (min, max, label) tuples defining concentration ranges

    Returns:
        DataFrame with performance metrics for each range
    """
    results = []

    for min_val, max_val, label in ranges:
        # Filter data for this range
        mask = (predictions_df['ElementValue'] >= min_val) & (predictions_df['ElementValue'] < max_val)
        range_data = predictions_df[mask]

        if len(range_data) < 2:
            continue

        y_true = range_data['ElementValue'].values
        y_pred = range_data['PredictedValue'].values

        # Calculate metrics
        r2 = r2_score(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        mape = calculate_mape(y_true, y_pred)
        rrmse = (rmse / np.mean(y_true)) * 100

        # Calculate percentage within tolerance
        errors = np.abs(y_true - y_pred)
        within_10pct = (errors <= y_true * 0.105).sum() / len(y_true) * 100
        within_20pct = (errors <= y_true * 0.205).sum() / len(y_true) * 100

        results.append({
            'Range': label,
            'Min': min_val,
            'Max': max_val,
            'N_Samples': len(range_data),
            'Mean_Actual': np.mean(y_true),
            'R²': r2,
            'RMSE': rmse,
            'MAE': mae,
            'MAPE': mape,
            'RRMSE': rrmse,
            'Within_10%': within_10pct,
            'Within_20%': within_20pct
        })

    return pd.DataFrame(results)


def plot_range_performance(
    range_results: pd.DataFrame,
    model_name: str,
    output_dir: Path
) -> None:
    """Create comprehensive visualizations of performance across ranges."""

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Performance Analysis Across NO3 Concentration Ranges\n{model_name}',
                 fontsize=16, fontweight='bold')

    # R² by range
    ax = axes[0, 0]
    bars = ax.bar(range(len(range_results)), range_results['R²'],
                  color=plt.cm.RdYlGn(range_results['R²']))
    ax.set_xticks(range(len(range_results)))
    ax.set_xticklabels(range_results['Range'], rotation=45, ha='right')
    ax.set_ylabel('R² Score', fontweight='bold')
    ax.set_title('R² Score by Range')
    ax.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Target: 0.5')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Add value labels on bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=9)

    # RMSE by range
    ax = axes[0, 1]
    bars = ax.bar(range(len(range_results)), range_results['RMSE'],
                  color=plt.cm.RdYlGn_r((range_results['RMSE'] - range_results['RMSE'].min()) /
                                        (range_results['RMSE'].max() - range_results['RMSE'].min())))
    ax.set_xticks(range(len(range_results)))
    ax.set_xticklabels(range_results['Range'], rotation=45, ha='right')
    ax.set_ylabel('RMSE', fontweight='bold')
    ax.set_title('RMSE by Range')
    ax.grid(True, alpha=0.3)

    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.0f}',
                ha='center', va='bottom', fontsize=9)

    # MAPE by range
    ax = axes[0, 2]
    bars = ax.bar(range(len(range_results)), range_results['MAPE'],
                  color=plt.cm.RdYlGn_r((range_results['MAPE'] - range_results['MAPE'].min()) /
                                        (range_results['MAPE'].max() - range_results['MAPE'].min())))
    ax.set_xticks(range(len(range_results)))
    ax.set_xticklabels(range_results['Range'], rotation=45, ha='right')
    ax.set_ylabel('MAPE (%)', fontweight='bold')
    ax.set_title('MAPE by Range')
    ax.grid(True, alpha=0.3)

    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom', fontsize=9)

    # Within tolerance
    ax = axes[1, 0]
    x = range(len(range_results))
    width = 0.35
    bars1 = ax.bar([i - width/2 for i in x], range_results['Within_10%'],
                   width, label='Within 10%', color='steelblue')
    bars2 = ax.bar([i + width/2 for i in x], range_results['Within_20%'],
                   width, label='Within 20%', color='lightcoral')
    ax.set_xticks(range(len(range_results)))
    ax.set_xticklabels(range_results['Range'], rotation=45, ha='right')
    ax.set_ylabel('Percentage (%)', fontweight='bold')
    ax.set_title('Predictions Within Tolerance')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Sample distribution
    ax = axes[1, 1]
    bars = ax.bar(range(len(range_results)), range_results['N_Samples'],
                  color='mediumpurple')
    ax.set_xticks(range(len(range_results)))
    ax.set_xticklabels(range_results['Range'], rotation=45, ha='right')
    ax.set_ylabel('Number of Samples', fontweight='bold')
    ax.set_title('Sample Distribution Across Ranges')
    ax.grid(True, alpha=0.3)

    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=9)

    # Combined score (normalized metrics)
    ax = axes[1, 2]
    # Normalize metrics (higher is better)
    norm_r2 = range_results['R²']
    norm_within20 = range_results['Within_20%'] / 100
    norm_mape = 1 - (range_results['MAPE'] / 100)
    combined_score = (norm_r2 + norm_within20 + norm_mape) / 3

    bars = ax.bar(range(len(range_results)), combined_score,
                  color=plt.cm.RdYlGn(combined_score))
    ax.set_xticks(range(len(range_results)))
    ax.set_xticklabels(range_results['Range'], rotation=45, ha='right')
    ax.set_ylabel('Combined Score', fontweight='bold')
    ax.set_title('Combined Performance Score\n(R² + Within_20% + (1-MAPE))/3')
    ax.grid(True, alpha=0.3)

    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    output_file = output_dir / f'{model_name}_range_performance.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved plot: {output_file}")
    plt.close()


def plot_predictions_by_range(
    predictions_df: pd.DataFrame,
    ranges: List[Tuple[float, float, str]],
    model_name: str,
    output_dir: Path
) -> None:
    """Plot actual vs predicted for each concentration range."""

    n_ranges = len(ranges)
    n_cols = 3
    n_rows = (n_ranges + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 6*n_rows))
    fig.suptitle(f'Actual vs Predicted by Concentration Range\n{model_name}',
                 fontsize=16, fontweight='bold')

    if n_rows == 1:
        axes = axes.reshape(1, -1)

    for idx, (min_val, max_val, label) in enumerate(ranges):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]

        # Filter data for this range
        mask = (predictions_df['ElementValue'] >= min_val) & (predictions_df['ElementValue'] < max_val)
        range_data = predictions_df[mask]

        if len(range_data) < 2:
            ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center')
            ax.set_title(label)
            continue

        y_true = range_data['ElementValue'].values
        y_pred = range_data['PredictedValue'].values

        # Scatter plot
        ax.scatter(y_true, y_pred, alpha=0.6, s=50, edgecolors='k', linewidth=0.5)

        # Perfect prediction line
        min_plot = min(y_true.min(), y_pred.min())
        max_plot = max(y_true.max(), y_pred.max())
        ax.plot([min_plot, max_plot], [min_plot, max_plot], 'r--', lw=2, label='Perfect')

        # ±10% and ±20% bands
        ax.fill_between([min_plot, max_plot],
                        [min_plot * 0.9, max_plot * 0.9],
                        [min_plot * 1.1, max_plot * 1.1],
                        alpha=0.2, color='green', label='±10%')
        ax.fill_between([min_plot, max_plot],
                        [min_plot * 0.8, max_plot * 0.8],
                        [min_plot * 1.2, max_plot * 1.2],
                        alpha=0.1, color='yellow', label='±20%')

        # Calculate metrics
        r2 = r2_score(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)

        ax.set_xlabel('Actual NO3', fontweight='bold')
        ax.set_ylabel('Predicted NO3', fontweight='bold')
        ax.set_title(f'{label}\nR²={r2:.3f}, RMSE={rmse:.0f}, MAE={mae:.0f}')
        ax.legend(loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3)

    # Hide empty subplots
    for idx in range(n_ranges, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].axis('off')

    plt.tight_layout()
    output_file = output_dir / f'{model_name}_predictions_by_range.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved plot: {output_file}")
    plt.close()


def generate_report(
    range_results: pd.DataFrame,
    model_name: str,
    output_dir: Path
) -> None:
    """Generate a text report summarizing range performance."""

    report_file = output_dir / f'{model_name}_range_analysis_report.txt'

    with open(report_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write(f"NO3 Concentration Range Performance Analysis\n")
        f.write(f"Model: {model_name}\n")
        f.write("="*80 + "\n\n")

        # Overall statistics
        f.write("OVERALL STATISTICS\n")
        f.write("-"*80 + "\n")
        f.write(f"Total samples analyzed: {range_results['N_Samples'].sum()}\n")
        f.write(f"Number of ranges: {len(range_results)}\n")
        f.write(f"Average R² across ranges: {range_results['R²'].mean():.4f}\n")
        f.write(f"Average MAPE across ranges: {range_results['MAPE'].mean():.2f}%\n")
        f.write(f"Average Within 20% tolerance: {range_results['Within_20%'].mean():.2f}%\n\n")

        # Best performing ranges
        f.write("BEST PERFORMING RANGES\n")
        f.write("-"*80 + "\n")

        # Best by R²
        best_r2_idx = range_results['R²'].idxmax()
        f.write(f"\nHighest R² Score:\n")
        f.write(f"  Range: {range_results.loc[best_r2_idx, 'Range']}\n")
        f.write(f"  R²: {range_results.loc[best_r2_idx, 'R²']:.4f}\n")
        f.write(f"  RMSE: {range_results.loc[best_r2_idx, 'RMSE']:.2f}\n")
        f.write(f"  Within 20%: {range_results.loc[best_r2_idx, 'Within_20%']:.2f}%\n")
        f.write(f"  Samples: {int(range_results.loc[best_r2_idx, 'N_Samples'])}\n")

        # Best by Within 20%
        best_within_idx = range_results['Within_20%'].idxmax()
        f.write(f"\nBest Within 20% Tolerance:\n")
        f.write(f"  Range: {range_results.loc[best_within_idx, 'Range']}\n")
        f.write(f"  Within 20%: {range_results.loc[best_within_idx, 'Within_20%']:.2f}%\n")
        f.write(f"  R²: {range_results.loc[best_within_idx, 'R²']:.4f}\n")
        f.write(f"  RMSE: {range_results.loc[best_within_idx, 'RMSE']:.2f}\n")
        f.write(f"  Samples: {int(range_results.loc[best_within_idx, 'N_Samples'])}\n")

        # Lowest MAPE
        best_mape_idx = range_results['MAPE'].idxmin()
        f.write(f"\nLowest MAPE:\n")
        f.write(f"  Range: {range_results.loc[best_mape_idx, 'Range']}\n")
        f.write(f"  MAPE: {range_results.loc[best_mape_idx, 'MAPE']:.2f}%\n")
        f.write(f"  R²: {range_results.loc[best_mape_idx, 'R²']:.4f}\n")
        f.write(f"  Within 20%: {range_results.loc[best_mape_idx, 'Within_20%']:.2f}%\n")
        f.write(f"  Samples: {int(range_results.loc[best_mape_idx, 'N_Samples'])}\n")

        # Worst performing ranges
        f.write("\nWORST PERFORMING RANGES\n")
        f.write("-"*80 + "\n")

        worst_r2_idx = range_results['R²'].idxmin()
        f.write(f"\nLowest R² Score:\n")
        f.write(f"  Range: {range_results.loc[worst_r2_idx, 'Range']}\n")
        f.write(f"  R²: {range_results.loc[worst_r2_idx, 'R²']:.4f}\n")
        f.write(f"  RMSE: {range_results.loc[worst_r2_idx, 'RMSE']:.2f}\n")
        f.write(f"  Samples: {int(range_results.loc[worst_r2_idx, 'N_Samples'])}\n")

        # Detailed table
        f.write("\nDETAILED RANGE PERFORMANCE\n")
        f.write("-"*80 + "\n")
        f.write(range_results.to_string(index=False))
        f.write("\n\n")

        # Recommendations
        f.write("RECOMMENDATIONS\n")
        f.write("-"*80 + "\n")

        # Identify ranges needing improvement
        poor_ranges = range_results[range_results['R²'] < 0.3]
        if len(poor_ranges) > 0:
            f.write(f"\nRanges requiring attention (R² < 0.3):\n")
            for _, row in poor_ranges.iterrows():
                f.write(f"  - {row['Range']}: R²={row['R²']:.4f}, "
                       f"Within_20%={row['Within_20%']:.2f}%, "
                       f"Samples={int(row['N_Samples'])}\n")

        # Identify well-performing ranges
        good_ranges = range_results[range_results['R²'] > 0.6]
        if len(good_ranges) > 0:
            f.write(f"\nStrong performing ranges (R² > 0.6):\n")
            for _, row in good_ranges.iterrows():
                f.write(f"  - {row['Range']}: R²={row['R²']:.4f}, "
                       f"Within_20%={row['Within_20%']:.2f}%, "
                       f"Samples={int(row['N_Samples'])}\n")

        f.write("\n" + "="*80 + "\n")

    print(f"Saved report: {report_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Analyze model performance across NO3 concentration ranges'
    )
    parser.add_argument(
        '--predictions',
        type=str,
        required=True,
        help='Path to predictions CSV file'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='reports/range_analysis',
        help='Output directory for results (default: reports/range_analysis)'
    )
    parser.add_argument(
        '--ranges',
        type=str,
        default='auto',
        help='Concentration ranges: "auto" or comma-separated "min1-max1,min2-max2,..."'
    )

    args = parser.parse_args()

    # Load predictions
    print(f"\nLoading predictions from: {args.predictions}")
    predictions_df = pd.read_csv(args.predictions)

    # Extract model name from filename
    model_name = Path(args.predictions).stem.replace('predictions_', '')

    # Define concentration ranges
    if args.ranges == 'auto':
        # Auto-create ranges based on data distribution
        min_val = predictions_df['ElementValue'].min()
        max_val = predictions_df['ElementValue'].max()

        # Create ranges using quantiles
        quantiles = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
        range_bounds = predictions_df['ElementValue'].quantile(quantiles).values

        ranges = []
        for i in range(len(range_bounds) - 1):
            label = f"{int(range_bounds[i])}-{int(range_bounds[i+1])}"
            ranges.append((range_bounds[i], range_bounds[i+1], label))

        print(f"\nAuto-generated {len(ranges)} concentration ranges based on quantiles")
    else:
        # Parse custom ranges
        ranges = []
        for r in args.ranges.split(','):
            min_val, max_val = map(float, r.split('-'))
            label = f"{int(min_val)}-{int(max_val)}"
            ranges.append((min_val, max_val, label))
        print(f"\nUsing {len(ranges)} custom concentration ranges")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Analyze performance by range
    print("\nAnalyzing performance across ranges...")
    range_results = analyze_range_performance(predictions_df, ranges)

    # Save detailed results to CSV
    csv_file = output_dir / f'{model_name}_range_performance.csv'
    range_results.to_csv(csv_file, index=False)
    print(f"Saved results: {csv_file}")

    # Generate visualizations
    print("\nGenerating visualizations...")
    plot_range_performance(range_results, model_name, output_dir)
    plot_predictions_by_range(predictions_df, ranges, model_name, output_dir)

    # Generate report
    print("\nGenerating report...")
    generate_report(range_results, model_name, output_dir)

    # Display summary
    print("\n" + "="*80)
    print("PERFORMANCE SUMMARY")
    print("="*80)
    print(f"\nModel: {model_name}")
    print(f"Total samples: {range_results['N_Samples'].sum()}")
    print(f"\nBest performing range (by R²):")
    best_idx = range_results['R²'].idxmax()
    print(f"  {range_results.loc[best_idx, 'Range']}: R²={range_results.loc[best_idx, 'R²']:.4f}")
    print(f"\nWorst performing range (by R²):")
    worst_idx = range_results['R²'].idxmin()
    print(f"  {range_results.loc[worst_idx, 'Range']}: R²={range_results.loc[worst_idx, 'R²']:.4f}")
    print("\n" + "="*80)
    print(f"\nAll results saved to: {output_dir}/")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()

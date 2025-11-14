#!/usr/bin/env python3
"""
Compare range performance across multiple models.
Identifies which models perform best in which concentration ranges.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 10)


def load_all_range_results(analysis_dir: Path) -> pd.DataFrame:
    """Load all range performance CSV files and combine them."""

    csv_files = list(analysis_dir.glob("*_range_performance.csv"))

    if not csv_files:
        raise FileNotFoundError(f"No range performance CSV files found in {analysis_dir}")

    all_results = []

    for csv_file in csv_files:
        df = pd.read_csv(csv_file)

        # Extract model name from filename
        model_name = csv_file.stem.replace('_range_performance', '')

        df['Model'] = model_name
        all_results.append(df)

    combined_df = pd.concat(all_results, ignore_index=True)

    return combined_df


def plot_model_comparison(combined_df: pd.DataFrame, output_dir: Path) -> None:
    """Create comparison visualizations across models."""

    # Get unique ranges and models
    ranges = combined_df['Range'].unique()
    models = combined_df['Model'].unique()

    # 1. R² comparison heatmap
    fig, ax = plt.subplots(figsize=(14, max(8, len(models) * 0.5)))

    pivot_r2 = combined_df.pivot(index='Model', columns='Range', values='R²')
    sns.heatmap(pivot_r2, annot=True, fmt='.3f', cmap='RdYlGn', center=0,
                cbar_kws={'label': 'R² Score'}, ax=ax)
    ax.set_title('R² Score Comparison Across Models and Ranges', fontsize=14, fontweight='bold')
    ax.set_xlabel('Concentration Range', fontweight='bold')
    ax.set_ylabel('Model', fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_dir / 'model_comparison_r2_heatmap.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'model_comparison_r2_heatmap.png'}")
    plt.close()

    # 2. Within 20% tolerance comparison
    fig, ax = plt.subplots(figsize=(14, max(8, len(models) * 0.5)))

    pivot_within20 = combined_df.pivot(index='Model', columns='Range', values='Within_20%')
    sns.heatmap(pivot_within20, annot=True, fmt='.1f', cmap='RdYlGn', vmin=0, vmax=100,
                cbar_kws={'label': 'Within 20% (%)'}, ax=ax)
    ax.set_title('Within 20% Tolerance Across Models and Ranges', fontsize=14, fontweight='bold')
    ax.set_xlabel('Concentration Range', fontweight='bold')
    ax.set_ylabel('Model', fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_dir / 'model_comparison_within20_heatmap.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'model_comparison_within20_heatmap.png'}")
    plt.close()

    # 3. MAPE comparison
    fig, ax = plt.subplots(figsize=(14, max(8, len(models) * 0.5)))

    pivot_mape = combined_df.pivot(index='Model', columns='Range', values='MAPE')
    sns.heatmap(pivot_mape, annot=True, fmt='.1f', cmap='RdYlGn_r', vmin=0,
                cbar_kws={'label': 'MAPE (%)'}, ax=ax)
    ax.set_title('MAPE Across Models and Ranges', fontsize=14, fontweight='bold')
    ax.set_xlabel('Concentration Range', fontweight='bold')
    ax.set_ylabel('Model', fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_dir / 'model_comparison_mape_heatmap.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'model_comparison_mape_heatmap.png'}")
    plt.close()

    # 4. Best model per range
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Best Performing Models by Range', fontsize=16, fontweight='bold')

    # Best by R²
    ax = axes[0, 0]
    best_r2 = combined_df.loc[combined_df.groupby('Range')['R²'].idxmax()]
    bars = ax.bar(range(len(best_r2)), best_r2['R²'],
                  color=plt.cm.RdYlGn(best_r2['R²'].values))
    ax.set_xticks(range(len(best_r2)))
    ax.set_xticklabels(best_r2['Range'], rotation=45, ha='right')
    ax.set_ylabel('R² Score', fontweight='bold')
    ax.set_title('Best R² by Range')
    ax.grid(True, alpha=0.3)

    for i, (bar, model) in enumerate(zip(bars, best_r2['Model'])):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{model[:20]}\n{height:.3f}',
                ha='center', va='bottom', fontsize=7)

    # Best by Within 20%
    ax = axes[0, 1]
    best_within20 = combined_df.loc[combined_df.groupby('Range')['Within_20%'].idxmax()]
    bars = ax.bar(range(len(best_within20)), best_within20['Within_20%'],
                  color=plt.cm.RdYlGn(best_within20['Within_20%'].values / 100))
    ax.set_xticks(range(len(best_within20)))
    ax.set_xticklabels(best_within20['Range'], rotation=45, ha='right')
    ax.set_ylabel('Within 20% (%)', fontweight='bold')
    ax.set_title('Best Within 20% Tolerance by Range')
    ax.grid(True, alpha=0.3)

    for i, (bar, model) in enumerate(zip(bars, best_within20['Model'])):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{model[:20]}\n{height:.1f}%',
                ha='center', va='bottom', fontsize=7)

    # Best by MAPE (lowest)
    ax = axes[1, 0]
    best_mape = combined_df.loc[combined_df.groupby('Range')['MAPE'].idxmin()]
    bars = ax.bar(range(len(best_mape)), best_mape['MAPE'],
                  color=plt.cm.RdYlGn_r(best_mape['MAPE'].values / 50))
    ax.set_xticks(range(len(best_mape)))
    ax.set_xticklabels(best_mape['Range'], rotation=45, ha='right')
    ax.set_ylabel('MAPE (%)', fontweight='bold')
    ax.set_title('Best MAPE (Lowest) by Range')
    ax.grid(True, alpha=0.3)

    for i, (bar, model) in enumerate(zip(bars, best_mape['Model'])):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{model[:20]}\n{height:.1f}%',
                ha='center', va='bottom', fontsize=7)

    # Model frequency (how often each model is best)
    ax = axes[1, 1]
    all_best_models = pd.concat([
        best_r2[['Model']],
        best_within20[['Model']],
        best_mape[['Model']]
    ])
    model_counts = all_best_models['Model'].value_counts()

    bars = ax.bar(range(len(model_counts)), model_counts.values, color='steelblue')
    ax.set_xticks(range(len(model_counts)))
    ax.set_xticklabels([m[:20] for m in model_counts.index], rotation=45, ha='right')
    ax.set_ylabel('Count', fontweight='bold')
    ax.set_title('Frequency: How Often Each Model is Best\n(Across R², Within 20%, MAPE)')
    ax.grid(True, alpha=0.3)

    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_dir / 'model_comparison_best_per_range.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'model_comparison_best_per_range.png'}")
    plt.close()


def generate_comparison_report(combined_df: pd.DataFrame, output_dir: Path) -> None:
    """Generate a comparison report across all models."""

    report_file = output_dir / 'model_range_comparison_report.txt'

    with open(report_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("MODEL COMPARISON: Range Performance Analysis\n")
        f.write("="*80 + "\n\n")

        # Overall statistics
        f.write("OVERALL STATISTICS\n")
        f.write("-"*80 + "\n")
        f.write(f"Models analyzed: {combined_df['Model'].nunique()}\n")
        f.write(f"Concentration ranges: {combined_df['Range'].nunique()}\n")
        f.write(f"Total comparisons: {len(combined_df)}\n\n")

        # Best models overall
        f.write("BEST MODELS OVERALL\n")
        f.write("-"*80 + "\n\n")

        # Best average R²
        avg_r2 = combined_df.groupby('Model')['R²'].mean().sort_values(ascending=False)
        f.write("Top 5 Models by Average R²:\n")
        for i, (model, r2) in enumerate(avg_r2.head(5).items(), 1):
            f.write(f"  {i}. {model}: R²={r2:.4f}\n")
        f.write("\n")

        # Best average Within 20%
        avg_within20 = combined_df.groupby('Model')['Within_20%'].mean().sort_values(ascending=False)
        f.write("Top 5 Models by Average Within 20% Tolerance:\n")
        for i, (model, within20) in enumerate(avg_within20.head(5).items(), 1):
            f.write(f"  {i}. {model}: {within20:.2f}%\n")
        f.write("\n")

        # Best average MAPE (lowest)
        avg_mape = combined_df.groupby('Model')['MAPE'].mean().sort_values()
        f.write("Top 5 Models by Average MAPE (Lowest):\n")
        for i, (model, mape) in enumerate(avg_mape.head(5).items(), 1):
            f.write(f"  {i}. {model}: {mape:.2f}%\n")
        f.write("\n")

        # Best model per range
        f.write("BEST MODEL PER CONCENTRATION RANGE\n")
        f.write("-"*80 + "\n\n")

        for range_name in combined_df['Range'].unique():
            range_data = combined_df[combined_df['Range'] == range_name]

            f.write(f"Range: {range_name}\n")

            # Best by R²
            best_r2 = range_data.loc[range_data['R²'].idxmax()]
            f.write(f"  Best R²: {best_r2['Model'][:40]}\n")
            f.write(f"    R²={best_r2['R²']:.4f}, RMSE={best_r2['RMSE']:.2f}, "
                   f"Within_20%={best_r2['Within_20%']:.2f}%\n")

            # Best by Within 20%
            best_within = range_data.loc[range_data['Within_20%'].idxmax()]
            f.write(f"  Best Within 20%: {best_within['Model'][:40]}\n")
            f.write(f"    Within_20%={best_within['Within_20%']:.2f}%, R²={best_within['R²']:.4f}, "
                   f"MAPE={best_within['MAPE']:.2f}%\n")

            # Best by MAPE
            best_mape = range_data.loc[range_data['MAPE'].idxmin()]
            f.write(f"  Best MAPE: {best_mape['Model'][:40]}\n")
            f.write(f"    MAPE={best_mape['MAPE']:.2f}%, R²={best_mape['R²']:.4f}, "
                   f"Within_20%={best_mape['Within_20%']:.2f}%\n")
            f.write("\n")

        # Recommendations
        f.write("RECOMMENDATIONS\n")
        f.write("-"*80 + "\n\n")

        # Most consistent model
        r2_std = combined_df.groupby('Model')['R²'].std().sort_values()
        f.write("Most Consistent Model (by R² variance):\n")
        f.write(f"  {r2_std.index[0]}: std={r2_std.values[0]:.4f}\n\n")

        # Best all-around model (combined score)
        combined_df['Combined_Score'] = (
            combined_df['R²'] +
            (combined_df['Within_20%'] / 100) +
            (1 - combined_df['MAPE'] / 100)
        ) / 3

        avg_combined = combined_df.groupby('Model')['Combined_Score'].mean().sort_values(ascending=False)
        f.write("Best All-Around Model (Combined Score):\n")
        f.write(f"  {avg_combined.index[0]}: score={avg_combined.values[0]:.4f}\n")
        f.write("  (Combined Score = (R² + Within_20%/100 + (1-MAPE/100))/3)\n\n")

        f.write("="*80 + "\n")

    print(f"Saved report: {report_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Compare range performance across multiple models'
    )
    parser.add_argument(
        '--analysis-dir',
        type=str,
        default='reports/range_analysis',
        help='Directory containing range performance CSV files'
    )

    args = parser.parse_args()

    analysis_dir = Path(args.analysis_dir)

    if not analysis_dir.exists():
        print(f"Error: Analysis directory not found: {analysis_dir}")
        print("Run analyze_all_models_ranges.sh first to generate range analysis files.")
        return

    print("\nLoading range performance results...")
    combined_df = load_all_range_results(analysis_dir)

    print(f"Loaded results for {combined_df['Model'].nunique()} models")
    print(f"Analyzing {combined_df['Range'].nunique()} concentration ranges\n")

    # Generate comparison visualizations
    print("Generating comparison visualizations...")
    plot_model_comparison(combined_df, analysis_dir)

    # Generate comparison report
    print("\nGenerating comparison report...")
    generate_comparison_report(combined_df, analysis_dir)

    # Save combined results
    combined_csv = analysis_dir / 'all_models_range_comparison.csv'
    combined_df.to_csv(combined_csv, index=False)
    print(f"\nSaved combined results: {combined_csv}")

    print("\n" + "="*80)
    print("COMPARISON COMPLETE")
    print("="*80)
    print(f"\nResults saved to: {analysis_dir}/")
    print("\nGenerated files:")
    print("  - all_models_range_comparison.csv")
    print("  - model_comparison_r2_heatmap.png")
    print("  - model_comparison_within20_heatmap.png")
    print("  - model_comparison_mape_heatmap.png")
    print("  - model_comparison_best_per_range.png")
    print("  - model_range_comparison_report.txt")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()

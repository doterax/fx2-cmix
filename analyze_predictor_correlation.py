#!/usr/bin/env python3
"""
Analyze correlations between predictors to identify redundant ones.
Helps optimize CPU performance by eliminating highly correlated predictors.
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

def analyze_correlations(csv_path='predictor_stats.csv', correlation_threshold=0.95):
    """
    Analyze predictor correlations and suggest optimizations.
    
    Args:
        csv_path: Path to the statistics CSV file
        correlation_threshold: Threshold for considering predictors redundant (default 0.95)
    """
    # Read the CSV file
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: Could not find {csv_path}")
        print("Make sure to run cmix first to generate the statistics file.")
        sys.exit(1)
    
    # Get predictor columns (exclude bit_position)
    predictor_cols = [col for col in df.columns if col != 'bit_position']
    
    # Calculate correlation matrix
    print(f"Analyzing {len(predictor_cols)} predictors from {len(df)} snapshots...")
    print("=" * 80)
    
    corr_matrix = df[predictor_cols].corr()
    
    # Find highly correlated pairs (potential redundancies)
    print(f"\n🔴 HIGHLY CORRELATED PREDICTORS (correlation >= {correlation_threshold}):")
    print("=" * 80)
    print("These predictors produce very similar outputs and may be redundant.\n")
    
    high_corr_pairs = []
    for i in range(len(predictor_cols)):
        for j in range(i + 1, len(predictor_cols)):
            corr = corr_matrix.iloc[i, j]
            if corr >= correlation_threshold:
                pred1 = predictor_cols[i]
                pred2 = predictor_cols[j]
                high_corr_pairs.append((pred1, pred2, corr))
                print(f"  {pred1:20s} <-> {pred2:20s} : {corr:.4f}")
    
    if not high_corr_pairs:
        print("  No highly correlated pairs found.")
    
    # Find low correlation pairs (most independent)
    print(f"\n🟢 LOW CORRELATION PREDICTORS (most independent):")
    print("=" * 80)
    print("These predictors contribute unique information.\n")
    
    low_corr_pairs = []
    for i in range(len(predictor_cols)):
        for j in range(i + 1, len(predictor_cols)):
            corr = corr_matrix.iloc[i, j]
            low_corr_pairs.append((predictor_cols[i], predictor_cols[j], corr))
    
    # Sort by correlation and show top 10 most independent pairs
    low_corr_pairs.sort(key=lambda x: abs(x[2]))
    for pred1, pred2, corr in low_corr_pairs[:10]:
        print(f"  {pred1:20s} <-> {pred2:20s} : {corr:.4f}")
    
    # Analyze predictor performance
    print(f"\n📊 PREDICTOR PERFORMANCE ANALYSIS:")
    print("=" * 80)
    print(f"{'Predictor':<20s} {'Mean Accuracy':>13s} {'Std Dev':>10s} {'Avg Correlation':>18s}")
    print("-" * 80)
    
    predictor_stats = []
    for pred in predictor_cols:
        mean_acc = df[pred].mean()
        std_acc = df[pred].std()
        # Average correlation with all other predictors
        avg_corr = corr_matrix[pred].drop(pred).mean()
        predictor_stats.append((pred, mean_acc, std_acc, avg_corr))
        print(f"{pred:<20s} {mean_acc:>12.2f}% {std_acc:>9.2f}% {avg_corr:>17.4f}")
    
    # Identify redundant predictors for elimination
    print(f"\n💡 OPTIMIZATION RECOMMENDATIONS:")
    print("=" * 80)
    
    if high_corr_pairs:
        print("\nRedundant predictor groups (consider eliminating some):\n")
        
        # Build redundancy groups
        redundancy_groups = []
        processed = set()
        
        for pred1, pred2, corr in high_corr_pairs:
            if pred1 in processed and pred2 in processed:
                continue
            
            # Find all predictors highly correlated with pred1 or pred2
            group = {pred1, pred2}
            for p1, p2, c in high_corr_pairs:
                if p1 in group or p2 in group:
                    group.add(p1)
                    group.add(p2)
            
            if group not in redundancy_groups:
                redundancy_groups.append(group)
                processed.update(group)
        
        for idx, group in enumerate(redundancy_groups, 1):
            group_list = sorted(list(group))
            print(f"  Group {idx}: {', '.join(group_list)}")
            
            # Recommend which one to keep (highest accuracy or lowest correlation with others)
            group_perfs = [(p, df[p].mean(), corr_matrix[p].drop(group_list).mean()) 
                          for p in group_list]
            group_perfs.sort(key=lambda x: (-x[1], abs(x[2])))  # Sort by accuracy desc, then correlation
            
            keep = group_perfs[0][0]
            eliminate = [p[0] for p in group_perfs[1:]]
            
            print(f"    → KEEP: {keep} (accuracy: {group_perfs[0][1]:.2f}%)")
            print(f"    → ELIMINATE: {', '.join(eliminate)}")
            print()
        
        # Calculate potential CPU savings
        total_predictors = len(predictor_cols)
        redundant_predictors = len(processed) - len(redundancy_groups)
        if redundant_predictors > 0:
            savings_pct = (redundant_predictors / total_predictors) * 100
            print(f"Potential CPU savings: {redundant_predictors}/{total_predictors} predictors ({savings_pct:.1f}%)")
    else:
        print("\nNo highly redundant predictors found at this threshold.")
        print("All predictors appear to contribute unique information.")
    
    # Find predictors with low accuracy and high correlation (worst candidates)
    print(f"\n⚠️  WEAKEST PREDICTORS (low accuracy + high redundancy):")
    print("=" * 80)
    
    weak_predictors = []
    for pred, mean_acc, std_acc, avg_corr in predictor_stats:
        if mean_acc < 60 and avg_corr > 0.9:  # Low accuracy and highly correlated
            weak_predictors.append((pred, mean_acc, avg_corr))
    
    if weak_predictors:
        weak_predictors.sort(key=lambda x: (x[1], -x[2]))  # Sort by accuracy ascending
        for pred, acc, corr in weak_predictors:
            print(f"  {pred:20s} : accuracy={acc:5.2f}%, avg_correlation={corr:.4f}")
        print(f"\nThese {len(weak_predictors)} predictors are prime candidates for elimination.")
    else:
        print("  No obvious weak predictors found.")
    
    print("\n" + "=" * 80)
    print("Analysis complete!")

if __name__ == '__main__':
    csv_file = sys.argv[1] if len(sys.argv) > 1 else 'predictor_stats.csv'
    threshold = float(sys.argv[2]) if len(sys.argv) > 2 else 0.95
    
    print(f"Reading statistics from: {csv_file}")
    print(f"Correlation threshold: {threshold}")
    print()
    analyze_correlations(csv_file, threshold)

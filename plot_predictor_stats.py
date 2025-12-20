#!/usr/bin/env python3
"""
Visualize predictor statistics from cmix run.
Reads predictor_stats.csv and plots accuracy over file progress.
"""

import pandas as pd
import matplotlib.pyplot as plt
import sys
from pathlib import Path

def plot_predictor_stats(csv_path='predictor_stats.csv', output_path='predictor_accuracy.png'):
    """
    Plot predictor accuracy over file progression.
    
    Args:
        csv_path: Path to the statistics CSV file
        output_path: Path to save the output plot
    """
    # Read the CSV file
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: Could not find {csv_path}")
        print("Make sure to run cmix first to generate the statistics file.")
        sys.exit(1)
    
    # Calculate file_percentage based on row index
    df['file_percentage'] = (df.index + 1) / len(df) * 100.0
    
    # Get predictor columns (all except bit_position and file_percentage)
    predictor_cols = [col for col in df.columns if col not in ['bit_position', 'file_percentage']]
    
    # Create figure with dark theme
    plt.style.use('dark_background')
    plt.figure(figsize=(14, 8), dpi=100)
    
    # Use multiple colormaps to get enough distinct colors for all predictors
    n_predictors = len(predictor_cols)
    if n_predictors <= 20:
        colors = plt.cm.tab20(range(n_predictors))
    elif n_predictors <= 40:
        colors = list(plt.cm.tab20(range(20))) + list(plt.cm.tab20b(range(n_predictors - 20)))
    else:
        # For many predictors, use a continuous colormap
        colors = plt.cm.viridis([i / n_predictors for i in range(n_predictors)])
    
    # Reverse colors so first predictor gets last color
    colors = colors[::-1]
    
    # Plot each predictor with transparency, except byte_mixer which gets red
    for idx, predictor in enumerate(predictor_cols):
        if predictor == 'byte_mixer':
            # byte_mixer gets bright red with high opacity
            plt.plot(df['file_percentage'], df[predictor], 
                    label=predictor, color='red', linewidth=1, alpha=0.8)
        else:
            # Other predictors get their assigned colors with transparency
            plt.plot(df['file_percentage'], df[predictor], 
                    label=predictor, color=colors[idx], linewidth=0.5, alpha=1)
    
    # Customize the plot
    plt.xlabel('File Progress (%)', fontsize=12, fontweight='bold', color='white')
    plt.ylabel('Prediction Accuracy (%)', fontsize=12, fontweight='bold', color='white')
    plt.title('Predictor Accuracy During Compression', fontsize=14, fontweight='bold', color='white')
    plt.grid(True, alpha=0.2, linestyle='--', color='gray')
    plt.ylim(0, 100)
    plt.xlim(0, 100)
    
    # Add legend at bottom with multiple columns and thicker lines
    n_cols = min(4, (len(predictor_cols) + 3) // 4)  # Up to 4 columns
    legend = plt.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', 
                       fontsize=8, ncol=n_cols, framealpha=0.9, 
                       columnspacing=1.0, handlelength=2.5)
    # Make legend lines thicker
    for line in legend.get_lines():
        line.set_linewidth(3.0)
    
    # Tight layout with extra space for bottom legend
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    
    # Save the plot
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")
    
    # Display statistics summary
    print("\nPredictor Accuracy Summary:")
    print("=" * 60)
    for predictor in predictor_cols:
        mean_acc = df[predictor].mean()
        std_acc = df[predictor].std()
        min_acc = df[predictor].min()
        max_acc = df[predictor].max()
        print(f"{predictor:20s}: Mean={mean_acc:5.1f}% Std={std_acc:5.1f}% Range=[{min_acc:5.1f}%-{max_acc:5.1f}%]")
    
    # Show the plot
    plt.show()

if __name__ == '__main__':
    csv_file = sys.argv[1] if len(sys.argv) > 1 else 'predictor_stats.csv'
    output_file = sys.argv[2] if len(sys.argv) > 2 else 'predictor_accuracy.png'
    
    print(f"Reading statistics from: {csv_file}")
    plot_predictor_stats(csv_file, output_file)

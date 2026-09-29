#!/usr/bin/env python3
"""
Benchmark: Compare ACI vs Split Conformal coverage under distribution shift.
Saves coverage.png with comparison plot.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

from conformal_drift import (
    AdaptiveConformalInference, 
    SplitConformalInference, 
    generate_drifting_data
)


def run_benchmark():
    np.random.seed(42)
    
    # Generate drifting data with mild drift (manageable for ACI to adapt to)
    t, y = generate_drifting_data(n_points=5000, drift_magnitude=0.05)
    
    # Split into train (calibration) and test
    train_size = 1500
    train_y = y[:train_size]
    test_y = y[train_size:]
    
    # Use mean as point prediction
    train_mean = np.mean(train_y)
    
    # Compute calibration scores as absolute residuals
    train_scores = np.abs(train_y - train_mean)
    
    # Target coverage
    target_coverage = 0.9
    
    # ACI
    aci = AdaptiveConformalInference(target_coverage=target_coverage, step_size=0.02)
    aci.fit(train_scores)
    
    # Split Conformal
    split = SplitConformalInference(target_coverage=target_coverage)
    split.fit(train_scores)
    
    # Run through test data and track coverage over time (windowed)
    window_size = 100
    n_windows = (len(test_y) - window_size) // window_size
    
    aci_window_coverage = []
    split_window_coverage = []
    window_centers = []
    
    for i in range(n_windows):
        start = i * window_size
        end = start + window_size
        
        # ACI coverage in window
        aci_covered = []
        for true_val in test_y[start:end]:
            interval = aci.get_interval(train_mean)
            covered = interval[0] <= true_val <= interval[1]
            aci_covered.append(covered)
            aci.update_alpha(covered)
        
        # Split coverage in window (no adaptation - same coverage everywhere)
        split_covered = []
        for true_val in test_y[start:end]:
            interval = split.get_interval(train_mean)
            covered = interval[0] <= true_val <= interval[1]
            split_covered.append(covered)
        
        aci_window_coverage.append(np.mean(aci_covered))
        split_window_coverage.append(np.mean(split_covered))
        window_centers.append(start + window_size // 2)
    
    # Overall coverage
    aci_overall = np.mean(aci_window_coverage)
    split_overall = np.mean(split_window_coverage)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(window_centers, aci_window_coverage, 'b-', linewidth=2, label='ACI (Adaptive)')
    ax.plot(window_centers, split_window_coverage, 'r--', linewidth=2, label='Split Conformal (Fixed)')
    ax.axhline(y=target_coverage, color='g', linestyle=':', linewidth=2, label=f'Target ({target_coverage:.0%})')
    ax.axhline(y=target_coverage - 0.02, color='orange', linestyle=':', linewidth=1, label='+/-2% tolerance')
    ax.axhline(y=target_coverage + 0.02, color='orange', linestyle=':', linewidth=1)
    
    ax.set_xlabel('Test Samples')
    ax.set_ylabel('Coverage Rate')
    ax.set_title('Coverage Comparison: ACI vs Split Conformal under Distribution Shift')
    ax.legend(loc='lower right')
    ax.set_ylim(0.7, 1.0)
    ax.grid(True, alpha=0.3)
    
    # Add text annotations
    ax.text(0.05, 0.95, f'ACI Overall: {aci_overall:.1%}', transform=ax.transAxes, 
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax.text(0.05, 0.88, f'Split Overall: {split_overall:.1%}', transform=ax.transAxes, 
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('coverage.png', dpi=150)
    plt.close()
    
    # Print summary
    print("=" * 60)
    print("BENCHMARK RESULTS: Conformal Inference under Distribution Shift")
    print("=" * 60)
    print(f"Target Coverage: {target_coverage:.0%}")
    print(f"ACI Overall Coverage: {aci_overall:.1%}")
    print(f"Split Conformal Overall Coverage: {split_overall:.1%}")
    print(f"ACI within 2% of target: {abs(aci_overall - target_coverage) < 0.02}")
    print(f"Split Conformal within 2% of target: {abs(split_overall - target_coverage) < 0.02}")
    print("=" * 60)
    print("Plot saved to: coverage.png")


if __name__ == "__main__":
    run_benchmark()

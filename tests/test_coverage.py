import numpy as np
from conformal_drift import (
    AdaptiveConformalInference, 
    SplitConformalInference, 
    generate_drifting_data
)


def test_aci_coverage_drifting_data():
    """
    Test that ACI maintains long-run coverage within 2% of target on drifting data.
    ACI adapts alpha online to maintain coverage under drift.
    """
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
    
    # ACI: target coverage 90%, should stay within 2% (88-92%)
    aci = AdaptiveConformalInference(target_coverage=0.9, step_size=0.02)
    aci.fit(train_scores)
    
    # Track ACI coverage
    aci_covered = []
    for true_val in test_y:
        # Use train mean as prediction
        pred = train_mean
        interval = aci.get_interval(pred)
        covered = interval[0] <= true_val <= interval[1]
        aci_covered.append(covered)
        aci.update_alpha(covered)
    
    aci_coverage = np.mean(aci_covered)
    target_coverage = 0.9
    
    # ACI should maintain coverage within 2% of target
    assert abs(aci_coverage - target_coverage) < 0.02, \
        f"ACI coverage {aci_coverage:.3f} not within 2% of target {target_coverage}"
    
    print(f"ACI Coverage: {aci_coverage:.3%} (target: {target_coverage:.0%})")


def test_split_conformal_coverage_drifting_data():
    """
    Test that plain split conformal does NOT maintain coverage under drift.
    Split conformal uses fixed alpha trained on calibration data only.
    """
    np.random.seed(42)
    
    # Generate drifting data with same drift
    t, y = generate_drifting_data(n_points=5000, drift_magnitude=0.05)
    
    # Split into train (calibration) and test
    train_size = 1500
    train_y = y[:train_size]
    test_y = y[train_size:]
    
    # Use mean as point prediction
    train_mean = np.mean(train_y)
    
    # Compute calibration scores as absolute residuals
    train_scores = np.abs(train_y - train_mean)
    
    # Split Conformal: fixed alpha, no adaptation
    split = SplitConformalInference(target_coverage=0.9)
    split.fit(train_scores)
    
    # Track split conformal coverage
    split_covered = []
    for true_val in test_y:
        pred = train_mean
        interval = split.get_interval(pred)
        covered = interval[0] <= true_val <= interval[1]
        split_covered.append(covered)
    
    split_coverage = np.mean(split_covered)
    target_coverage = 0.9
    
    # Split conformal should NOT maintain coverage within 2% of target under drift
    assert abs(split_coverage - target_coverage) >= 0.02, \
        f"Split Conformal coverage {split_coverage:.3f} is within 2% of target - this is unexpected under drift"
    
    print(f"Split Conformal Coverage: {split_coverage:.3%} (target: {target_coverage:.0%})")


if __name__ == "__main__":
    test_aci_coverage_drifting_data()
    test_split_conformal_coverage_drifting_data()

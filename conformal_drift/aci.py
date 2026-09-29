import numpy as np

class AdaptiveConformalInference:
    """
    Adaptive Conformal Inference (ACI) - Gibbs & Candes 2021.
    Updates alpha_t online under distribution shift to maintain target coverage.
    """
    def __init__(self, target_coverage=0.9, step_size=0.05):
        self.target_coverage = target_coverage
        self.alpha = 1.0 - target_coverage  # Initial alpha
        self.step_size = step_size
        self.calibration_scores = []
        self.n_calib = 0

    def fit(self, scores):
        """Fit using calibration scores (absolute residuals from calibration data)."""
        self.calibration_scores = np.array(scores)
        self.n_calib = len(self.calibration_scores)

    def get_interval(self, pred):
        """
        Compute prediction interval based on current alpha and calibration scores.
        Returns (lower, upper) bounds such that P(y in [L, U]) >= 1 - alpha.
        The interval is around the point prediction `pred`.
        """
        if self.n_calib == 0:
            raise ValueError("Model not fitted. Call fit() with calibration scores first.")
        
        # Edge cases for alpha
        if self.alpha <= 0:
            # Infinite interval: return pred ± infinity
            return (-np.inf, np.inf)
        elif self.alpha >= 1:
            # Empty-width interval: return pred ± 0
            return (pred, pred)
        
        # Quantile adjustment for finite sample: (n+1)(1-alpha)/n
        q_level = np.ceil((self.n_calib + 1) * (1 - self.alpha)) / self.n_calib
        q_level = min(q_level, 1.0)
        
        threshold = np.quantile(self.calibration_scores, q_level)
        
        # Interval: [pred - threshold, pred + threshold]
        return (pred - threshold, pred + threshold)

    def update_alpha(self, covered):
        """
        Update alpha based on whether the true value was covered.
        This is the core ACI mechanism to adapt to drift.
        """
        # err = 1 if missed, 0 if covered
        err = 0.0 if covered else 1.0
        target_alpha = 1.0 - self.target_coverage
        
        # Gibbs & Candes 2021 rule:
        # alpha_{t+1} = alpha_t + step_size * (target_alpha - err_t)
        self.alpha = self.alpha + self.step_size * (target_alpha - err)


def generate_drifting_data(n_points=1000, drift_magnitude=0.5):
    """
    Generate synthetic time-series data with distribution shift (drift).
    Mean shifts linearly over time to simulate concept drift.
    """
    t = np.linspace(0, 10, n_points)
    true_mean = 5.0 + drift_magnitude * t  # Linear drift
    true_std = 1.0 + 0.1 * t  # Slight variance drift
    noise = np.random.normal(0, true_std)
    y = true_mean + noise
    return t, y

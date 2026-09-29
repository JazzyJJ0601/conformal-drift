import numpy as np

class SplitConformalInference:
    """
    Split Conformal Inference - Fixed alpha, no adaptation.
    Used as a baseline to compare against ACI under distribution shift.
    """
    def __init__(self, target_coverage=0.9):
        self.target_coverage = target_coverage
        self.alpha = 1.0 - target_coverage
        self.calibration_scores = []
        self.n_calib = 0

    def fit(self, scores):
        """Fit using calibration scores."""
        self.calibration_scores = np.array(scores)
        self.n_calib = len(self.calibration_scores)

    def get_interval(self, score_pred):
        """
        Compute interval based on fixed alpha and calibration scores.
        Returns (lower, upper) bounds.
        """
        if self.n_calib == 0:
            raise ValueError("Model not fitted. Call fit() with calibration scores first.")
        
        # Quantile adjustment for finite sample
        q_level = np.ceil((self.n_calib + 1) * (1 - self.alpha)) / self.n_calib
        q_level = min(q_level, 1.0)
        
        threshold = np.quantile(self.calibration_scores, q_level)
        
        # Interval: [score_pred - threshold, score_pred + threshold]
        return (score_pred - threshold, score_pred + threshold)

    def predict(self, score_pred):
        """Return whether prediction is covered (for benchmarking)."""
        lower, upper = self.get_interval(score_pred)
        return (lower, upper)

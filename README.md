# conformal-drift

Prediction intervals that keep their promised coverage when the data drifts.
A from-scratch NumPy implementation of Adaptive Conformal Inference (ACI, Gibbs & Candès 2021), compared with ordinary split conformal prediction on a drifting stream.

## The idea

Split conformal prediction takes a calibration set, finds the residual size that 90% of points stay under, and uses it as a fixed interval half-width. That guarantee assumes future data looks like the calibration data. When the data drifts, coverage quietly falls.

ACI keeps a running miscoverage level α<sub>t</sub> and nudges it after every point:

α<sub>t+1</sub> = α<sub>t</sub> + γ (α − err<sub>t</sub>), where err<sub>t</sub> = 1 if the point fell outside the interval, else 0.

A miss lowers α<sub>t</sub>, so the next interval uses a higher quantile of the calibration residuals and is wider; a hit raises it slightly. Over time the miss rate is pulled to α. If α<sub>t</sub> drops to 0 or below, the interval is infinite.

## Result

`python benchmark.py` (one seed) and `python results/seeds.py` (5 seeds, `results/seeds.json`).
Synthetic stream: 5,000 points whose mean rises linearly (slope 0.05 per unit time, plus a slowly growing spread); first 1,500 points calibrate, the next 3,500 are scored. The point prediction is the calibration mean for both methods. Target coverage 90%, ACI step γ = 0.02.

| | Coverage (mean of 5 seeds) | Median interval width | Steps with an infinite interval |
|---|---|---|---|
| Split conformal | 74.7% | 3.78 | 0 |
| **ACI** | **89.9%** | 5.22 | 336 of 3,500 (9.6%) |

ACI hits the 90% target; split conformal falls 15 points short. The price is wider intervals, and on about one step in ten an infinite one: the point prediction never moves while the data drifts away from it, so the only way left to cover the point is to widen the interval.

## Limits

- Synthetic data with one simple kind of drift (a linear trend). No real dataset yet.
- A constant point prediction on purpose, to isolate the interval method. With a model that tracks the trend, the intervals would be far narrower; that comparison is the obvious next step.
- ACI guarantees long-run average coverage, not coverage in every window.

## Use

```python
import numpy as np
from conformal_drift import AdaptiveConformalInference

aci = AdaptiveConformalInference(target_coverage=0.9, step_size=0.02)
aci.fit(np.abs(calib_y - calib_pred))       # calibration residuals
for pred, y in stream:
    lo, hi = aci.get_interval(pred)
    aci.update_alpha(lo <= y <= hi)
```

`SplitConformalInference` has the same `fit` / `get_interval` interface with a fixed α.

```bash
pip install -e . && pip install matplotlib pytest
python benchmark.py          # one seed, writes coverage.png
python results/seeds.py      # 5 seeds, writes results/seeds.json
pytest -q tests
```

## References

1. Gibbs, I. & Candès, E. (2021). Adaptive Conformal Inference Under Distribution Shift. NeurIPS 2021.
2. Angelopoulos, A. N. & Bates, S. (2021). A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification. arXiv:2107.07511.
3. Vovk, V., Gammerman, A. & Shafer, G. (2005). Algorithmic Learning in a Random World. Springer.

"""Coverage and interval width of ACI vs split conformal over 5 seeds (same setup as benchmark.py)."""
import json
import numpy as np
from conformal_drift import AdaptiveConformalInference, SplitConformalInference, generate_drifting_data

rows = []
for seed in range(5):
    np.random.seed(seed)
    _, y = generate_drifting_data(n_points=5000, drift_magnitude=0.05)
    tr, te = y[:1500], y[1500:]
    mu = tr.mean()
    scores = np.abs(tr - mu)
    aci = AdaptiveConformalInference(0.9, step_size=0.02); aci.fit(scores)
    spl = SplitConformalInference(0.9); spl.fit(scores)
    a_cov, a_w, n_inf, s_cov, s_w = [], [], 0, [], []
    for v in te:
        lo, hi = aci.get_interval(mu)
        c = lo <= v <= hi
        a_cov.append(c); aci.update_alpha(c)
        if np.isinf(hi):
            n_inf += 1
        else:
            a_w.append(hi - lo)
        lo, hi = spl.get_interval(mu)
        s_cov.append(lo <= v <= hi); s_w.append(hi - lo)
    rows.append({"seed": seed, "aci_coverage": float(np.mean(a_cov)), "split_coverage": float(np.mean(s_cov)),
                 "aci_median_width": float(np.median(a_w)), "split_width": float(s_w[0]), "aci_infinite_steps": n_inf,
                 "steps": len(te)})
summary = {k: float(np.mean([r[k] for r in rows])) for k in rows[0] if k not in ("seed", "steps")}
json.dump({"runs": rows, "mean": summary}, open("results/seeds.json", "w"), indent=1)
print(json.dumps(summary, indent=1))

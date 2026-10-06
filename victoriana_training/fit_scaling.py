#!/usr/bin/env python3
"""
fit_scaling.py — fit a compute-optimal scaling law from the probe runs and extrapolate the ~2.7B
final. Each probe trains at its own Chinchilla-optimal budget (D ≈ 20·N), so the (N, final_val_loss)
points lie on the compute-optimal frontier; we fit L(N) = E + A·N^(-alpha) and predict N=2.7B.

Usage: python fit_scaling.py logs/train_345M.log logs/train_500M.log logs/train_1.2B.log
"""
import re, sys, warnings
import numpy as np
warnings.filterwarnings("ignore")   # quiet scipy's covariance note on a 3-point fit

FINAL_N = 2.705e9   # the 2.7B target (llama_probe_configs FINAL)

def parse(path):
    txt = open(path).read()
    params = re.search(r"\|\s*([\d.]+)M params", txt)
    loss = re.findall(r"final val_loss\s+([\d.]+)", txt)
    toks = re.search(r"budget\s+([\d.]+)B tok", txt)
    if not (params and loss):
        raise SystemExit(f"could not parse params/final-loss from {path}")
    return float(params.group(1)) * 1e6, float(loss[-1]), float(toks.group(1)) if toks else None

def main():
    pts = [parse(p) for p in sys.argv[1:]]
    pts.sort()
    Ns = np.array([n for n, _, _ in pts])
    Ls = np.array([l for _, l, _ in pts])
    print(f"{'params':>10}{'tokens':>10}{'val_loss':>10}")
    for n, l, t in pts:
        print(f"{n/1e6:>9.0f}M{(str(t)+'B' if t else '?'):>10}{l:>10.4f}")

    # Fit L = E + A * N^(-alpha). 3 unknowns, 3+ points.
    try:
        from scipy.optimize import curve_fit
        f = lambda N, E, A, a: E + A * np.power(N, -a)
        # reasonable inits/bounds: irreducible loss E in [0, min(L)], positive A, alpha in (0,1)
        (E, A, a), _ = curve_fit(f, Ns, Ls, p0=[min(Ls) * 0.7, 10.0, 0.1],
                                 bounds=([0, 0, 0.01], [min(Ls), 1e6, 1.0]), maxfev=100000)
        pred = f(FINAL_N, E, A, a)
        print(f"\nfit  L(N) = {E:.3f} + {A:.3g}·N^(-{a:.3f})   (irreducible loss E≈{E:.3f})")
        print(f"→ predicted val_loss at {FINAL_N/1e9:.2f}B params: {pred:.4f}  (ppl ≈ {np.exp(pred):.2f})")
        # marginal gain from the largest probe to the 2.7B
        print(f"  Δloss {Ls[-1]:.4f}→{pred:.4f} = {Ls[-1]-pred:.4f} going {Ns[-1]/1e9:.2f}B→{FINAL_N/1e9:.2f}B params")
        print("  (small Δ ⇒ diminishing returns near the optimal — a leaner final size loses little.)")
    except Exception as e:
        # fallback: log-log power-law slope (ignores irreducible term)
        a, b = np.polyfit(np.log(Ns), np.log(Ls), 1)
        pred = np.exp(b) * FINAL_N ** a
        print(f"\n[scipy unavailable: {e}] log-log slope {a:.3f} → predicted loss {pred:.4f}")

if __name__ == "__main__":
    main()

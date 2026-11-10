"""
INTEGRITY CODE SERIES -- Week 8
Design and hot-spot cases, 2,000-sample LHS sweep, surrogate, every figure in assets/.

    python run_all.py
"""

import os

import numpy as np

from src import config, creep_fatigue
from src.monte_carlo import run_sweep, sensitivity_analysis
from src.surrogate import TubeLifeSurrogate
from src.tube_model import simulate_tube_life
from src.visualization import plot_all

HOT_SPOT_K = 40.0


def report_case(name, r):
    D_f, D_c = r.D_fatigue[-1], r.D_creep[-1]
    limit = D_c + creep_fatigue.envelope_margin(D_f, D_c)
    print(f"{name}: {r.failure_mode} at {r.failure_time_hours:,.0f} h, D_c={D_c:.3f}, "
          f"D_f={D_f:.4f}, D_c limit={limit:.3f} ({limit / D_c:.1f}x)")


def main():
    design = simulate_tube_life()
    hot = simulate_tube_life(T=config.DESIGN_TEMPERATURE + HOT_SPOT_K)
    omega = simulate_tube_life(use_omega=True)
    report_case("Design", design)
    report_case("Hot spot +40 C", hot)
    print(f"Creep damage ratio: {hot.D_creep[-1] / design.D_creep[-1]:.1f}x")
    print(f"Omega branch: {omega.failure_mode} at {omega.failure_time_hours:,.0f} h")

    print(f"Running LHS sweep ({config.MC_N_SAMPLES:,} samples)...")
    X, res = run_sweep()
    life = res["life_hours"]
    modes, counts = np.unique(res["failure_mode"], return_counts=True)
    print(dict(zip(modes.tolist(), counts.tolist())))
    corr = sensitivity_analysis(X, life)
    for k, v in corr.items():
        print(f"  {k}: {v:+.2f}")

    # Hold out the last 20% for the parity plot; LHS rows are in random order
    n_fit = int(0.8 * len(X))
    surrogate = TubeLifeSurrogate()
    surrogate.fit(X[:n_fit], life[:n_fit], res["D_f_final"][:n_fit], res["D_c_final"][:n_fit])
    y_true, y_pred, r2, _ = surrogate.parity_data(X[n_fit:], life[n_fit:])
    print(f"Surrogate held-out R2 (log life): {r2:.3f}")

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
    plot_all(out, design, omega, hot, life, corr, (y_true, y_pred, r2),
             surrogate.feature_importance())
    print(f"Figures written to {out}")


if __name__ == "__main__":
    main()

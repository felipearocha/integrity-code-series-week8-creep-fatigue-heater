"""
Figures for the Week 8 package. Every panel is drawn from computed results.
run_all.py calls plot_all(); figures are written to assets/.
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

from . import config, creep_fatigue, fatigue, oxidation
from .tube_model import simulate_tube_life

NAVY, RED, TEAL, GREY = "#1a3a5c", "#8b2318", "#2a7d7a", "#777777"
LABELS = ["Temperature", "Pressure", "Wall thickness", "Cycles/year", "Thermal cycle"]


def _style():
    plt.rcParams.update({
        "font.size": 10, "axes.grid": True, "grid.alpha": 0.25, "axes.titleweight": "bold",
        "savefig.dpi": 200, "savefig.bbox": "tight",
    })


def _save(fig, out, name):
    os.makedirs(out, exist_ok=True)
    fig.savefig(os.path.join(out, name))
    plt.close(fig)


def _envelope(ax):
    df, dc = creep_fatigue.interaction_envelope()
    ax.fill_between(df, dc, color=TEAL, alpha=0.12)
    ax.plot(df, dc, color=TEAL, lw=1.8, label="Gr91 bilinear envelope")


def life_grid(n=25):
    """Life over temperature and pressure at nominal wall, full simulation per point."""
    T = np.linspace(*config.MC_RANGES["temperature_K"], n)
    P = np.linspace(*config.MC_RANGES["pressure_Pa"], n)
    life = np.array([[simulate_tube_life(T=t, pressure=p, dt_hours=200).failure_time_hours
                      for t in T] for p in P])
    return T - 273.15, P / 1e6, life


def plot_hero(norton, omega, grid, out):
    """Stress and wall, Norton vs Omega strain, damage path, life map."""
    fig, axs = plt.subplots(2, 2, figsize=(11, 8.5), constrained_layout=True)
    a = axs[0, 0]
    a.plot(norton.times_hours / 1e3, norton.hoop_stress / 1e6, color=NAVY)
    a.set(xlabel="Time [1000 h]", ylabel="Hoop stress [MPa]", title="(a) Wall thinning feedback")
    b = a.twinx()
    b.plot(norton.times_hours / 1e3, norton.wall_thickness * 1e3, "--", color=TEAL)
    b.set_ylabel("Wall thickness [mm]")
    b.grid(False)

    a = axs[0, 1]
    a.plot(norton.times_hours / 1e3, norton.creep_strain * 100, color=NAVY, label="Norton")
    a.plot(omega.times_hours / 1e3, omega.creep_strain * 100, "--", color=RED,
           label="Omega ([ASSUMED] constants)")
    a.set(xlabel="Time [1000 h]", ylabel="Creep strain [%]", title="(b) Creep strain")
    a.legend()

    a = axs[1, 0]
    _envelope(a)
    a.plot(norton.D_fatigue, norton.D_creep, color=NAVY, label="Design trajectory")
    a.plot(norton.D_fatigue[-1], norton.D_creep[-1], "o", color=NAVY)
    a.set(xlabel="Fatigue damage $D_f$", ylabel="Creep damage $D_c$",
          title="(c) Interaction diagram", xlim=(0, 0.2), ylim=(0, 1.05))
    a.legend()

    a = axs[1, 1]
    T_C, P_MPa, life = grid
    c = a.contourf(T_C, P_MPa, life / 1e3, levels=np.linspace(0, 100, 11), cmap="RdYlGn")
    a.plot(config.DESIGN_TEMPERATURE - 273.15, config.INTERNAL_PRESSURE / 1e6, "X",
           color="k", ms=11, label="Design point")
    a.set(xlabel="Temperature [°C]", ylabel="Pressure [MPa]", title="(d) Tube life")
    a.grid(False)
    a.legend(loc="lower left")
    fig.colorbar(c, ax=a, label="Life [1000 h], capped at the 100,000 h run")
    _save(fig, out, "hero_creep_fatigue_week8.png")


def plot_fatigue_curve(out):
    nf, tot, el, pl = fatigue.fatigue_life_curve()
    eps = fatigue.thermal_strain_range(config.DELTA_T_CYCLE)
    fig, a = plt.subplots(figsize=(7, 5))
    a.loglog(nf, tot, color=NAVY, label="Total")
    a.loglog(nf, el, "--", color=TEAL, label="Elastic term")
    a.loglog(nf, pl, "--", color=RED, label="Plastic term")
    a.plot(fatigue.coffin_manson_life(eps), eps, "o", color="k",
           label=f"Design cycle ($\\Delta T$ = {config.DELTA_T_CYCLE:.0f} K)")
    a.set(xlabel="Cycles to failure $N_f$", ylabel="Strain range $\\Delta\\varepsilon$",
          title="Strain-life curve")
    a.legend()
    _save(fig, out, "fatigue_curve.png")


def plot_oxide(out):
    t = np.linspace(0, config.SIM_TOTAL_HOURS, 200)
    ox = oxidation.oxide_thickness_profile(t * 3600.0, config.DESIGN_TEMPERATURE)
    loss = 100 * ox * config.OXIDE_METAL_LOSS_FRACTION / config.TUBE_WT_NOMINAL
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4))
    a.plot(t / 1e3, ox * 1e6, color=RED)
    a.set(xlabel="Time [1000 h]", ylabel="Oxide thickness [µm]", title="Oxide scale growth")
    b.plot(t / 1e3, loss, color=NAVY)
    b.axhline(100 * config.SIM_MAX_WALL_LOSS_FRACTION, ls="--", color=RED,
              label="Retirement criterion")
    b.set(xlabel="Time [1000 h]", ylabel="Wall loss [%]", title="Effective wall loss")
    b.legend()
    _save(fig, out, "oxide_wall_loss.png")


def plot_life_distribution(life_hours, out):
    failed = life_hours[life_hours < config.SIM_TOTAL_HOURS] / 1e3
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4))
    a.hist(failed, bins=25, color=NAVY, alpha=0.75)
    a.set(xlabel="Life of failed samples [1000 h]", ylabel="Samples",
          title=f"Failed samples: {failed.size} of {life_hours.size}")
    x = np.sort(life_hours) / 1e3
    b.step(x, np.arange(1, x.size + 1) / x.size, color=NAVY)
    b.set(xlabel="Life [1000 h], survivors censored at 100", ylabel="Cumulative probability",
          title="Tube life CDF")
    _save(fig, out, "life_distribution.png")


def plot_tornado(corr, out):
    v = np.array(list(corr.values()))
    order = np.argsort(np.abs(v))
    fig, a = plt.subplots(figsize=(8, 4.5))
    a.barh([LABELS[i] for i in order], v[order],
           color=[RED if v[i] < 0 else NAVY for i in order])
    a.axvline(0, color="k", lw=0.8)
    a.set(xlabel="Spearman rank correlation with tube life [-]", title="Sensitivity: tube life")
    _save(fig, out, "sensitivity_tornado.png")


def plot_surrogate(y_true, y_pred, r2, importance, out):
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.5))
    a.scatter(y_true / 1e3, y_pred / 1e3, s=8, color=NAVY, alpha=0.5)
    top = max(y_true.max(), y_pred.max()) / 1e3
    a.plot([0, top], [0, top], "--", color=GREY)
    a.set(xlabel="Simulated life [1000 h]", ylabel="Predicted life [1000 h]",
          title=f"Held-out parity, $R^2$ on log life = {r2:.3f}")
    v = np.array(list(importance.values()))
    order = np.argsort(v)
    b.barh([LABELS[i] for i in order], v[order], color=NAVY)
    b.set(xlabel="Relative importance [-]", title="Feature importance (life model)")
    _save(fig, out, "surrogate_parity_importance.png")


def plot_heatmap(out, n=29):
    T = np.linspace(*config.MC_RANGES["temperature_K"], n)
    t = np.arange(0, config.SIM_TOTAL_HOURS + 1, 1000.0)
    D = np.full((n, t.size), np.nan)
    for i, Ti in enumerate(T):
        r = simulate_tube_life(T=Ti, dt_hours=1000)
        D[i, :r.D_creep.size] = r.D_creep
    fig, a = plt.subplots(figsize=(9, 5.5))
    c = a.contourf(t / 1e3, T - 273.15, D, levels=np.linspace(0, 1, 21), cmap="YlOrRd",
                   extend="max")
    a.set(xlabel="Time [1000 h]", ylabel="Temperature [°C]", title="Creep damage fraction")
    a.grid(False)
    fig.colorbar(c, ax=a, label="$D_c$ [-] (blank after failure)")
    _save(fig, out, "spatiotemporal_heatmap.png")


def plot_gif(design, hot, out, frames=50):
    """Damage paths of the design and hot-spot cases against the envelope."""
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 4.5), constrained_layout=True)
    _envelope(a)
    a.set(xlabel="$D_f$", ylabel="$D_c$", xlim=(0, 0.15), ylim=(0, 1.05))
    b.set(xlabel="Time [1000 h]", ylabel="$D_c$", xlim=(0, 100), ylim=(0, 1.05))
    runs = [(design, NAVY, "Design 580 °C"), (hot, RED, "Hot spot 620 °C")]
    paths = [a.plot([], [], color=c, label=lab)[0] for _, c, lab in runs]
    lines = [b.plot([], [], color=c)[0] for _, c, _ in runs]
    a.legend(loc="upper right")

    def update(k):
        for (r, _, _), p, ln in zip(runs, paths, lines):
            m = max(2, int(r.times_hours.size * (k + 1) / frames))
            p.set_data(r.D_fatigue[:m], r.D_creep[:m])
            ln.set_data(r.times_hours[:m] / 1e3, r.D_creep[:m])
        fig.suptitle(f"t = {(k + 1) * 100 / frames:.0f},000 h")
        return paths + lines

    os.makedirs(out, exist_ok=True)
    FuncAnimation(fig, update, frames=frames).save(
        os.path.join(out, "creep_fatigue_evolution.gif"), writer=PillowWriter(fps=8), dpi=90)
    plt.close(fig)


def plot_all(out, norton, omega, hot, life_hours, corr, parity, importance):
    """Write every figure in assets/ to the folder `out`."""
    _style()
    plot_hero(norton, omega, life_grid(), out)
    plot_fatigue_curve(out)
    plot_oxide(out)
    plot_life_distribution(life_hours, out)
    plot_tornado(corr, out)
    plot_surrogate(*parity, importance, out)
    plot_heatmap(out)
    plot_gif(norton, hot, out)

"""
Latin Hypercube Sampling (LHS) parametric sweep for tube life uncertainty.

Samples 5 parameters:
  1. Temperature [K]
  2. Internal pressure [Pa]
  3. Wall thickness [m]
  4. Cycles per year
  5. Thermal cycle amplitude Delta_T [K]

Each sample runs the coupled creep-fatigue-oxidation simulation.
"""

import numpy as np
from scipy.stats import qmc
from . import config
from .tube_model import simulate_tube_life


def generate_lhs_samples(n_samples: int = None,
                          seed: int = None) -> np.ndarray:
    """
    Generate LHS sample matrix.

    Returns
    -------
    samples : ndarray, shape (n_samples, 5)
        Columns: [temperature_K, pressure_Pa, wall_thickness_m,
                  cycles_per_year, delta_T_K]
    """
    if n_samples is None:
        n_samples = config.MC_N_SAMPLES
    if seed is None:
        seed = config.MC_RANDOM_SEED

    sampler = qmc.LatinHypercube(d=5, seed=seed)
    unit_samples = sampler.random(n=n_samples)

    ranges = config.MC_RANGES
    lower = np.array([
        ranges["temperature_K"][0],
        ranges["pressure_Pa"][0],
        ranges["wall_thickness_m"][0],
        ranges["cycles_per_year"][0],
        ranges["delta_T_K"][0],
    ])
    upper = np.array([
        ranges["temperature_K"][1],
        ranges["pressure_Pa"][1],
        ranges["wall_thickness_m"][1],
        ranges["cycles_per_year"][1],
        ranges["delta_T_K"][1],
    ])

    samples = qmc.scale(unit_samples, lower, upper)
    return samples


def run_sweep(n_samples: int = None, seed: int = None,
              use_omega: bool = False, dt_hours: float = 200,
              total_hours: float = None):
    """
    Run full LHS parametric sweep.

    Parameters
    ----------
    n_samples : int, optional
    seed : int, optional
    use_omega : bool
        Use Omega method vs Norton.
    dt_hours : float
        Time step (larger than baseline for speed).
    total_hours : float, optional

    Returns
    -------
    samples : ndarray, shape (n_samples, 5)
    results : dict with keys:
        'life_hours' : ndarray
        'failure_mode' : list of str
        'D_f_final' : ndarray
        'D_c_final' : ndarray
        'max_creep_strain' : ndarray
        'max_wall_loss_frac' : ndarray
    """
    if total_hours is None:
        total_hours = config.SIM_TOTAL_HOURS

    samples = generate_lhs_samples(n_samples, seed)
    n = len(samples)

    life_hours = np.zeros(n)
    failure_modes = []
    D_f_final = np.zeros(n)
    D_c_final = np.zeros(n)
    max_creep_strain = np.zeros(n)
    max_wall_loss_frac = np.zeros(n)

    for i in range(n):
        T_i = samples[i, 0]
        P_i = samples[i, 1]
        wt_i = samples[i, 2]
        cyc_i = samples[i, 3]
        dT_i = samples[i, 4]

        res = simulate_tube_life(
            T=T_i,
            pressure=P_i,
            wt_initial=wt_i,
            delta_T=dT_i,
            cycles_per_year=cyc_i,
            total_hours=total_hours,
            dt_hours=dt_hours,
            use_omega=use_omega,
        )

        life_hours[i] = res.failure_time_hours
        failure_modes.append(res.failure_mode)
        D_f_final[i] = res.D_fatigue[-1] if len(res.D_fatigue) > 0 else 0.0
        D_c_final[i] = res.D_creep[-1] if len(res.D_creep) > 0 else 0.0
        max_creep_strain[i] = np.max(res.creep_strain)
        wt_init = wt_i
        wt_min = np.min(res.wall_thickness)
        max_wall_loss_frac[i] = 1.0 - wt_min / wt_init if wt_init > 0 else 1.0

    results = {
        "life_hours": life_hours,
        "failure_mode": failure_modes,
        "D_f_final": D_f_final,
        "D_c_final": D_c_final,
        "max_creep_strain": max_creep_strain,
        "max_wall_loss_frac": max_wall_loss_frac,
    }

    return samples, results


def sensitivity_analysis(samples: np.ndarray,
                          life_hours: np.ndarray):
    """
    Spearman rank correlation sensitivity analysis.

    Parameters
    ----------
    samples : ndarray, shape (n, 5)
    life_hours : ndarray, shape (n,)

    Returns
    -------
    correlations : dict
        feature_name -> Spearman rho
    """
    from scipy.stats import spearmanr

    feature_names = [
        "temperature_K",
        "pressure_Pa",
        "wall_thickness_m",
        "cycles_per_year",
        "delta_T_K",
    ]

    correlations = {}
    for j, name in enumerate(feature_names):
        rho, _ = spearmanr(samples[:, j], life_hours)
        correlations[name] = rho

    return correlations

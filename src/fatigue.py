"""
Thermal fatigue model for fired heater tubes.

Governing equations:
  Thermal strain range: Delta_eps_th = alpha * Delta_T
  Coffin-Manson (total strain amplitude):
    Delta_eps_t / 2 = C1 * (2*Nf)^(-beta1) + C2 * (2*Nf)^(-beta2)
  Simplified (range form):
    Delta_eps_t = C1 * Nf^(-beta1) + C2 * Nf^(-beta2)

Reference: Fournier et al. 2008, Int. J. Fatigue 30, 1797-1812
Material: 9Cr-1Mo-V-Nb (T91) at 500-600 C
"""

import numpy as np
from scipy.optimize import brentq
from . import config


def thermal_strain_range(delta_T: float, cte: float = None) -> float:
    """
    Thermal strain range from temperature swing.

    Parameters
    ----------
    delta_T : float
        Temperature change [K].
    cte : float, optional
        Coefficient of thermal expansion [1/K].

    Returns
    -------
    float
        Thermal strain range (dimensionless).
    """
    if cte is None:
        cte = config.CTE
    return cte * abs(delta_T)


def thermal_stress_range(delta_T: float, E: float = None,
                          cte: float = None) -> float:
    """
    Thermal stress range (fully constrained tube, elastic).

    Delta_sigma = E * alpha * Delta_T

    Parameters
    ----------
    delta_T : float
        Temperature change [K].
    E : float, optional
        Young's modulus [Pa].
    cte : float, optional
        CTE [1/K].

    Returns
    -------
    float
        Stress range [Pa].
    """
    if E is None:
        E = config.YOUNGS_MODULUS
    if cte is None:
        cte = config.CTE
    return E * cte * abs(delta_T)


def coffin_manson_life(delta_eps: float) -> float:
    """
    Fatigue life from Coffin-Manson relation for T91 at high temperature.

    Solves: delta_eps = C1 * Nf^(-beta1) + C2 * Nf^(-beta2)

    Parameters
    ----------
    delta_eps : float
        Total strain range (dimensionless). Must be > 0.

    Returns
    -------
    float
        Fatigue life Nf [cycles].
    """
    if delta_eps <= 0.0:
        return np.inf

    def residual(log_nf):
        nf = 10.0 ** log_nf
        eps_calc = (config.FATIGUE_C1 * nf ** (-config.FATIGUE_BETA1)
                    + config.FATIGUE_C2 * nf ** (-config.FATIGUE_BETA2))
        return eps_calc - delta_eps

    # Search between 1 and 1e8 cycles
    try:
        log_nf = brentq(residual, 0.0, 8.0)
        return 10.0 ** log_nf
    except ValueError:
        # If strain range is too small for even 1e8 cycles, return inf
        if residual(8.0) > 0:
            return np.inf
        # If strain range is too large for even 1 cycle
        return 1.0


def strain_range_at_life(nf: float) -> float:
    """
    Compute strain range for a given fatigue life.

    Parameters
    ----------
    nf : float
        Number of cycles to failure.

    Returns
    -------
    float
        Total strain range (dimensionless).
    """
    if nf <= 0:
        return np.inf
    return (config.FATIGUE_C1 * nf ** (-config.FATIGUE_BETA1)
            + config.FATIGUE_C2 * nf ** (-config.FATIGUE_BETA2))


def fatigue_damage_fraction(n_cycles: float, delta_eps: float) -> float:
    """
    Fatigue damage fraction for creep-fatigue interaction.

    D_f = n / N_f

    Parameters
    ----------
    n_cycles : float
        Number of applied cycles.
    delta_eps : float
        Total strain range per cycle.

    Returns
    -------
    float
        Fatigue damage fraction (0 to 1+).
    """
    nf = coffin_manson_life(delta_eps)
    if nf == np.inf or nf <= 0:
        return 0.0
    return n_cycles / nf


def fatigue_life_curve(n_points: int = 100):
    """
    Generate strain-life (epsilon-N) curve for plotting.

    Returns
    -------
    nf_array : ndarray
        Fatigue life array [cycles].
    eps_total : ndarray
        Total strain range.
    eps_elastic : ndarray
        Elastic strain component.
    eps_plastic : ndarray
        Plastic strain component.
    """
    nf_array = np.logspace(1, 7, n_points)
    eps_elastic = config.FATIGUE_C1 * nf_array ** (-config.FATIGUE_BETA1)
    eps_plastic = config.FATIGUE_C2 * nf_array ** (-config.FATIGUE_BETA2)
    eps_total = eps_elastic + eps_plastic
    return nf_array, eps_total, eps_elastic, eps_plastic

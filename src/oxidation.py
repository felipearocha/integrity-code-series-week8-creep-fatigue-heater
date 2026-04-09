"""
High-temperature oxidation model for 9Cr-1Mo steel.

Governing equation (parabolic law, Wagner 1933):
  x^2 = k_p * t
  k_p = k_p0 * exp(-Q_ox / (R * T))

where:
  x   = oxide scale thickness [m]
  t   = exposure time [s]
  k_p = parabolic rate constant [m^2/s]

Effective metal loss:
  delta_wall = x * OXIDE_METAL_LOSS_FRACTION

Reference: Quadakkers et al., Oxidation of Metals 64 (2005) 175-209
"""

import numpy as np
from . import config


def parabolic_rate_constant(T: float) -> float:
    """
    Parabolic oxidation rate constant at temperature T.

    Parameters
    ----------
    T : float
        Temperature [K].

    Returns
    -------
    float
        k_p [m^2/s].
    """
    return config.OXIDE_KP0 * np.exp(-config.OXIDE_Q / (config.R_GAS * T))


def oxide_thickness(t: float, T: float) -> float:
    """
    Oxide scale thickness at time t and temperature T.

    Parameters
    ----------
    t : float
        Exposure time [seconds].
    T : float
        Temperature [K].

    Returns
    -------
    float
        Oxide thickness [m].
    """
    if t <= 0.0:
        return 0.0
    kp = parabolic_rate_constant(T)
    return np.sqrt(kp * t)


def metal_loss(t: float, T: float) -> float:
    """
    Effective metal loss due to oxidation at time t.

    Parameters
    ----------
    t : float
        Exposure time [seconds].
    T : float
        Temperature [K].

    Returns
    -------
    float
        Metal loss [m].
    """
    x = oxide_thickness(t, T)
    return x * config.OXIDE_METAL_LOSS_FRACTION


def effective_wall_thickness(t: float, T: float,
                              wt_initial: float = None) -> float:
    """
    Effective structural wall thickness after oxidation-driven thinning.

    Parameters
    ----------
    t : float
        Exposure time [seconds].
    T : float
        Temperature [K].
    wt_initial : float, optional
        Initial wall thickness [m]. Defaults to config.

    Returns
    -------
    float
        Effective wall thickness [m]. Clipped at 0.
    """
    if wt_initial is None:
        wt_initial = config.TUBE_WT_NOMINAL
    loss = metal_loss(t, T)
    return max(wt_initial - loss, 0.0)


def oxide_thickness_profile(times_s: np.ndarray, T: float) -> np.ndarray:
    """
    Oxide thickness over a time array at constant temperature.

    Parameters
    ----------
    times_s : ndarray
        Time array [seconds].
    T : float
        Temperature [K].

    Returns
    -------
    ndarray
        Oxide thickness array [m].
    """
    kp = parabolic_rate_constant(T)
    return np.sqrt(kp * np.maximum(times_s, 0.0))


def metal_loss_profile(times_s: np.ndarray, T: float) -> np.ndarray:
    """
    Effective metal loss profile over time.

    Parameters
    ----------
    times_s : ndarray
        Time array [seconds].
    T : float
        Temperature [K].

    Returns
    -------
    ndarray
        Metal loss array [m].
    """
    return oxide_thickness_profile(times_s, T) * config.OXIDE_METAL_LOSS_FRACTION


def time_to_critical_loss(wt_initial: float, T: float,
                           max_loss_fraction: float = None) -> float:
    """
    Time until wall thinning reaches critical fraction.

    Solves: OXIDE_METAL_LOSS_FRACTION * sqrt(k_p * t) = max_loss_fraction * wt_initial

    Parameters
    ----------
    wt_initial : float
        Initial wall thickness [m].
    T : float
        Temperature [K].
    max_loss_fraction : float, optional
        Maximum allowable wall loss fraction. Defaults to config.

    Returns
    -------
    float
        Time to critical loss [seconds].
    """
    if max_loss_fraction is None:
        max_loss_fraction = config.SIM_MAX_WALL_LOSS_FRACTION
    kp = parabolic_rate_constant(T)
    if kp <= 0.0:
        return np.inf
    critical_loss = max_loss_fraction * wt_initial
    x_crit = critical_loss / config.OXIDE_METAL_LOSS_FRACTION
    return x_crit**2 / kp

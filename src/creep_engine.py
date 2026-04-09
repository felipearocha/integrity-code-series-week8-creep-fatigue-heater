"""
Creep engine: Norton power-law, Larson-Miller (API 530), and MPC Omega (API 579-1).

Governing equations:
  Norton:  d(eps_cr)/dt = A * sigma^n * exp(-Q / (R*T))
  LMP:     T * (C + log10(t_r)) = f(sigma)   [API 530]
  Omega:   eps_dot = eps_dot_0 * exp(Omega * eps_cr)  [API 579-1 Part 10]
"""

import numpy as np
from . import config


def norton_creep_rate(sigma: float, T: float) -> float:
    """
    Steady-state (secondary) creep strain rate via Norton power law.

    Parameters
    ----------
    sigma : float
        Applied stress [Pa]. Must be > 0.
    T : float
        Temperature [K]. Must be > 0.

    Returns
    -------
    float
        Creep strain rate [1/s].
    """
    if sigma <= 0.0 or T <= 0.0:
        return 0.0
    rate = (config.NORTON_A
            * (sigma ** config.NORTON_N)
            * np.exp(-config.NORTON_Q / (config.R_GAS * T)))
    return rate


def larson_miller_parameter(T: float, t_r: float) -> float:
    """
    Compute Larson-Miller Parameter.

    Parameters
    ----------
    T : float
        Temperature [K].
    t_r : float
        Rupture time [hours]. Must be > 0.

    Returns
    -------
    float
        LMP value [K].
    """
    if t_r <= 0.0:
        raise ValueError("Rupture time must be positive.")
    return T * (config.LMP_C + np.log10(t_r))


def rupture_stress_from_lmp(lmp: float) -> float:
    """
    Minimum rupture stress from LMP using API 530 polynomial.

    Parameters
    ----------
    lmp : float
        Larson-Miller parameter value [K].

    Returns
    -------
    float
        Minimum rupture stress [Pa].
    """
    a = config.LMP_COEFF
    log_sigma_mpa = a[0] + a[1] * lmp + a[2] * lmp**2
    sigma_mpa = 10.0 ** log_sigma_mpa
    return sigma_mpa * 1.0e6  # convert MPa to Pa


def rupture_time_api530(sigma: float, T: float) -> float:
    """
    Estimate rupture time using API 530 Larson-Miller method.

    Solves: T * (C + log10(t_r)) = LMP(sigma)
    where LMP(sigma) is obtained by inverting the polynomial.

    Parameters
    ----------
    sigma : float
        Applied hoop stress [Pa].
    T : float
        Temperature [K].

    Returns
    -------
    float
        Estimated rupture time [hours].
    """
    from scipy.optimize import brentq

    sigma_mpa = sigma / 1.0e6
    log_sigma = np.log10(sigma_mpa) if sigma_mpa > 0 else 0.0

    # Solve: a0 + a1*LMP + a2*LMP^2 = log10(sigma_mpa) for LMP
    a = config.LMP_COEFF

    def residual(lmp):
        return a[0] + a[1] * lmp + a[2] * lmp**2 - log_sigma

    # LMP typically ranges 15000-30000 for 9Cr-1Mo
    lmp_sol = brentq(residual, 10000.0, 35000.0)

    # Solve: T * (C + log10(t_r)) = lmp_sol for t_r
    log_tr = lmp_sol / T - config.LMP_C
    t_r = 10.0 ** log_tr
    return t_r


def omega_creep_rate(eps_cr: float, T: float, sigma: float,
                     omega: float = None,
                     eps_dot_0: float = None) -> float:
    """
    MPC Omega method creep strain rate (API 579-1 Part 10).

    The Omega method captures tertiary creep acceleration:
      eps_dot = eps_dot_0 * exp(Omega * eps_cr)

    Parameters
    ----------
    eps_cr : float
        Accumulated creep strain (dimensionless).
    T : float
        Temperature [K] (used if omega/eps_dot_0 not provided).
    sigma : float
        Applied stress [Pa] (used if omega/eps_dot_0 not provided).
    omega : float, optional
        Omega parameter. Defaults to config value.
    eps_dot_0 : float, optional
        Initial strain rate [1/s]. Defaults to config value.

    Returns
    -------
    float
        Instantaneous creep strain rate [1/s].
    """
    if omega is None:
        omega = config.OMEGA_PARAM
    if eps_dot_0 is None:
        eps_dot_0 = config.OMEGA_EPS_DOT_0
    return eps_dot_0 * np.exp(omega * eps_cr)


def omega_rupture_strain(omega: float = None) -> float:
    """
    Critical creep strain at rupture per Omega method.

    eps_cr_rupture = 1 / Omega  (from integrating Omega equation to infinity)

    Parameters
    ----------
    omega : float, optional
        Omega parameter.

    Returns
    -------
    float
        Rupture strain (dimensionless).
    """
    if omega is None:
        omega = config.OMEGA_PARAM
    return 1.0 / omega


def omega_rupture_time(omega: float = None,
                       eps_dot_0: float = None) -> float:
    """
    Rupture time from Omega method.

    t_r = 1 / (Omega * eps_dot_0)

    Parameters
    ----------
    omega : float, optional
    eps_dot_0 : float, optional

    Returns
    -------
    float
        Rupture time [seconds].
    """
    if omega is None:
        omega = config.OMEGA_PARAM
    if eps_dot_0 is None:
        eps_dot_0 = config.OMEGA_EPS_DOT_0
    return 1.0 / (omega * eps_dot_0)


def integrate_creep_norton(sigma_func, T: float,
                           dt_s: float, total_time_s: float,
                           max_strain: float = 0.05):
    """
    Integrate Norton creep over time with time-varying stress.

    Parameters
    ----------
    sigma_func : callable
        Function sigma(t, eps_cr) -> stress [Pa].
    T : float
        Temperature [K] (constant for this integration).
    dt_s : float
        Time step [seconds].
    total_time_s : float
        Total simulation time [seconds].
    max_strain : float
        Failure strain criterion.

    Returns
    -------
    times : ndarray
        Time array [s].
    strains : ndarray
        Creep strain array.
    rates : ndarray
        Strain rate array [1/s].
    """
    n_steps = int(total_time_s / dt_s)
    times = np.zeros(n_steps + 1)
    strains = np.zeros(n_steps + 1)
    rates = np.zeros(n_steps + 1)

    for i in range(n_steps):
        t = times[i]
        eps = strains[i]
        sigma = sigma_func(t, eps)
        rate = norton_creep_rate(sigma, T)
        rates[i] = rate
        strains[i + 1] = eps + rate * dt_s
        times[i + 1] = t + dt_s
        if strains[i + 1] >= max_strain:
            # Truncate at failure
            times = times[:i + 2]
            strains = strains[:i + 2]
            rates = rates[:i + 2]
            rates[i + 1] = rate
            break
    else:
        rates[-1] = norton_creep_rate(sigma_func(times[-1], strains[-1]), T)

    return times, strains, rates


def integrate_creep_omega(sigma_func, T: float,
                          dt_s: float, total_time_s: float,
                          omega: float = None,
                          eps_dot_0: float = None,
                          max_strain: float = None):
    """
    Integrate Omega creep model over time with time-varying stress.

    Parameters
    ----------
    sigma_func : callable
        Function sigma(t, eps_cr) -> stress [Pa].
    T : float
        Temperature [K].
    dt_s : float
        Time step [seconds].
    total_time_s : float
        Total simulation time [seconds].
    omega : float, optional
    eps_dot_0 : float, optional
    max_strain : float, optional
        Defaults to 1/Omega (rupture strain).

    Returns
    -------
    times, strains, rates : ndarrays
    """
    if omega is None:
        omega = config.OMEGA_PARAM
    if eps_dot_0 is None:
        eps_dot_0 = config.OMEGA_EPS_DOT_0
    if max_strain is None:
        max_strain = omega_rupture_strain(omega)

    n_steps = int(total_time_s / dt_s)
    times = np.zeros(n_steps + 1)
    strains = np.zeros(n_steps + 1)
    rates = np.zeros(n_steps + 1)

    for i in range(n_steps):
        t = times[i]
        eps = strains[i]
        rate = omega_creep_rate(eps, T, sigma_func(t, eps), omega, eps_dot_0)
        rates[i] = rate
        strains[i + 1] = eps + rate * dt_s
        times[i + 1] = t + dt_s
        if strains[i + 1] >= max_strain:
            times = times[:i + 2]
            strains = strains[:i + 2]
            rates = rates[:i + 2]
            rates[i + 1] = rate
            break
    else:
        eps_final = strains[-1]
        rates[-1] = omega_creep_rate(eps_final, T,
                                      sigma_func(times[-1], eps_final),
                                      omega, eps_dot_0)

    return times, strains, rates


def hoop_stress_thin_wall(pressure: float, od: float,
                          wall_thickness: float) -> float:
    """
    Hoop stress for thin-walled cylinder (Barlow's formula per API 530).

    sigma = P * D_mean / (2 * t)

    Parameters
    ----------
    pressure : float
        Internal pressure [Pa].
    od : float
        Outside diameter [m].
    wall_thickness : float
        Effective wall thickness [m].

    Returns
    -------
    float
        Hoop stress [Pa].
    """
    if wall_thickness <= 0.0:
        return np.inf
    d_mean = od - wall_thickness
    return pressure * d_mean / (2.0 * wall_thickness)

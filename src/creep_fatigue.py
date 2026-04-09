"""
Creep-fatigue interaction assessment per ASME Section III Division 5.

Governing rule (time fraction / linear damage):
  D_total = D_f + D_c
  D_f = sum(n_j / N_f_j)       fatigue damage fraction
  D_c = sum(delta_t_k / t_r_k) creep damage fraction

Acceptance: (D_f, D_c) must lie within the interaction envelope.

Envelope for 9Cr-1Mo (Gr91) per ASME III-5 / Code Case N-812:
  Bilinear: (1, 0) -> (0.1, 0.01) -> (0, 1)
  This is significantly more restrictive than the (0.3, 0.3) envelope
  used for austenitic steels.

Reference: ORNL/TM-2018/868, Messner & Sham (2019)
"""

import numpy as np
from . import config


def creep_damage_fraction(hold_time_s: float, T: float,
                           sigma: float) -> float:
    """
    Creep damage fraction for a single hold period.

    D_c = delta_t / t_r

    Uses API 530 Larson-Miller to estimate rupture time.

    Parameters
    ----------
    hold_time_s : float
        Hold time at stress/temperature [seconds].
    T : float
        Temperature [K].
    sigma : float
        Applied stress [Pa].

    Returns
    -------
    float
        Creep damage fraction.
    """
    from .creep_engine import rupture_time_api530
    t_r_hours = rupture_time_api530(sigma, T)
    t_r_seconds = t_r_hours * 3600.0
    if t_r_seconds <= 0:
        return np.inf
    return hold_time_s / t_r_seconds


def interaction_envelope(n_points: int = 200):
    """
    Generate the creep-fatigue interaction envelope for 9Cr-1Mo.

    Bilinear envelope per ASME III-5 Code Case N-812:
      Segment 1: (1, 0) -> (0.1, 0.01)
      Segment 2: (0.1, 0.01) -> (0, 1)

    Returns
    -------
    df_envelope : ndarray
        Fatigue damage fraction values.
    dc_envelope : ndarray
        Creep damage fraction values.
    """
    df_int, dc_int = config.CF_INTERSECTION

    # Segment 1: (1, 0) -> (df_int, dc_int)
    n1 = n_points // 2
    df_seg1 = np.linspace(1.0, df_int, n1)
    dc_seg1 = dc_int * (1.0 - df_seg1) / (1.0 - df_int)

    # Segment 2: (df_int, dc_int) -> (0, 1)
    n2 = n_points - n1
    df_seg2 = np.linspace(df_int, 0.0, n2)
    dc_seg2 = dc_int + (1.0 - dc_int) * (df_int - df_seg2) / df_int

    df_envelope = np.concatenate([df_seg1, df_seg2[1:]])
    dc_envelope = np.concatenate([dc_seg1, dc_seg2[1:]])

    return df_envelope, dc_envelope


def is_within_envelope(D_f: float, D_c: float) -> bool:
    """
    Check if a (D_f, D_c) point is within the interaction envelope.

    Parameters
    ----------
    D_f : float
        Fatigue damage fraction.
    D_c : float
        Creep damage fraction.

    Returns
    -------
    bool
        True if within envelope (acceptable), False if outside (failure).
    """
    df_int, dc_int = config.CF_INTERSECTION

    if D_f < 0 or D_c < 0:
        return True  # No damage

    # Check against bilinear envelope
    if D_f >= df_int:
        # Segment 1 region: line from (1,0) to (df_int, dc_int)
        dc_limit = dc_int * (1.0 - D_f) / (1.0 - df_int)
    else:
        # Segment 2 region: line from (df_int, dc_int) to (0,1)
        dc_limit = dc_int + (1.0 - dc_int) * (df_int - D_f) / df_int

    return D_c <= dc_limit


def envelope_margin(D_f: float, D_c: float) -> float:
    """
    Distance from point to nearest envelope boundary (negative = outside).

    Parameters
    ----------
    D_f : float
        Fatigue damage fraction.
    D_c : float
        Creep damage fraction.

    Returns
    -------
    float
        Margin to envelope boundary. Positive = safe, negative = failed.
    """
    df_int, dc_int = config.CF_INTERSECTION

    if D_f >= df_int:
        dc_limit = dc_int * (1.0 - D_f) / (1.0 - df_int)
    else:
        dc_limit = dc_int + (1.0 - dc_int) * (df_int - D_f) / df_int

    return dc_limit - D_c


def accumulated_damage(n_cycles_list, delta_eps_list,
                        hold_times_s_list, T_list, sigma_list):
    """
    Compute total accumulated creep and fatigue damage.

    Parameters
    ----------
    n_cycles_list : list of float
        Number of cycles for each cycle type.
    delta_eps_list : list of float
        Strain range for each cycle type.
    hold_times_s_list : list of float
        Hold time per cycle [seconds] for each period.
    T_list : list of float
        Temperature [K] for each hold period.
    sigma_list : list of float
        Stress [Pa] for each hold period.

    Returns
    -------
    D_f : float
        Total fatigue damage fraction.
    D_c : float
        Total creep damage fraction.
    within : bool
        Whether point is within envelope.
    """
    from .fatigue import fatigue_damage_fraction

    D_f = sum(fatigue_damage_fraction(n, de)
              for n, de in zip(n_cycles_list, delta_eps_list))

    D_c = sum(creep_damage_fraction(dt, T, s)
              for dt, T, s in zip(hold_times_s_list, T_list, sigma_list))

    within = is_within_envelope(D_f, D_c)

    return D_f, D_c, within

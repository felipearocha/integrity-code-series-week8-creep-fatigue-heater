"""
Integrated fired heater tube life model.

Couples three degradation mechanisms:
  1. High-temperature creep (Norton + Omega)
  2. Thermal fatigue (Coffin-Manson)
  3. Oxidation wall thinning (parabolic)

The coupling is physical:
  - Oxide growth reduces effective wall thickness over time
  - Reduced wall increases hoop stress
  - Higher stress accelerates creep
  - Thermal cycling adds fatigue damage
  - Total life governed by creep-fatigue interaction envelope

Reference conditions: 9Cr-1Mo (T91) radiant tube, API 530 design
Anchor: Marathon Martinez CSB Report (2025-03-13)
"""

import numpy as np
from . import config
from . import creep_engine
from . import oxidation
from . import fatigue
from . import creep_fatigue


class TubeLifeResult:
    """Container for tube life simulation results."""

    def __init__(self):
        self.times_hours = None  # ndarray
        self.wall_thickness = None  # ndarray [m]
        self.hoop_stress = None  # ndarray [Pa]
        self.creep_strain = None  # ndarray
        self.creep_rate = None  # ndarray [1/s]
        self.oxide_thickness = None  # ndarray [m]
        self.D_fatigue = None  # ndarray (cumulative)
        self.D_creep = None  # ndarray (cumulative)
        self.within_envelope = None  # ndarray (bool)
        self.failure_mode = None  # str
        self.failure_time_hours = None  # float
        self.temperature_K = None  # float
        self.pressure_Pa = None  # float


def simulate_tube_life(
    T: float = None,
    pressure: float = None,
    wt_initial: float = None,
    od: float = None,
    delta_T: float = None,
    cycles_per_year: float = None,
    total_hours: float = None,
    dt_hours: float = None,
    use_omega: bool = False,
    omega: float = None,
    eps_dot_0: float = None,
) -> TubeLifeResult:
    """
    Run coupled creep-fatigue-oxidation simulation.

    Parameters
    ----------
    T : float
        Service temperature [K].
    pressure : float
        Internal pressure [Pa].
    wt_initial : float
        Initial wall thickness [m].
    od : float
        Tube OD [m].
    delta_T : float
        Thermal cycle amplitude [K].
    cycles_per_year : float
        Number of thermal cycles per year.
    total_hours : float
        Simulation duration [hours].
    dt_hours : float
        Time step [hours].
    use_omega : bool
        If True, use Omega method instead of Norton.
    omega : float, optional
        Omega parameter (if use_omega=True).
    eps_dot_0 : float, optional
        Initial strain rate for Omega (if use_omega=True).

    Returns
    -------
    TubeLifeResult
        Complete simulation results.
    """
    # Defaults from config
    if T is None:
        T = config.DESIGN_TEMPERATURE
    if pressure is None:
        pressure = config.INTERNAL_PRESSURE
    if wt_initial is None:
        wt_initial = config.TUBE_WT_NOMINAL
    if od is None:
        od = config.TUBE_OD
    if delta_T is None:
        delta_T = config.DELTA_T_CYCLE
    if cycles_per_year is None:
        cycles_per_year = config.CYCLES_PER_YEAR
    if total_hours is None:
        total_hours = config.SIM_TOTAL_HOURS
    if dt_hours is None:
        dt_hours = config.SIM_DT_HOURS

    dt_s = dt_hours * 3600.0
    n_steps = int(total_hours / dt_hours)

    # Pre-allocate arrays
    times_h = np.zeros(n_steps + 1)
    wall_t = np.zeros(n_steps + 1)
    hoop_s = np.zeros(n_steps + 1)
    creep_eps = np.zeros(n_steps + 1)
    creep_r = np.zeros(n_steps + 1)
    oxide_t = np.zeros(n_steps + 1)
    D_f_cum = np.zeros(n_steps + 1)
    D_c_cum = np.zeros(n_steps + 1)
    within = np.ones(n_steps + 1, dtype=bool)

    # Initial conditions
    wall_t[0] = wt_initial
    hoop_s[0] = creep_engine.hoop_stress_thin_wall(pressure, od, wt_initial)

    # Thermal strain range for fatigue
    eps_thermal = fatigue.thermal_strain_range(delta_T)
    Nf = fatigue.coffin_manson_life(eps_thermal)

    # Fatigue damage per year
    D_f_per_year = cycles_per_year / Nf if Nf > 0 and Nf != np.inf else 0.0

    failure_mode = None
    failure_idx = n_steps

    for i in range(n_steps):
        t_h = i * dt_hours
        t_s = t_h * 3600.0
        times_h[i] = t_h

        # 1. Oxide growth and wall thinning
        ox = oxidation.oxide_thickness(t_s, T)
        oxide_t[i] = ox
        wt_eff = max(wt_initial - ox * config.OXIDE_METAL_LOSS_FRACTION, 1e-6)
        wall_t[i] = wt_eff

        # 2. Hoop stress (increases as wall thins)
        sigma = creep_engine.hoop_stress_thin_wall(pressure, od, wt_eff)
        hoop_s[i] = sigma

        # 3. Creep strain increment
        if use_omega:
            rate = creep_engine.omega_creep_rate(
                creep_eps[i], T, sigma, omega, eps_dot_0
            )
        else:
            rate = creep_engine.norton_creep_rate(sigma, T)
        creep_r[i] = rate
        creep_eps[i + 1] = creep_eps[i] + rate * dt_s

        # 4. Fatigue damage accumulation
        years_elapsed = t_h / 8760.0
        D_f_cum[i] = D_f_per_year * years_elapsed

        # 5. Creep damage accumulation (time fraction)
        # Rupture time at current conditions
        try:
            t_r_h = creep_engine.rupture_time_api530(sigma, T)
            D_c_increment = dt_hours / t_r_h if t_r_h > 0 else np.inf
        except Exception:
            D_c_increment = 0.0
        D_c_cum[i + 1] = D_c_cum[i] + D_c_increment

        # 6. Check interaction envelope
        within[i] = creep_fatigue.is_within_envelope(D_f_cum[i], D_c_cum[i])

        # 7. Check failure criteria
        if creep_eps[i + 1] >= config.SIM_MAX_CREEP_STRAIN:
            failure_mode = "creep_rupture"
            failure_idx = i + 1
            break
        if wt_eff <= wt_initial * (1.0 - config.SIM_MAX_WALL_LOSS_FRACTION):
            failure_mode = "oxidation_wall_loss"
            failure_idx = i + 1
            break
        if not within[i]:
            failure_mode = "creep_fatigue_interaction"
            failure_idx = i + 1
            break

    # Fill final step
    if failure_mode is None:
        failure_idx = n_steps
        failure_mode = "survived"
        t_s_final = total_hours * 3600.0
        oxide_t[n_steps] = oxidation.oxide_thickness(t_s_final, T)
        wall_t[n_steps] = max(
            wt_initial - oxide_t[n_steps] * config.OXIDE_METAL_LOSS_FRACTION, 1e-6
        )
        hoop_s[n_steps] = creep_engine.hoop_stress_thin_wall(
            pressure, od, wall_t[n_steps]
        )
        if use_omega:
            creep_r[n_steps] = creep_engine.omega_creep_rate(
                creep_eps[n_steps], T, hoop_s[n_steps], omega, eps_dot_0
            )
        else:
            creep_r[n_steps] = creep_engine.norton_creep_rate(
                hoop_s[n_steps], T
            )
        times_h[n_steps] = total_hours
        D_f_cum[n_steps] = D_f_per_year * (total_hours / 8760.0)
        within[n_steps] = creep_fatigue.is_within_envelope(
            D_f_cum[n_steps], D_c_cum[n_steps]
        )

    # Trim arrays to actual simulation length
    idx = failure_idx + 1
    result = TubeLifeResult()
    result.times_hours = times_h[:idx]
    result.wall_thickness = wall_t[:idx]
    result.hoop_stress = hoop_s[:idx]
    result.creep_strain = creep_eps[:idx]
    result.creep_rate = creep_r[:idx]
    result.oxide_thickness = oxide_t[:idx]
    result.D_fatigue = D_f_cum[:idx]
    result.D_creep = D_c_cum[:idx]
    result.within_envelope = within[:idx]
    result.failure_mode = failure_mode
    result.failure_time_hours = times_h[min(failure_idx, len(times_h) - 1)]
    result.temperature_K = T
    result.pressure_Pa = pressure

    return result


def run_baseline():
    """
    Run baseline simulation with default parameters.

    Returns
    -------
    TubeLifeResult
    """
    return simulate_tube_life()


def run_baseline_omega():
    """
    Run baseline simulation with Omega method.

    Returns
    -------
    TubeLifeResult
    """
    return simulate_tube_life(use_omega=True)

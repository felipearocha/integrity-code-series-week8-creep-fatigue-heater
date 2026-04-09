"""
Configuration: Material properties, operating conditions, and simulation parameters.

Material: 9Cr-1Mo-V-Nb (ASME Grade T91 / P91)
Reference: API 530 7th Ed., API 579-1/ASME FFS-1 Part 10,
           NIMS Creep Data Sheet No. 43A (2014)

All parameters are anchored to published data with [ASSUMED] flags
where engineering judgment was applied.
"""

import numpy as np

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
R_GAS = 8.314  # J/(mol*K), universal gas constant

# ---------------------------------------------------------------------------
# Tube geometry (typical refinery radiant tube)
# ---------------------------------------------------------------------------
TUBE_OD = 0.1143  # m, 4.5 inch OD [API 530 typical]
TUBE_WT_NOMINAL = 0.00635  # m, 0.250 inch nominal wall [API 530 typical]
TUBE_ID = TUBE_OD - 2 * TUBE_WT_NOMINAL  # m
TUBE_LENGTH = 12.0  # m [ASSUMED] typical radiant tube length

# ---------------------------------------------------------------------------
# Operating conditions (design basis)
# ---------------------------------------------------------------------------
INTERNAL_PRESSURE = 5.0e6  # Pa, 725 psi [ASSUMED] catalytic reformer
DESIGN_TEMPERATURE = 853.15  # K (580 C) [ASSUMED] high-severity reformer
AMBIENT_TEMPERATURE = 298.15  # K

# Thermal cycling parameters
STARTUP_RATE = 50.0  # K/hr, controlled startup ramp [API 560 limit]
SHUTDOWN_RATE = 30.0  # K/hr [ASSUMED]
DELTA_T_CYCLE = 300.0  # K, thermal cycle amplitude (580C - 280C) [ASSUMED]
CYCLES_PER_YEAR = 8.0  # planned + unplanned shutdowns [ASSUMED]

# ---------------------------------------------------------------------------
# 9Cr-1Mo (T91) material properties
# ---------------------------------------------------------------------------
# Elastic properties at 565 C
YOUNGS_MODULUS = 160.0e9  # Pa at 580 C, API 530 Table 3
POISSONS_RATIO = 0.30  # [ASSUMED] typical for ferritic steel
CTE = 12.5e-6  # 1/K, mean CTE 20-565C, API 530 Table 3

# ---------------------------------------------------------------------------
# Norton creep law: d(eps_cr)/dt = A * sigma^n * exp(-Q_cr / (R*T))
# Calibrated to NIMS Data Sheet 43A for T91 at 550-600C
# ---------------------------------------------------------------------------
NORTON_A = 7.86e-57  # 1/(s * Pa^n), calibrated to NIMS 43A: 1e-8 /s at 100 MPa, 600C
NORTON_N = 10.5  # stress exponent at 550-600C, NIMS 43A
NORTON_Q = 600.0e3  # J/mol, activation energy, Maruyama et al. 2001

# ---------------------------------------------------------------------------
# Larson-Miller parameter (API 530 method)
# LMP = T * (C + log10(t_r))  where T in K, t_r in hours
# For 9Cr-1Mo: C ~ 20, from API 530 Fig. F-14
# Minimum rupture stress correlation (API 530 polynomial):
#   log10(sigma_r) = a0 + a1*LMP + a2*LMP^2  [sigma_r in MPa]
# ---------------------------------------------------------------------------
LMP_C = 20.0  # Larson-Miller constant, API 530
# Polynomial coefficients for minimum rupture stress (9Cr-1Mo)
# Fitted to API 530 Fig F-14 / NIMS 43A data [ASSUMED fit]
# Calibrated to NIMS 43A + API 530 Fig F-14 for T91:
#   LMP=19000 -> sigma_r=300 MPa, LMP=21325 -> 90 MPa, LMP=23000 -> 30 MPa
LMP_COEFF = np.array([0.672, 3.80e-4, -1.50e-8])  # a0, a1, a2

# ---------------------------------------------------------------------------
# MPC Omega method (API 579-1 Part 10)
# eps_dot = eps_dot_0 * exp(Omega * eps_cr)
# Omega and eps_dot_0 from API 579-1 Table F.30 for 9Cr-1Mo
# ---------------------------------------------------------------------------
OMEGA_PARAM = 7.5  # dimensionless, median value at 565C [API 579-1 Table F.30]
OMEGA_EPS_DOT_0 = 3.2e-10  # 1/s, initial strain rate at 565C/50MPa [ASSUMED]
# Scatter band factors (API 579-1)
OMEGA_DELTA_CD = 0.0  # creep ductility adjustment [default = 0]
OMEGA_DELTA_SR = 0.0  # scatter band adjustment [default = 0]

# ---------------------------------------------------------------------------
# Oxidation kinetics (parabolic law)
# x^2 = k_p * t   where x = oxide thickness (m), t = time (s)
# k_p = k_p0 * exp(-Q_ox / (R*T))
# For 9Cr steel in air/flue gas: Quadakkers et al. 2005
# ---------------------------------------------------------------------------
OXIDE_KP0 = 4.08e-4  # m^2/s, pre-exponential [Quadakkers 2005]
OXIDE_Q = 250.0e3  # J/mol, activation energy [Quadakkers 2005]
OXIDE_PILLING_BEDWORTH = 2.07  # PBR for Fe2O3/Cr2O3 on Fe-9Cr [ASSUMED]
# Effective wall loss factor: not all oxide is structural loss
# Typically ~50-70% of oxide thickness is effective metal loss
OXIDE_METAL_LOSS_FRACTION = 0.60  # [ASSUMED]

# ---------------------------------------------------------------------------
# Coffin-Manson fatigue parameters at 565 C for T91
# Delta_eps_total = C1 * Nf^(-beta1) + C2 * Nf^(-beta2)
# Elastic term (Basquin) + Plastic term (Coffin-Manson)
# From Fournier et al. 2008, Int. J. Fatigue
# ---------------------------------------------------------------------------
FATIGUE_C1 = 0.0045  # elastic coefficient [Fournier 2008]
FATIGUE_BETA1 = 0.085  # Basquin exponent [Fournier 2008]
FATIGUE_C2 = 0.48  # plastic coefficient [Fournier 2008]
FATIGUE_BETA2 = 0.58  # Coffin-Manson exponent [Fournier 2008]

# ---------------------------------------------------------------------------
# Creep-fatigue interaction envelope (ASME Section III Div 5)
# Bilinear envelope for 9Cr-1Mo:
#   Intersection point: (D_f, D_c) = (0.1, 0.01) per Code Case N-812
#   Corner points: (1,0), (0.1, 0.01), (0,1)
# Time fraction rule: sum(n/Nf) + sum(dt/tr) <= D_envelope
# ---------------------------------------------------------------------------
CF_INTERSECTION = (0.1, 0.01)  # (fatigue fraction, creep fraction)
# This is the most restrictive intersection in ASME III-5 for Gr91

# ---------------------------------------------------------------------------
# Monte Carlo / LHS parameters
# ---------------------------------------------------------------------------
MC_N_SAMPLES = 2000  # Latin Hypercube samples
MC_RANDOM_SEED = 42

# Parameter ranges for LHS sweep (min, max)
MC_RANGES = {
    "temperature_K": (833.15, 903.15),  # 560-630 C
    "pressure_Pa": (3.0e6, 7.0e6),  # 435-1015 psi
    "wall_thickness_m": (0.004, 0.007),  # degraded to nominal range
    "cycles_per_year": (4.0, 15.0),  # conservative to aggressive
    "delta_T_K": (200.0, 380.0),  # thermal cycle range
}

# ---------------------------------------------------------------------------
# Simulation control
# ---------------------------------------------------------------------------
SIM_TOTAL_HOURS = 100_000  # design life target (API 530)
SIM_DT_HOURS = 100  # time step for creep integration
SIM_MAX_CREEP_STRAIN = 0.05  # 5% failure criterion [ASSUMED]
SIM_MAX_WALL_LOSS_FRACTION = 0.40  # retire at 40% wall loss [ASSUMED]

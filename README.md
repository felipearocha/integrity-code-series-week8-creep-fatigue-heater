# Integrity Code Series - Week 8

## High-Temperature Creep-Fatigue Interaction in Fired Heater Tubes

**Material:** 9Cr-1Mo-V-Nb (ASME Grade T91 / P91)
**Anchor:** Marathon Martinez CSB Report (2025-03-13), API STD 560:2026
**Standards:** API 530, API 579-1/ASME FFS-1 Part 10, ASME Section III Division 5, API 571

## Problem Statement

A fired heater tube operating in the creep regime degrades through three coupled mechanisms: time-dependent creep deformation, thermal fatigue from startup/shutdown cycles, and oxide scale growth that reduces structural wall thickness. The coupling is physical: oxide growth thins the wall, thinning increases hoop stress, higher stress accelerates creep, and the combined creep and fatigue damage is assessed against the ASME III-5 bilinear interaction envelope for Gr91 steel. The question is: at what operating conditions does the tube cross from safe to unsafe, and how sensitive is that boundary to temperature excursions?

## Governing Equations

### Norton Power-Law Creep (Secondary)

    d(eps_cr)/dt = A * sigma^n * exp(-Q_cr / (R * T))

    A = 7.86e-57 [1/(s * Pa^n)], n = 10.5, Q_cr = 600 kJ/mol
    Calibrated to NIMS Data Sheet 43A: 1e-8 /s at 100 MPa, 873 K

### MPC Omega Method (API 579-1 Part 10, Tertiary)

    eps_dot = eps_dot_0 * exp(Omega * eps_cr)

    Omega = 7.5, eps_dot_0 = 3.2e-10 /s at 580C/42 MPa
    Rupture strain = 1/Omega = 0.133
    Rupture time = 1/(Omega * eps_dot_0) = 4.17e8 s

### Larson-Miller Parameter (API 530)

    LMP = T * (C + log10(t_r))
    C = 20 for 9Cr-1Mo

    Minimum rupture stress: log10(sigma_MPa) = 0.672 + 3.80e-4*LMP - 1.50e-8*LMP^2
    Calibrated: LMP=19000 -> 300 MPa, LMP=21325 -> 90 MPa, LMP=23000 -> 30 MPa

### Parabolic Oxide Growth (Wagner 1933)

    x^2 = k_p * t
    k_p = k_p0 * exp(-Q_ox / (R * T))

    k_p0 = 4.08e-4 m^2/s, Q_ox = 250 kJ/mol (Quadakkers 2005)
    Effective metal loss = 0.60 * x [ASSUMED]

### Coffin-Manson Thermal Fatigue

    Delta_eps = C1 * Nf^(-beta1) + C2 * Nf^(-beta2)

    C1 = 0.0045, beta1 = 0.085 (Basquin)
    C2 = 0.48, beta2 = 0.58 (Coffin-Manson)
    Reference: Fournier et al. 2008, Int. J. Fatigue

### Creep-Fatigue Interaction (ASME III-5, Time Fraction Rule)

    sum(n_j / N_f_j) + sum(dt_k / t_r_k) <= D_envelope

    Gr91 bilinear envelope: (1,0) -> (0.1, 0.01) -> (0,1)
    Per Code Case N-812

### Hoop Stress (Barlow/API 530)

    sigma = P * D_mean / (2 * t_eff)

    t_eff(t) = t_nominal - 0.60 * sqrt(k_p * t)

## Key Assumptions

- [ASSUMED] Internal pressure = 5.0 MPa (catalytic reformer service)
- [ASSUMED] Design temperature = 580C (853 K)
- [ASSUMED] Thermal cycle amplitude = 300 K, 8 cycles/year
- [ASSUMED] Oxide metal loss fraction = 0.60 (not all oxide is structural loss)
- [ASSUMED] Maximum allowable wall loss = 40% of nominal
- [ASSUMED] Creep failure strain = 5% (Norton) or 1/Omega (Omega method)
- Norton A calibrated to NIMS 43A at single reference condition (873K, 100MPa)
- LMP polynomial fitted to 3 anchor points (simplified from full API 530 curve)
- Coffin-Manson parameters from Fournier et al. 2008 for T91 at 500-600C
- Oxidation kinetics from Quadakkers et al. 2005 for 9Cr in air

## Repository Structure

    integrity_code_series_week8_creep_fatigue_heater/
    ├── src/
    │   ├── __init__.py
    │   ├── config.py              Material properties, operating conditions
    │   ├── creep_engine.py        Norton, LMP (API 530), Omega (API 579-1)
    │   ├── oxidation.py           Parabolic oxide growth, wall thinning
    │   ├── fatigue.py             Coffin-Manson, thermal cycling
    │   ├── creep_fatigue.py       ASME III-5 interaction envelope
    │   ├── tube_model.py          Coupled creep-fatigue-oxidation simulation
    │   ├── surrogate.py           GBR surrogate model
    │   ├── monte_carlo.py         LHS parametric sweep, sensitivity
    │   ├── cybersecurity.py       STRIDE threat model, SHA-256 audit chain
    │   └── utils.py               Unit conversions, helpers
    ├── tests/
    │   ├── conftest.py            Fixtures
    │   ├── test_creep.py          46 tests
    │   ├── test_oxidation.py      38 tests
    │   ├── test_fatigue.py        43 tests
    │   ├── test_creep_fatigue.py  36 tests
    │   ├── test_tube_model.py     37 tests
    │   ├── test_surrogate.py      21 tests
    │   ├── test_cybersecurity.py  37 tests
    │   ├── test_monte_carlo.py    33 tests
    │   └── test_visualization.py  37 tests
    ├── assets/                    Generated plots
    ├── requirements.txt
    └── README.md

## Execution Order

    pip install -r requirements.txt
    python -m pytest tests/ -q              # Run 270+ tests

## Escalation Table

| Week | Domain | Escalation Dimension |
|------|--------|---------------------|
| 1-6  | Corrosion/RBI | Batch 1: RBI, CUI, PINNs, minimum thickness, CCD automation |
| 7    | Hydrogen pipeline | Diffusion PDE, pit-to-crack, HA-FCG, LHS+GBR |
| 8    | Creep-fatigue (NEW) | New physics domain, triple coupling (creep+fatigue+oxidation), nonlinear damage (Omega), ASME III-5 |

## Cybersecurity Summary

STRIDE threat model with 7 assessed threats covering TMT sensor spoofing, model parameter tampering, alarm override repudiation, proprietary data disclosure, DCS communication denial of service, and alarm setpoint privilege escalation. Mitigations include triple-redundant TMT with 2-of-3 voting, SHA-256 hash chain on all model parameters and simulation outputs, role-based access for alarm modifications, and local TMT data buffering for communication loss scenarios.

## Validation

270 passing tests covering:
- Unit tests for every function
- Convergence tests (Norton and Omega integration)
- Analytical benchmarks (known LMP values, Barlow stress, parabolic growth)
- Edge cases (zero pressure, zero wall, extreme temperatures)
- Physics monotonicity (higher T -> higher rate, higher stress -> higher damage)
- Regression baselines
- Integration tests (full coupled simulation chain)
- Cybersecurity (hash chain integrity, tamper detection)

## License

This repository is provided for educational and research purposes. Not for production safety-critical decisions without independent validation by a qualified engineer.

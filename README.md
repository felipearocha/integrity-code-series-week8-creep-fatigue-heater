# Integrity Code Series — Week 8

## High-Temperature Creep-Fatigue Interaction in Fired Heater Tubes

[![CI](https://github.com/felipearocha/integrity-code-series-week8-creep-fatigue-heater/actions/workflows/ci.yml/badge.svg)](https://github.com/felipearocha/integrity-code-series-week8-creep-fatigue-heater/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests: 324](https://img.shields.io/badge/tests-324%20passing-brightgreen.svg)]()
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

---

**Material:** 9Cr-1Mo-V-Nb (ASME Grade T91 / P91)
**Anchor:** Marathon Martinez CSB Report (2025-03-13), API STD 560:2026
**Standards:** API 530, API 579-1/ASME FFS-1 Part 10, ASME Section III Division 5, API 571

## Problem Statement

A fired heater tube operating in the creep regime degrades through three coupled mechanisms: time-dependent creep deformation, thermal fatigue from startup/shutdown cycles, and oxide scale growth that reduces structural wall thickness. The coupling is physical — oxide growth thins the wall, thinning increases hoop stress, higher stress accelerates creep, and the combined creep and fatigue damage is assessed against the ASME III-5 bilinear interaction envelope for Gr91 steel.

**The question:** at what operating conditions does the tube cross from safe to unsafe, and how sensitive is that boundary to temperature excursions?

## Key Results

| Condition | D_creep | D_fatigue | Margin | Status |
|-----------|---------|-----------|--------|--------|
| Design (580 C, 5 MPa) | 0.043 | 0.003 | 2.3x to envelope | Safe |
| Hot spot (+40 C) | 0.647 | 0.003 | Exceeded | **Fail** |

> A 40 C hot spot increases creep damage by **15x** — exponential temperature sensitivity dominates all other parameters.

## Governing Equations

### Norton Power-Law Creep (Secondary)

```
d(eps_cr)/dt = A * sigma^n * exp(-Q_cr / (R * T))

A = 7.86e-57 [1/(s * Pa^n)], n = 10.5, Q_cr = 600 kJ/mol
Calibrated to NIMS Data Sheet 43A: 1e-8 /s at 100 MPa, 873 K
```

### MPC Omega Method (API 579-1 Part 10, Tertiary)

```
eps_dot = eps_dot_0 * exp(Omega * eps_cr)

Omega = 7.5, eps_dot_0 = 3.2e-10 /s at 580C/42 MPa
Rupture strain = 1/Omega = 0.133
```

### Larson-Miller Parameter (API 530)

```
LMP = T * (C + log10(t_r)),  C = 20 for 9Cr-1Mo

Minimum rupture stress: log10(sigma_MPa) = 0.672 + 3.80e-4*LMP - 1.50e-8*LMP^2
Anchor points: LMP=19000 -> 300 MPa, LMP=21325 -> 90 MPa, LMP=23000 -> 30 MPa
```

### Parabolic Oxide Growth (Wagner 1933)

```
x^2 = k_p * t
k_p = k_p0 * exp(-Q_ox / (R * T))

k_p0 = 4.08e-4 m^2/s, Q_ox = 250 kJ/mol (Quadakkers 2005)
Effective metal loss = 0.60 * x [ASSUMED]
```

### Coffin-Manson Thermal Fatigue

```
Delta_eps = C1 * Nf^(-beta1) + C2 * Nf^(-beta2)

C1 = 0.0045, beta1 = 0.085 (Basquin)
C2 = 0.48, beta2 = 0.58 (Coffin-Manson)
Reference: Fournier et al. 2008, Int. J. Fatigue 30, 1797-1812
```

### Creep-Fatigue Interaction (ASME III-5)

```
D_f = sum(n_j / N_f_j)        fatigue damage fraction
D_c = sum(dt_k / t_r_k)       creep damage fraction (time-fraction rule)

Gr91 bilinear envelope: (1,0) -> (0.1, 0.01) -> (0,1)
Per Code Case N-812 / ORNL/TM-2018/868
```

### Hoop Stress (Barlow / API 530)

```
sigma = P * D_mean / (2 * t_eff)
t_eff(t) = t_nominal - 0.60 * sqrt(k_p * t)
```

## Architecture

```
src/
├── config.py              Material properties, operating conditions
├── creep_engine.py        Norton, LMP (API 530), Omega (API 579-1)
├── oxidation.py           Parabolic oxide growth, wall thinning
├── fatigue.py             Coffin-Manson, thermal cycling
├── creep_fatigue.py       ASME III-5 interaction envelope
├── tube_model.py          Coupled creep-fatigue-oxidation simulation
├── surrogate.py           GBR surrogate model for fast prediction
├── monte_carlo.py         LHS parametric sweep, Spearman sensitivity
├── cybersecurity.py       STRIDE threat model, SHA-256 audit chain
└── utils.py               Unit conversions
tests/
├── conftest.py            Fixtures and path setup
├── test_creep.py          46 tests — Norton, LMP, Omega, hoop stress
├── test_oxidation.py      38 tests — parabolic kinetics, wall degradation
├── test_fatigue.py        43 tests — Coffin-Manson, thermal strain
├── test_creep_fatigue.py  36 tests — ASME III-5 envelope, damage rules
├── test_tube_model.py     37 tests — coupled simulation, failure modes
├── test_surrogate.py      21 tests — GBR training, cross-validation
├── test_monte_carlo.py    33 tests — LHS sampling, sensitivity
├── test_cybersecurity.py  37 tests — audit chain, STRIDE coverage
└── test_visualization.py  33 tests — project structure validation
```

## Quick Start

```bash
git clone https://github.com/felipearocha/integrity-code-series-week8-creep-fatigue-heater.git
cd integrity-code-series-week8-creep-fatigue-heater
pip install -r requirements.txt
python -m pytest tests/ -v
```

## Sensitivity Analysis (Monte Carlo)

2,000 Latin Hypercube samples across 5 parameters:

| Parameter | Spearman rho | Effect |
|-----------|-------------|--------|
| Temperature | -0.59 | **Dominant** — exponential via Arrhenius |
| Pressure | -0.50 | Strong — linear via Barlow hoop stress |
| Wall thickness | +0.40 | Protective — inverse stress relationship |
| Cycles/year | -0.15 | Weak — fatigue secondary to creep |
| Delta T | -0.08 | Weak — thermal strain below plastic threshold |

## Key Assumptions

| Parameter | Value | Source | Flag |
|-----------|-------|--------|------|
| Internal pressure | 5.0 MPa | Catalytic reformer | [ASSUMED] |
| Design temperature | 580 C (853 K) | High-severity reformer | [ASSUMED] |
| Thermal cycle | 300 K, 8 cycles/yr | Startup/shutdown | [ASSUMED] |
| Oxide metal loss | 60% of scale | Spallation estimate | [ASSUMED] |
| Max wall loss | 40% of nominal | Retirement criterion | [ASSUMED] |
| Norton A | 7.86e-57 | NIMS 43A single point | Calibrated |
| LMP polynomial | 3 anchor points | Simplified API 530 | Fitted |
| Coffin-Manson | Fournier 2008 | T91 at 500-600 C | Published |
| Oxidation k_p | Quadakkers 2005 | 9Cr in air | Published |

## Cybersecurity

STRIDE threat model with 7 assessed threats:
- **Spoofing:** Triple-redundant TMT with 2-of-3 voting
- **Tampering:** SHA-256 hash chain on all model parameters and outputs
- **Repudiation:** Immutable audit log for alarm overrides
- **Information Disclosure:** Role-based access to proprietary correlations
- **Denial of Service:** Local TMT data buffering for DCS communication loss
- **Elevation of Privilege:** Restricted alarm setpoint modification

## Escalation Table

| Week | Domain | Escalation |
|------|--------|------------|
| 1-6 | Corrosion / RBI | RBI, CUI, PINNs, minimum thickness, CCD automation |
| 7 | Hydrogen pipeline | Diffusion PDE, pit-to-crack, HA-FCG, LHS+GBR |
| **8** | **Creep-fatigue** | **New physics domain, triple coupling, Omega nonlinearity, ASME III-5** |

## References

1. API 530, 7th Edition — Calculation of Heater-Tube Thickness in Petroleum Refineries
2. API 579-1/ASME FFS-1 — Fitness-For-Service, Part 10 (Creep Assessment)
3. ASME Section III Division 5 — High Temperature Reactors (Code Case N-812)
4. NIMS Creep Data Sheet No. 43A (2014) — 9Cr-1Mo-V-Nb steel
5. Fournier et al. 2008, Int. J. Fatigue 30, 1797-1812
6. Quadakkers et al. 2005, Oxidation of Metals 64, 175-209
7. Maruyama et al. 2001 — Creep activation energy for ferritic steels
8. ORNL/TM-2018/868 — Messner & Sham, Creep-fatigue procedures for Gr91
9. Marathon Martinez CSB Report (2025-03-13)

## License

[MIT](LICENSE) — See [DISCLAIMER](LICENSE) for safety-critical use limitations.

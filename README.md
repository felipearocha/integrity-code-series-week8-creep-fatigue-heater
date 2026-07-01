# Integrity Code Series — Week 8

## High-Temperature Creep-Fatigue Interaction in Fired Heater Tubes

[![CI](https://github.com/felipearocha/integrity-code-series-week8-creep-fatigue-heater/actions/workflows/ci.yml/badge.svg)](https://github.com/felipearocha/integrity-code-series-week8-creep-fatigue-heater/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![Tests: 324 passing](https://img.shields.io/badge/tests-324%20passing-brightgreen.svg)](tests)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20172467.svg)](https://doi.org/10.5281/zenodo.20172467)

---

**Material:** 9Cr-1Mo-V-Nb (ASME Grade T91 / P91)
**Anchor:** Marathon Martinez CSB Report (2025-03-13), API STD 560:2026
**Standards:** API 530, API 579-1/ASME FFS-1 Part 10, ASME Section III Division 5, API 571

## Integrity Code Series

Part of an ongoing series of physics-first integrity simulators by Felipe Rocha:

| # | Repo | Domain |
|---|---|---|
| Week 3 | [Integrity-code-series-3](https://github.com/felipearocha/Integrity-code-series-3) | F1 lap simulation (six coupled ODEs) |
| Week 6 | [Integrity-code-series-week6-smartphone-galvanic](https://github.com/felipearocha/Integrity-code-series-week6-smartphone-galvanic) | Smartphone galvanic corrosion (Laplace + Butler-Volmer) |
| Week 7 | [integrity_code_series_week7_h2_lferw](https://github.com/felipearocha/integrity_code_series_week7_h2_lferw) | LF-ERW H2 conversion (B31.12 + NACE TM0316) |
| **Week 8** | **[integrity-code-series-week8-creep-fatigue-heater](https://github.com/felipearocha/integrity-code-series-week8-creep-fatigue-heater)** | **Creep-fatigue 9Cr-1Mo (Norton/Omega + Coffin-Manson) — this repo** |
| Week 9 | [integrity-code-series-week9-cui](https://github.com/felipearocha/integrity-code-series-week9-cui) | CUI thermohygro-electrochemical (3 PDEs, Strang) |
| Week 10 | [integrity-code-series-week-10_nnph_scc](https://github.com/felipearocha/integrity-code-series-week-10_nnph_scc) | NNpHSCC full-physics (Chen-Sutherby-Xing + BS 7910) |
| Week 11 | [integrity-code-series-week11-erosion-corrosion-multiphase](https://github.com/felipearocha/integrity-code-series-week11-erosion-corrosion-multiphase) | Erosion-corrosion multiphase (NORSOK M-506 + DNV-RP-O501 + G119 + API 579 Part 5) |
| Bonus | [Vibration-Accelerated-Corrosion-Coupled-Mechano-Electrochemical-Simulation](https://github.com/felipearocha/Vibration-Accelerated-Corrosion-Coupled-Mechano-Electrochemical-Simulation) | Vibration-accelerated corrosion (SDOF + Butler-Volmer + Archard) |
| Bonus | [synthetic-integrity-digital-twin-piml](https://github.com/felipearocha/synthetic-integrity-digital-twin-piml) | Physics-informed neural-network surrogate |
| Bonus | [integrity-data-foundation](https://github.com/felipearocha/integrity-data-foundation) | Engineering data validation baseline |

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

Every constant is tagged to its source standard or paper. Full rendered (MathJax)
reference: **[docs/equations.html](docs/equations.html)** — open in any browser.

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
docs/
└── equations.html         Rendered (MathJax) governing-equations reference
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

## Cybersecurity (STRIDE)

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

## Anti-Hallucination Note

Every equation and constant in this package is tagged to its origin, and the
tiers are applied honestly:

- **T1 (published standard / paper):** the Norton, MPC Omega and Larson-Miller
  forms and the ASME III-5 bilinear envelope come from API 530, API 579-1
  Part 10 and ASME Section III Division 5 (Code Case N-812 / ORNL-TM-2018/868);
  the Coffin-Manson coefficients from Fournier et al. 2008; the parabolic
  oxidation kinetics from Quadakkers et al. 2005; the creep single-point anchor
  from NIMS Creep Data Sheet 43A.
- **T2 (derived / fitted):** the Norton pre-exponential `A` is *calibrated* to
  the single NIMS 43A point, and the Larson-Miller minimum-stress polynomial is a
  3-anchor *fit* of the API 530 curve. These are labelled Calibrated / Fitted in
  the Key Assumptions table.
- **T3 (assumed operating / modelling):** internal pressure, design temperature,
  the thermal-cycle count, the 60% oxide metal-loss fraction and the 40%
  wall-loss retirement criterion are engineering assumptions, each carried with
  an **[ASSUMED]** flag in the Key Assumptions table.

No constant, equation or citation is used that is not traceable to one of the
references above; where a value is an assumption rather than a standard number it
is flagged, not presented as measured.

## Disclaimer

Research tool only. Not for design, fitness-for-service, or safety-critical decisions without site-specific calibration and independent PE review.

## License

MIT — Felipe Rocha. See [LICENSE](LICENSE).


## How to Cite

If this software contributes to your work, please cite both the software (this repository) and the underlying methods it implements.

**Software (archived release):**

> Rocha, F. (2026). *Integrity Code Series - Week 8 - High-Temperature Creep-Fatigue Interaction in Fired Heater Tubes* (Version 1.0.1) [Computer software]. Zenodo. https://doi.org/10.5281/zenodo.20172467

**BibTeX:**

```bibtex
@software{rocha_2026_week8,
  author       = {Rocha, Felipe},
  title        = {{Integrity Code Series - Week 8 - High-Temperature Creep-Fatigue Interaction in Fired Heater Tubes}},
  year         = 2026,
  publisher    = {Zenodo},
  version      = {v1.0.1},
  doi          = {10.5281/zenodo.20172467},
  url          = {https://doi.org/10.5281/zenodo.20172467}
}
```

The two DOIs Zenodo provides are:

| DOI                                  | What it points to                                                  |
|--------------------------------------|--------------------------------------------------------------------|
| `10.5281/zenodo.20172467` (concept)   | Always resolves to the latest version - use this for citation.     |
| `10.5281/zenodo.20172468` (version)   | Pinned to v1.0.1 specifically - use when reproducibility matters.  |

A machine-readable citation file is also available in [`CITATION.cff`](CITATION.cff) - GitHub will display a "Cite this repository" widget at the top right of the repo page that exports BibTeX / APA / RIS automatically.


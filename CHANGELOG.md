# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-04-09

### Added
- Norton power-law creep engine calibrated to NIMS Data Sheet 43A (T91 at 550-600C)
- MPC Omega method for tertiary creep acceleration per API 579-1 Part 10
- Larson-Miller parameter rupture prediction per API 530 7th Edition
- Parabolic oxidation model (Wagner 1933) with Quadakkers 2005 kinetics
- Coffin-Manson thermal fatigue with Fournier 2008 parameters for T91
- ASME Section III Division 5 creep-fatigue interaction envelope (Code Case N-812)
- Coupled tube life simulation with oxide-stress-creep feedback loop
- Gradient Boosted Regression surrogate model for fast life prediction
- Latin Hypercube Sampling Monte Carlo with Spearman sensitivity analysis
- SHA-256 audit chain and STRIDE threat model for cybersecurity
- 324 passing tests across 9 test files
- Hero figure, secondary plots, and animated GIF generation

### References
- Anchor: Marathon Martinez CSB Report (2025-03-13)
- API 530 7th Ed., API 579-1/ASME FFS-1 Part 10, ASME Section III Division 5
- NIMS Creep Data Sheet No. 43A (2014)
- Fournier et al. 2008, Int. J. Fatigue 30, 1797-1812
- Quadakkers et al. 2005, Oxidation of Metals 64, 175-209
- Maruyama et al. 2001 (activation energy)

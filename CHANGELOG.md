# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed
- `simulate_tube_life`: a failed run reported 0 h and zeroed the last history entry. Omega baseline 0 h -> 36,300 h. All 457 failed sweep samples had 0 h life and 100% wall loss.
- `omega_rupture_strain` -> `omega_characteristic_strain`: 1/Omega is reached at 0.632 t_r, it is not a rupture strain
- Stress beyond the LMP fit scored zero creep damage; it now counts as full damage
- README key results and sensitivity table now match the code: D_f 0.003 -> 0.006; the +40 C hot spot is inside the envelope (was "Fail"); Spearman -0.59/-0.50/+0.40/-0.15/-0.08 -> -0.53/-0.42/+0.34/0.00/-0.03
- pyproject: valid `build-backend`, so `pip install .` works; ruff target matches `requires-python`
- TEST_SUMMARY: counts match the collected suite
- Docstring of `omega_creep_rate`: `T` and `sigma` are not used

### Added
- `run_all.py` and `src/visualization.py`: every figure in `assets/` now comes from computed results and is regenerated

### Changed
- Oxidation rate constants, Omega constants, the Larson-Miller constant and fit, E and CTE are tagged `[ASSUMED]`: their cited tables and figures could not be verified
- Removed the Code Case N-812 and ORNL/TM-2018/868 cites and the Oxidation of Metals 64 cite for the same reason

### CI
- Remove the ruff format step, which always passed (`|| true`) over 19 unformatted files
- Bump checkout and setup-python to v7, cache to v6
- Dependabot keeps pip floors unless required and groups github-actions updates

## [1.0.1] - 2026-05-13

### Changed
- chore(release): prep v1.0.1 - Zenodo metadata, DOI badge, How to Cite
- chore(meta): add CI matrix, Dependabot, and Integrity Code Series cross-links
- Lint cleanup: unused imports, unused variables, lambda assignments

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
- Hero figure, secondary plots, and animated GIF

### References
- Anchor: Marathon Martinez CSB Report (2025-03-13)
- API 530 7th Ed., API 579-1/ASME FFS-1 Part 10, ASME Section III Division 5
- NIMS Creep Data Sheet No. 43A (2014)
- Fournier et al. 2008, Int. J. Fatigue 30, 1797-1812
- Quadakkers et al. 2005, Oxidation of Metals 64, 175-209
- Maruyama et al. 2001 (activation energy)

"""
Comprehensive test suite for src/monte_carlo.py.

Tests Latin Hypercube Sampling, parametric sweep, and sensitivity analysis.
"""

import numpy as np


class TestLHSSampling:
    """Tests for Latin Hypercube Sampling."""

    def test_lhs_generates_samples(self, monte_carlo):
        """Should generate LHS samples without error."""
        samples = monte_carlo.generate_lhs_samples(n_samples=100)

        assert samples is not None
        assert samples.shape == (100, 5)

    def test_lhs_default_parameters(self, monte_carlo, config):
        """Should use default n_samples and seed from config."""
        samples = monte_carlo.generate_lhs_samples()

        assert samples.shape[0] == config.MC_N_SAMPLES
        assert samples.shape[1] == 5

    def test_lhs_samples_within_bounds(self, monte_carlo, config):
        """All samples should be within specified bounds."""
        samples = monte_carlo.generate_lhs_samples(n_samples=100)

        ranges = config.MC_RANGES

        # Check temperature bounds
        assert np.all(samples[:, 0] >= ranges["temperature_K"][0])
        assert np.all(samples[:, 0] <= ranges["temperature_K"][1])

        # Check pressure bounds
        assert np.all(samples[:, 1] >= ranges["pressure_Pa"][0])
        assert np.all(samples[:, 1] <= ranges["pressure_Pa"][1])

        # Check wall thickness bounds
        assert np.all(samples[:, 2] >= ranges["wall_thickness_m"][0])
        assert np.all(samples[:, 2] <= ranges["wall_thickness_m"][1])

        # Check cycles bounds
        assert np.all(samples[:, 3] >= ranges["cycles_per_year"][0])
        assert np.all(samples[:, 3] <= ranges["cycles_per_year"][1])

        # Check delta_T bounds
        assert np.all(samples[:, 4] >= ranges["delta_T_K"][0])
        assert np.all(samples[:, 4] <= ranges["delta_T_K"][1])

    def test_lhs_correct_number_of_samples(self, monte_carlo):
        """Should generate exact number requested."""
        for n in [10, 50, 100, 500]:
            samples = monte_carlo.generate_lhs_samples(n_samples=n)
            assert samples.shape[0] == n

    def test_lhs_five_dimensions(self, monte_carlo):
        """Should generate exactly 5 features."""
        samples = monte_carlo.generate_lhs_samples(n_samples=50)
        assert samples.shape[1] == 5

    def test_lhs_reproducibility(self, monte_carlo):
        """Same seed should give same samples."""
        samples1 = monte_carlo.generate_lhs_samples(n_samples=100, seed=42)
        samples2 = monte_carlo.generate_lhs_samples(n_samples=100, seed=42)

        assert np.allclose(samples1, samples2), "Same seed should give same samples"

    def test_lhs_different_seeds_different_samples(self, monte_carlo):
        """Different seeds should give different samples."""
        samples1 = monte_carlo.generate_lhs_samples(n_samples=100, seed=42)
        samples2 = monte_carlo.generate_lhs_samples(n_samples=100, seed=43)

        assert not np.allclose(samples1, samples2), "Different seeds should differ"

    def test_lhs_coverage(self, monte_carlo, config):
        """LHS should provide good coverage of parameter space."""
        samples = monte_carlo.generate_lhs_samples(n_samples=1000)

        # Check that samples are well-distributed (no huge gaps)
        ranges = config.MC_RANGES

        # For temperature
        T_min, T_max = ranges["temperature_K"]
        T_bins = np.linspace(T_min, T_max, 11)  # 10 bins
        T_counts = np.histogram(samples[:, 0], bins=T_bins)[0]

        # Each bin should have roughly equal counts (100 per bin for 1000 samples)
        mean_count = np.mean(T_counts)
        max_deviation = np.max(np.abs(T_counts - mean_count))

        # Allow some deviation but not too much
        assert max_deviation < mean_count * 0.5, "Should have good coverage"


class TestParametricSweep:
    """Tests for full LHS parametric sweep."""

    def test_sweep_runs_without_error(self, monte_carlo):
        """Sweep should complete without error."""
        samples, results = monte_carlo.run_sweep(
            n_samples=5,
            total_hours=5000.0,
            dt_hours=500.0
        )

        assert samples is not None
        assert results is not None

    def test_sweep_returns_correct_structure(self, monte_carlo):
        """Sweep should return samples and results dict."""
        samples, results = monte_carlo.run_sweep(
            n_samples=5,
            total_hours=5000.0,
            dt_hours=500.0
        )

        assert samples.shape == (5, 5)

        required_keys = {
            "life_hours", "failure_mode", "D_f_final", "D_c_final",
            "max_creep_strain", "max_wall_loss_frac"
        }
        assert set(results.keys()) == required_keys

    def test_sweep_result_arrays_length(self, monte_carlo):
        """Result arrays should match number of samples."""
        n_samples = 10
        samples, results = monte_carlo.run_sweep(
            n_samples=n_samples,
            total_hours=5000.0,
            dt_hours=500.0
        )

        assert len(results["life_hours"]) == n_samples
        assert len(results["failure_mode"]) == n_samples
        assert len(results["D_f_final"]) == n_samples
        assert len(results["D_c_final"]) == n_samples
        assert len(results["max_creep_strain"]) == n_samples
        assert len(results["max_wall_loss_frac"]) == n_samples

    def test_sweep_life_hours_positive(self, monte_carlo):
        """All life hours should be positive."""
        samples, results = monte_carlo.run_sweep(
            n_samples=5,
            total_hours=5000.0,
            dt_hours=500.0
        )

        assert all(life > 0.0 for life in results["life_hours"])

    def test_sweep_damage_fractions_valid(self, monte_carlo):
        """Damage fractions should be in reasonable ranges."""
        samples, results = monte_carlo.run_sweep(
            n_samples=5,
            total_hours=5000.0,
            dt_hours=500.0
        )

        # D_f and D_c can be > 1 but should be reasonable
        assert all(0.0 <= df for df in results["D_f_final"])
        assert all(0.0 <= dc for dc in results["D_c_final"])
        assert all(df < 10.0 for df in results["D_f_final"])
        assert all(dc < 10.0 for dc in results["D_c_final"])

    def test_sweep_failure_modes_valid(self, monte_carlo):
        """Failure modes should be from valid set."""
        samples, results = monte_carlo.run_sweep(
            n_samples=10,
            total_hours=5000.0,
            dt_hours=500.0
        )

        valid_modes = {
            "creep_rupture",
            "oxidation_wall_loss",
            "creep_fatigue_interaction",
            "survived"
        }

        for mode in results["failure_mode"]:
            assert mode in valid_modes, f"Invalid failure mode: {mode}"

    def test_sweep_with_omega_method(self, monte_carlo):
        """Sweep with Omega method should work."""
        samples, results = monte_carlo.run_sweep(
            n_samples=3,
            use_omega=True,
            total_hours=5000.0,
            dt_hours=500.0
        )

        assert len(results["life_hours"]) == 3

    def test_sweep_reproducibility(self, monte_carlo):
        """Same seed should give same results."""
        samples1, results1 = monte_carlo.run_sweep(
            n_samples=5,
            seed=42,
            total_hours=5000.0,
            dt_hours=500.0
        )

        samples2, results2 = monte_carlo.run_sweep(
            n_samples=5,
            seed=42,
            total_hours=5000.0,
            dt_hours=500.0
        )

        # Samples should be identical
        assert np.allclose(samples1, samples2)

        # Life hours should be very close (may vary slightly due to numerical errors)
        assert np.allclose(results1["life_hours"], results2["life_hours"], rtol=0.01)


class TestSensitivityAnalysis:
    """Tests for Spearman rank correlation sensitivity analysis."""

    def test_sensitivity_analysis_returns_dict(self, monte_carlo):
        """Should return dictionary of feature names to correlations."""
        samples = monte_carlo.generate_lhs_samples(n_samples=100)
        life = np.random.rand(100) * 100000 + 10000

        correlations = monte_carlo.sensitivity_analysis(samples, life)

        assert isinstance(correlations, dict)
        assert len(correlations) == 5

    def test_sensitivity_has_all_features(self, monte_carlo):
        """Should include all 5 features."""
        samples = monte_carlo.generate_lhs_samples(n_samples=100)
        life = np.random.rand(100) * 100000 + 10000

        correlations = monte_carlo.sensitivity_analysis(samples, life)

        expected_features = [
            "temperature_K", "pressure_Pa", "wall_thickness_m",
            "cycles_per_year", "delta_T_K"
        ]

        for feature in expected_features:
            assert feature in correlations

    def test_correlations_in_valid_range(self, monte_carlo):
        """Correlations should be in [-1, 1]."""
        samples = monte_carlo.generate_lhs_samples(n_samples=100)
        life = np.random.rand(100) * 100000 + 10000

        correlations = monte_carlo.sensitivity_analysis(samples, life)

        for feature, rho in correlations.items():
            assert -1.0 <= rho <= 1.0, f"Invalid correlation for {feature}"

    def test_sensitivity_temperature_correlation(self, monte_carlo, config):
        """Temperature should show negative correlation with life."""
        # Create data where life decreases with temperature
        samples = monte_carlo.generate_lhs_samples(n_samples=200)
        ranges = config.MC_RANGES

        # Make life inversely proportional to temperature
        T_normalized = (samples[:, 0] - ranges["temperature_K"][0]) / \
                       (ranges["temperature_K"][1] - ranges["temperature_K"][0])
        life = 200000 * (1.0 - T_normalized) + np.random.rand(200) * 10000

        correlations = monte_carlo.sensitivity_analysis(samples, life)

        # Temperature should have negative correlation
        assert correlations["temperature_K"] < 0, \
            "Temperature should negatively correlate with life"

    def test_sensitivity_pressure_correlation(self, monte_carlo, config):
        """Pressure should show negative correlation with life."""
        samples = monte_carlo.generate_lhs_samples(n_samples=200)
        ranges = config.MC_RANGES

        # Make life inversely proportional to pressure
        P_normalized = (samples[:, 1] - ranges["pressure_Pa"][0]) / \
                       (ranges["pressure_Pa"][1] - ranges["pressure_Pa"][0])
        life = 200000 * (1.0 - P_normalized) + np.random.rand(200) * 10000

        correlations = monte_carlo.sensitivity_analysis(samples, life)

        # Pressure should have negative correlation
        assert correlations["pressure_Pa"] < 0, \
            "Pressure should negatively correlate with life"

    def test_sensitivity_wall_thickness_correlation(self, monte_carlo, config):
        """Wall thickness should show positive correlation with life."""
        samples = monte_carlo.generate_lhs_samples(n_samples=200)
        ranges = config.MC_RANGES

        # Make life proportional to wall thickness
        wt_normalized = (samples[:, 2] - ranges["wall_thickness_m"][0]) / \
                        (ranges["wall_thickness_m"][1] - ranges["wall_thickness_m"][0])
        life = 50000 + wt_normalized * 200000 + np.random.rand(200) * 10000

        correlations = monte_carlo.sensitivity_analysis(samples, life)

        # Wall thickness should have positive correlation
        assert correlations["wall_thickness_m"] > 0, \
            "Wall thickness should positively correlate with life"

    def test_sensitivity_random_data(self, monte_carlo):
        """Random data should give correlations near zero."""
        np.random.seed(42)
        samples = monte_carlo.generate_lhs_samples(n_samples=100, seed=42)
        life = np.random.rand(100)  # Uncorrelated random

        correlations = monte_carlo.sensitivity_analysis(samples, life)

        # All correlations should be small (near zero)
        for rho in correlations.values():
            assert abs(rho) < 0.3, "Random data should show weak correlation"


class TestParameterRangeValidity:
    """Tests for parameter range definitions."""

    def test_temperature_range_realistic(self, monte_carlo, config):
        """Temperature range should be reasonable."""
        T_min, T_max = config.MC_RANGES["temperature_K"]

        # Should be in 550-650 C range (823-923 K)
        assert 800.0 < T_min < 900.0
        assert 800.0 < T_max < 950.0
        assert T_max > T_min

    def test_pressure_range_realistic(self, monte_carlo, config):
        """Pressure range should be reasonable for heater tubes."""
        P_min, P_max = config.MC_RANGES["pressure_Pa"]

        # Should be in 3-7 MPa range
        assert 2.0e6 < P_min < 5.0e6
        assert 5.0e6 < P_max < 10.0e6
        assert P_max > P_min

    def test_wall_thickness_range_realistic(self, monte_carlo, config):
        """Wall thickness range should be reasonable."""
        wt_min, wt_max = config.MC_RANGES["wall_thickness_m"]

        # Typically 4-7 mm for heater tubes
        assert 0.003 < wt_min < 0.006
        assert 0.006 < wt_max < 0.008
        assert wt_max > wt_min

    def test_cycles_per_year_realistic(self, monte_carlo, config):
        """Cycles per year should be reasonable."""
        cyc_min, cyc_max = config.MC_RANGES["cycles_per_year"]

        # Typically 4-15 per year
        assert 1.0 < cyc_min < 10.0
        assert 5.0 < cyc_max < 20.0
        assert cyc_max > cyc_min

    def test_delta_t_range_realistic(self, monte_carlo, config):
        """Thermal cycle amplitude should be reasonable."""
        dT_min, dT_max = config.MC_RANGES["delta_T_K"]

        # Typically 200-400 K
        assert 100.0 < dT_min < 300.0
        assert 200.0 < dT_max < 500.0
        assert dT_max > dT_min


class TestSweepEdgeCases:
    """Edge cases for parametric sweep."""

    def test_sweep_single_sample(self, monte_carlo):
        """Should handle single sample."""
        samples, results = monte_carlo.run_sweep(
            n_samples=1,
            total_hours=5000.0,
            dt_hours=500.0
        )

        assert len(results["life_hours"]) == 1

    def test_sweep_large_sample_set(self, monte_carlo):
        """Should handle larger sample set (but use short simulation)."""
        samples, results = monte_carlo.run_sweep(
            n_samples=50,
            total_hours=1000.0,  # Reduce simulation time
            dt_hours=500.0
        )

        assert len(results["life_hours"]) == 50

    def test_sweep_very_short_simulation(self, monte_carlo):
        """Should handle very short simulation times."""
        samples, results = monte_carlo.run_sweep(
            n_samples=3,
            total_hours=100.0,
            dt_hours=50.0
        )

        assert len(results["life_hours"]) == 3


class TestSensitivityConsistency:
    """Consistency checks for sensitivity analysis."""

    def test_sensitivity_many_samples(self, monte_carlo):
        """Sensitivity with many samples should converge."""
        samples = monte_carlo.generate_lhs_samples(n_samples=500)

        # Create synthetic life data with clear temperature dependence
        life = 200000 - samples[:, 0] * 100  # Strong T dependence

        correlations = monte_carlo.sensitivity_analysis(samples, life)

        # Temperature should have strong negative correlation
        assert correlations["temperature_K"] < -0.8, \
            "Strong temperature dependence should be detected"

    def test_sensitivity_many_samples_weak_effect(self, monte_carlo):
        """Weak effects should be detected with many samples."""
        samples = monte_carlo.generate_lhs_samples(n_samples=500)

        # Create life with weak correlation to all features
        life = 100000 + np.random.rand(500) * 50000

        correlations = monte_carlo.sensitivity_analysis(samples, life)

        # All correlations should be weak
        for rho in correlations.values():
            assert abs(rho) < 0.3

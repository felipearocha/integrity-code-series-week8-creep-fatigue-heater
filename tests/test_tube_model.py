"""
Comprehensive test suite for src/tube_model.py.

Tests the integrated fired heater tube life model that couples creep, fatigue, and oxidation.
Includes baseline runs, parameter sensitivity, physics monotonicity, and failure modes.
"""

import numpy as np


class TestBaselineSimulation:
    """Tests for baseline tube life simulation."""

    def test_baseline_runs_without_error(self, tube_model):
        """Baseline simulation should complete successfully."""
        result = tube_model.run_baseline()

        assert result is not None
        assert result.failure_mode is not None

    def test_baseline_with_omega_runs(self, tube_model):
        """Baseline simulation with Omega method should run."""
        result = tube_model.run_baseline_omega()

        assert result is not None
        assert result.failure_mode is not None

    def test_result_has_all_fields(self, tube_model):
        """TubeLifeResult should have all output fields populated."""
        result = tube_model.simulate_tube_life(total_hours=1000.0, dt_hours=100.0)

        assert result.times_hours is not None
        assert result.wall_thickness is not None
        assert result.hoop_stress is not None
        assert result.creep_strain is not None
        assert result.creep_rate is not None
        assert result.oxide_thickness is not None
        assert result.D_fatigue is not None
        assert result.D_creep is not None
        assert result.within_envelope is not None
        assert result.failure_mode is not None
        assert result.failure_time_hours is not None
        assert result.temperature_K is not None
        assert result.pressure_Pa is not None

    def test_simulation_output_arrays_correct_length(self, tube_model):
        """Output arrays should all have same length."""
        result = tube_model.simulate_tube_life(total_hours=1000.0, dt_hours=100.0)

        n = len(result.times_hours)
        assert len(result.wall_thickness) == n
        assert len(result.hoop_stress) == n
        assert len(result.creep_strain) == n
        assert len(result.creep_rate) == n
        assert len(result.oxide_thickness) == n
        assert len(result.D_fatigue) == n
        assert len(result.D_creep) == n
        assert len(result.within_envelope) == n


class TestPhysicsMonotonicity:
    """Physics monotonicity and consistency checks."""

    def test_stress_increases_as_wall_thins(self, tube_model):
        """Hoop stress should increase as wall thickness decreases (due to oxidation)."""
        result = tube_model.simulate_tube_life(total_hours=50000.0, dt_hours=1000.0)

        # At beginning, wall is thicker
        stress_early = result.hoop_stress[0]
        # Later, wall has thinned due to oxidation
        stress_late = result.hoop_stress[-1]

        # Stress should increase (wall thinner)
        assert stress_late > stress_early, \
            "Hoop stress should increase as wall thins from oxidation"

    def test_wall_thickness_decreases(self, tube_model):
        """Effective wall thickness should decrease monotonically."""
        result = tube_model.simulate_tube_life(total_hours=50000.0, dt_hours=1000.0)

        for i in range(len(result.wall_thickness)-1):
            assert result.wall_thickness[i+1] <= result.wall_thickness[i] + 1e-10, \
                "Wall thickness must decrease monotonically"

    def test_oxide_thickness_increases(self, tube_model):
        """Oxide thickness should increase monotonically."""
        result = tube_model.simulate_tube_life(total_hours=50000.0, dt_hours=1000.0)

        for i in range(len(result.oxide_thickness)-1):
            assert result.oxide_thickness[i+1] >= result.oxide_thickness[i] - 1e-10, \
                "Oxide thickness must increase monotonically"

    def test_creep_strain_increases(self, tube_model):
        """Creep strain should increase monotonically."""
        result = tube_model.simulate_tube_life(total_hours=10000.0, dt_hours=1000.0)

        for i in range(len(result.creep_strain)-1):
            assert result.creep_strain[i+1] >= result.creep_strain[i] - 1e-10, \
                "Creep strain must increase monotonically"

    def test_fatigue_damage_increases(self, tube_model):
        """Fatigue damage fraction should increase monotonically."""
        result = tube_model.simulate_tube_life(total_hours=10000.0, dt_hours=1000.0)

        for i in range(len(result.D_fatigue)-1):
            assert result.D_fatigue[i+1] >= result.D_fatigue[i] - 1e-10, \
                "Fatigue damage must increase monotonically"

    def test_creep_damage_increases(self, tube_model):
        """Creep damage fraction should increase monotonically."""
        result = tube_model.simulate_tube_life(total_hours=10000.0, dt_hours=1000.0)

        for i in range(len(result.D_creep)-1):
            assert result.D_creep[i+1] >= result.D_creep[i] - 1e-10, \
                "Creep damage must increase monotonically"


class TestParameterSensitivity:
    """Tests for sensitivity to input parameters."""

    def test_higher_temperature_reduces_life(self, tube_model):
        """Higher service temperature should reduce tube life."""
        result_low = tube_model.simulate_tube_life(
            T=823.15, total_hours=100000.0, dt_hours=1000.0
        )
        result_high = tube_model.simulate_tube_life(
            T=923.15, total_hours=100000.0, dt_hours=1000.0
        )

        assert result_high.failure_time_hours < result_low.failure_time_hours, \
            "Higher temperature should reduce life"

    def test_higher_pressure_reduces_life(self, tube_model):
        """Higher internal pressure should reduce tube life."""
        result_low = tube_model.simulate_tube_life(
            pressure=3.0e6, total_hours=100000.0, dt_hours=1000.0
        )
        result_high = tube_model.simulate_tube_life(
            pressure=7.0e6, total_hours=100000.0, dt_hours=1000.0
        )

        # Both may survive 100kh at design T; compare creep damage instead
        assert result_high.D_creep[-1] > result_low.D_creep[-1], \
            "Higher pressure should accumulate more creep damage"

    def test_thinner_wall_reduces_life(self, tube_model):
        """Thinner initial wall should increase creep damage (higher stress)."""
        result_thick = tube_model.simulate_tube_life(
            wt_initial=0.0070, total_hours=100000.0, dt_hours=1000.0
        )
        result_thin = tube_model.simulate_tube_life(
            wt_initial=0.0040, total_hours=100000.0, dt_hours=1000.0
        )

        # Thinner wall -> higher stress -> more creep damage
        assert result_thin.D_creep[-1] > result_thick.D_creep[-1], \
            "Thinner wall should accumulate more creep damage"

    def test_more_thermal_cycles_increases_fatigue_damage(self, tube_model):
        """More thermal cycles per year should increase fatigue damage."""
        result_low_cycles = tube_model.simulate_tube_life(
            cycles_per_year=4.0, total_hours=50000.0, dt_hours=1000.0
        )
        result_high_cycles = tube_model.simulate_tube_life(
            cycles_per_year=15.0, total_hours=50000.0, dt_hours=1000.0
        )

        # Fatigue damage should be higher with more cycles
        assert result_high_cycles.D_fatigue[-1] > result_low_cycles.D_fatigue[-1], \
            "More cycles should increase fatigue damage"

    def test_larger_delta_T_increases_fatigue_damage(self, tube_model):
        """Larger thermal cycle amplitude should increase fatigue damage."""
        result_small_swing = tube_model.simulate_tube_life(
            delta_T=200.0, total_hours=50000.0, dt_hours=1000.0
        )
        result_large_swing = tube_model.simulate_tube_life(
            delta_T=380.0, total_hours=50000.0, dt_hours=1000.0
        )

        # Fatigue damage should be higher with larger swings
        assert result_large_swing.D_fatigue[-1] > result_small_swing.D_fatigue[-1], \
            "Larger delta_T should increase fatigue damage"


class TestFailureModes:
    """Tests for different failure mechanisms."""

    def test_failure_mode_assigned(self, tube_model):
        """Simulation should assign a failure mode."""
        result = tube_model.simulate_tube_life(total_hours=100000.0, dt_hours=1000.0)

        assert result.failure_mode in [
            "creep_rupture",
            "oxidation_wall_loss",
            "creep_fatigue_interaction",
            "survived"
        ], f"Invalid failure mode: {result.failure_mode}"

    def test_creep_rupture_possible(self, tube_model, config):
        """High stress should trigger creep rupture."""
        # Very high pressure and high temperature
        result = tube_model.simulate_tube_life(
            T=923.15,
            pressure=7.0e6,
            wt_initial=0.005,
            total_hours=100000.0,
            dt_hours=500.0,
            delta_T=100.0  # Low cycles to avoid fatigue failure
        )

        # Should fail at some point
        assert result.failure_time_hours < config.SIM_TOTAL_HOURS

    def test_oxidation_wall_loss_possible(self, tube_model, config):
        """Very long operation at high temperature should trigger wall loss."""
        result = tube_model.simulate_tube_life(
            T=923.15,
            wt_initial=0.005,  # Thin initial wall
            total_hours=1000000.0,  # Very long time
            dt_hours=5000.0
        )

        # Should eventually fail
        assert result.failure_time_hours < config.SIM_TOTAL_HOURS


class TestTimeArrayConsistency:
    """Tests for time array properties."""

    def test_times_start_at_zero(self, tube_model):
        """Time array should start at t=0."""
        result = tube_model.simulate_tube_life(total_hours=10000.0, dt_hours=100.0)

        assert result.times_hours[0] == 0.0, "First time should be zero"

    def test_times_increase_monotonically(self, tube_model):
        """Time should increase monotonically."""
        result = tube_model.simulate_tube_life(total_hours=10000.0, dt_hours=100.0)

        for i in range(len(result.times_hours)-1):
            assert result.times_hours[i+1] > result.times_hours[i], \
                "Time must increase monotonically"

    def test_failure_time_consistent(self, tube_model):
        """Failure time should match last time in array."""
        result = tube_model.simulate_tube_life(total_hours=10000.0, dt_hours=100.0)

        last_time = result.times_hours[-1]
        assert abs(result.failure_time_hours - last_time) < 1.0, \
            "Failure time should match last array entry"


class TestOmegaIntegration:
    """Tests for Omega creep method integration."""

    def test_omega_simulation_runs(self, tube_model):
        """Simulation with Omega method should complete."""
        result = tube_model.simulate_tube_life(
            use_omega=True,
            total_hours=10000.0,
            dt_hours=100.0
        )

        assert result is not None
        assert len(result.creep_strain) > 0

    def test_omega_accelerates_creep(self, tube_model):
        """Omega method should accelerate creep toward rupture."""
        result_omega = tube_model.simulate_tube_life(
            T=873.15,
            pressure=5.0e6,
            use_omega=True,
            total_hours=100000.0,
            dt_hours=1000.0
        )

        result_norton = tube_model.simulate_tube_life(
            T=873.15,
            pressure=5.0e6,
            use_omega=False,
            total_hours=100000.0,
            dt_hours=1000.0
        )

        # Omega method should reach higher final strain or fail earlier
        assert result_omega.creep_strain[-1] >= result_norton.creep_strain[-1], \
            "Omega should accelerate tertiary creep"


class TestCustomParameters:
    """Tests for custom parameter specification."""

    def test_custom_temperature(self, tube_model):
        """Should respect custom temperature parameter."""
        T_custom = 850.0
        result = tube_model.simulate_tube_life(T=T_custom, total_hours=1000.0, dt_hours=100.0)

        assert result.temperature_K == T_custom

    def test_custom_pressure(self, tube_model):
        """Should respect custom pressure parameter."""
        P_custom = 4.5e6
        result = tube_model.simulate_tube_life(pressure=P_custom, total_hours=1000.0, dt_hours=100.0)

        assert result.pressure_Pa == P_custom

    def test_custom_wall_thickness(self, tube_model):
        """Initial wall thickness should be respected."""
        wt_custom = 0.006
        result = tube_model.simulate_tube_life(wt_initial=wt_custom, total_hours=1000.0, dt_hours=100.0)

        # First value should match
        assert abs(result.wall_thickness[0] - wt_custom) < 1.0e-10

    def test_custom_thermal_cycle_params(self, tube_model):
        """Should respect custom thermal cycle parameters."""
        delta_T_custom = 250.0
        cycles_custom = 6.0

        result = tube_model.simulate_tube_life(
            delta_T=delta_T_custom,
            cycles_per_year=cycles_custom,
            total_hours=10000.0,
            dt_hours=1000.0
        )

        # Fatigue damage should reflect these parameters
        assert len(result.D_fatigue) > 0


class TestNumericalStability:
    """Tests for numerical stability and edge cases."""

    def test_fine_time_stepping(self, tube_model):
        """Should handle fine time steps without error."""
        result = tube_model.simulate_tube_life(
            total_hours=10000.0,
            dt_hours=10.0  # Fine time step
        )

        assert len(result.times_hours) > 500
        assert result.failure_time_hours >= 0.0

    def test_coarse_time_stepping(self, tube_model):
        """Should handle coarse time steps."""
        result = tube_model.simulate_tube_life(
            total_hours=100000.0,
            dt_hours=5000.0  # Coarse time step
        )

        assert len(result.times_hours) > 0
        assert result.failure_time_hours >= 0.0

    def test_very_short_simulation(self, tube_model):
        """Should handle very short simulations."""
        result = tube_model.simulate_tube_life(
            total_hours=100.0,
            dt_hours=10.0
        )

        assert len(result.times_hours) > 0
        assert all(np.isfinite(result.hoop_stress))

    def test_array_values_finite(self, tube_model):
        """All output arrays should contain finite values."""
        result = tube_model.simulate_tube_life(total_hours=10000.0, dt_hours=1000.0)

        assert all(np.isfinite(result.times_hours))
        assert all(np.isfinite(result.wall_thickness))
        assert all(np.isfinite(result.hoop_stress))
        assert all(np.isfinite(result.creep_strain))
        assert all(np.isfinite(result.creep_rate))
        assert all(np.isfinite(result.oxide_thickness))
        assert all(np.isfinite(result.D_fatigue))
        assert all(np.isfinite(result.D_creep))


class TestStressComputations:
    """Tests for stress computation within simulation."""

    def test_stress_positive_throughout(self, tube_model):
        """Hoop stress should be positive throughout simulation."""
        result = tube_model.simulate_tube_life(total_hours=10000.0, dt_hours=1000.0)

        assert all(s > 0.0 for s in result.hoop_stress), \
            "Hoop stress must be positive"

    def test_stress_monotonic_increase(self, tube_model):
        """Hoop stress should increase as wall thins."""
        result = tube_model.simulate_tube_life(total_hours=50000.0, dt_hours=1000.0)

        # Generally should increase (may have small fluctuations due to discretization)
        first_stress = result.hoop_stress[0]
        last_stress = result.hoop_stress[-1]

        assert last_stress >= first_stress, "Stress should increase overall"


class TestCreepStrainLimits:
    """Tests for creep strain failure criterion."""

    def test_creep_strain_respects_max_criterion(self, tube_model, config):
        """Final creep strain should not greatly exceed maximum."""
        result = tube_model.simulate_tube_life(total_hours=100000.0, dt_hours=1000.0)

        # If failed by creep, should be near max
        if "creep" in result.failure_mode:
            assert result.creep_strain[-1] <= config.SIM_MAX_CREEP_STRAIN * 1.01

    def test_creep_rates_positive(self, tube_model):
        """All creep rates should be non-negative."""
        result = tube_model.simulate_tube_life(total_hours=10000.0, dt_hours=1000.0)

        assert all(r >= 0.0 for r in result.creep_rate), \
            "Creep rates must be non-negative"


class TestWallThicknessLimits:
    """Tests for wall thickness failure criterion."""

    def test_wall_respects_max_loss(self, tube_model, config):
        """Effective wall should not be less than (1-max_loss) * initial."""
        result = tube_model.simulate_tube_life(total_hours=100000.0, dt_hours=1000.0)

        wt_min_allowed = result.wall_thickness[0] * (1.0 - config.SIM_MAX_WALL_LOSS_FRACTION)

        # If failed by oxidation, should be near limit
        if "oxidation" in result.failure_mode:
            assert result.wall_thickness[-1] >= wt_min_allowed * 0.99


class TestEnvelopeComplianceIntegration:
    """Tests for creep-fatigue envelope compliance within simulation."""

    def test_envelope_respected_until_failure(self, tube_model):
        """Points should be within envelope until failure."""
        result = tube_model.simulate_tube_life(total_hours=50000.0, dt_hours=1000.0)

        # Count how many points violate envelope
        violations = 0
        for within in result.within_envelope:
            if not within:
                violations += 1

        # Should have violations only at or near end
        if violations > 0:
            # Last few points may be outside
            assert violations <= 5, "Violations should be only at end"

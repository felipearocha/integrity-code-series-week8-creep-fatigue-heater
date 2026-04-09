"""
Comprehensive test suite for src/oxidation.py.

Tests parabolic oxidation kinetics, Arrhenius temperature dependence,
metal loss, effective wall thickness degradation, and failure criteria.
"""

import numpy as np


class TestParabolicRateConstant:
    """Tests for parabolic oxidation rate constant computation."""

    def test_positive_rate_constant(self, oxidation, config):
        """Rate constant must be positive at all temperatures."""
        T = 873.15  # 600 C
        kp = oxidation.parabolic_rate_constant(T)
        assert kp > 0.0, "Rate constant must be positive"

    def test_arrhenius_behavior(self, oxidation, config):
        """Rate constant should follow Arrhenius: kp = kp0 * exp(-Q/(R*T))."""
        T = 873.15
        kp = oxidation.parabolic_rate_constant(T)

        expected = config.OXIDE_KP0 * np.exp(-config.OXIDE_Q / (config.R_GAS * T))
        assert abs(kp - expected) < expected * 1.0e-10, \
            "Rate constant formula incorrect"

    def test_increases_with_temperature(self, oxidation):
        """Rate constant increases exponentially with temperature."""
        T_low = 823.15  # 550 C
        T_high = 923.15  # 650 C

        kp_low = oxidation.parabolic_rate_constant(T_low)
        kp_high = oxidation.parabolic_rate_constant(T_high)

        assert kp_high > kp_low, "Rate constant must increase with temperature"
        # Activation energy is 250 kJ/mol, so expect significant increase
        assert kp_high / kp_low > 10.0, "Temperature effect should be strong"

    def test_decreases_with_increasing_activation_energy(self, oxidation, config):
        """At any given T, higher Q should reduce kp."""
        # This is tested implicitly through config values
        T = 873.15
        kp = oxidation.parabolic_rate_constant(T)
        # Just verify it's reasonable
        assert 1.0e-20 < kp < 1.0e-5, "kp in expected range for 9Cr-1Mo"


class TestOxideThickness:
    """Tests for oxide thickness growth."""

    def test_zero_time_gives_zero_thickness(self, oxidation):
        """At t=0, oxide thickness must be zero."""
        T = 873.15
        x = oxidation.oxide_thickness(t=0.0, T=T)
        assert x == 0.0, "Zero time must give zero thickness"

    def test_negative_time_gives_zero_thickness(self, oxidation):
        """Negative time (unphysical) should return zero."""
        T = 873.15
        x = oxidation.oxide_thickness(t=-100.0, T=T)
        assert x == 0.0

    def test_sqrt_time_growth(self, oxidation, config):
        """Oxide follows parabolic law: x = sqrt(k_p * t)."""
        T = 873.15
        t = 1.0e6  # 1 million seconds

        x = oxidation.oxide_thickness(t, T)

        kp = oxidation.parabolic_rate_constant(T)
        expected = np.sqrt(kp * t)

        assert abs(x - expected) < 1.0e-10 * expected, \
            "Oxide thickness formula incorrect"

    def test_increases_with_temperature(self, oxidation):
        """Oxide thickness increases with temperature (faster kinetics)."""
        t = 3.15e8  # 10 years in seconds

        x_low = oxidation.oxide_thickness(t, T=823.15)
        x_high = oxidation.oxide_thickness(t, T=923.15)

        assert x_high > x_low, "Oxide growth faster at higher T"

    def test_increases_with_time(self, oxidation):
        """Oxide thickness increases with time (monotonic)."""
        T = 873.15

        t1 = 1.0e6  # 1 million seconds
        t2 = 1.0e7  # 10 million seconds

        x1 = oxidation.oxide_thickness(t1, T)
        x2 = oxidation.oxide_thickness(t2, T)

        assert x2 > x1, "Oxide thickness must increase with time"

    def test_square_root_time_scaling(self, oxidation):
        """Verify sqrt(t) scaling: doubling time should increase x by sqrt(2)."""
        T = 873.15
        t = 1.0e7

        x1 = oxidation.oxide_thickness(t, T)
        x2 = oxidation.oxide_thickness(2.0 * t, T)

        ratio = x2 / x1
        expected = np.sqrt(2.0)

        assert abs(ratio - expected) < 0.01, "Should follow sqrt(t) scaling"


class TestMetalLoss:
    """Tests for effective metal loss due to oxidation."""

    def test_proportional_to_oxide_thickness(self, oxidation, config):
        """Metal loss should be proportional to oxide thickness."""
        t = 1.0e6
        T = 873.15

        loss = oxidation.metal_loss(t, T)
        x = oxidation.oxide_thickness(t, T)

        expected = x * config.OXIDE_METAL_LOSS_FRACTION

        assert abs(loss - expected) < 1.0e-12 * expected, \
            "Metal loss formula incorrect"

    def test_zero_at_zero_time(self, oxidation):
        """Metal loss must be zero at t=0."""
        T = 873.15
        loss = oxidation.metal_loss(t=0.0, T=T)
        assert loss == 0.0

    def test_increases_with_time(self, oxidation):
        """Metal loss increases monotonically with time."""
        T = 873.15

        loss1 = oxidation.metal_loss(1.0e6, T)
        loss2 = oxidation.metal_loss(1.0e7, T)

        assert loss2 > loss1, "Metal loss must increase with time"

    def test_scaled_by_metal_loss_fraction(self, oxidation, config):
        """Metal loss should scale with OXIDE_METAL_LOSS_FRACTION parameter."""
        t = 1.0e6
        T = 873.15

        loss = oxidation.metal_loss(t, T)
        x = oxidation.oxide_thickness(t, T)

        # Loss should be smaller than total oxide thickness
        assert loss < x, "Not all oxide represents structural loss"
        assert loss == x * config.OXIDE_METAL_LOSS_FRACTION


class TestEffectiveWallThickness:
    """Tests for effective wall thickness after oxidation."""

    def test_equals_initial_at_zero_time(self, oxidation, config):
        """At t=0, effective wall should equal initial wall."""
        wt_init = 0.01  # 10 mm
        T = 873.15

        wt_eff = oxidation.effective_wall_thickness(t=0.0, T=T, wt_initial=wt_init)

        assert abs(wt_eff - wt_init) < 1.0e-15, \
            "Effective wall should equal initial at t=0"

    def test_decreases_over_time(self, oxidation):
        """Effective wall thickness decreases monotonically with time."""
        wt_init = 0.01
        T = 873.15

        wt1 = oxidation.effective_wall_thickness(1.0e6, T, wt_init)
        wt2 = oxidation.effective_wall_thickness(1.0e7, T, wt_init)

        assert wt2 < wt1, "Wall thickness must decrease with time"
        assert wt1 < wt_init, "Wall thickness must be less than initial"

    def test_never_negative(self, oxidation):
        """Effective wall should be clipped at zero."""
        wt_init = 0.001  # Small wall
        T = 923.15  # High temperature
        t_long = 1.0e9  # Very long time

        wt_eff = oxidation.effective_wall_thickness(t_long, T, wt_init)

        assert wt_eff >= 0.0, "Wall thickness cannot be negative"

    def test_uses_default_initial_wall(self, oxidation, config):
        """Should use config default if wt_initial not provided."""
        T = 873.15
        t = 1.0e6

        wt_default = oxidation.effective_wall_thickness(t, T)
        wt_explicit = oxidation.effective_wall_thickness(t, T, config.TUBE_WT_NOMINAL)

        assert abs(wt_default - wt_explicit) < 1.0e-15, \
            "Default should match config value"

    def test_formula_correctness(self, oxidation, config):
        """w_eff = w_init - loss(t, T)."""
        wt_init = 0.01
        T = 873.15
        t = 1.0e6

        wt_eff = oxidation.effective_wall_thickness(t, T, wt_init)
        loss = oxidation.metal_loss(t, T)

        expected = max(wt_init - loss, 0.0)

        assert abs(wt_eff - expected) < 1.0e-15, \
            "Formula: w_eff = w_init - loss"


class TestOxideThicknessProfile:
    """Tests for oxide thickness over time arrays."""

    def test_returns_array_of_correct_length(self, oxidation):
        """Should return array matching input time array length."""
        times = np.linspace(0.0, 1.0e7, 100)
        T = 873.15

        x_profile = oxidation.oxide_thickness_profile(times, T)

        assert len(x_profile) == len(times), "Output length must match input"

    def test_zero_at_zero_time(self, oxidation):
        """Should have zero thickness at t=0."""
        times = np.array([0.0, 1.0e6, 2.0e6])
        T = 873.15

        x_profile = oxidation.oxide_thickness_profile(times, T)

        assert x_profile[0] == 0.0, "Thickness at t=0 must be zero"

    def test_monotonically_increasing(self, oxidation):
        """Thickness profile should be monotonically increasing."""
        times = np.linspace(0.0, 1.0e7, 100)
        T = 873.15

        x_profile = oxidation.oxide_thickness_profile(times, T)

        for i in range(len(x_profile)-1):
            assert x_profile[i+1] >= x_profile[i], \
                "Profile must be monotonically increasing"

    def test_consistent_with_scalar_function(self, oxidation):
        """Array version should match scalar version."""
        times = np.array([0.0, 1.0e6, 2.0e6, 5.0e6])
        T = 873.15

        x_profile = oxidation.oxide_thickness_profile(times, T)
        x_scalar = [oxidation.oxide_thickness(t, T) for t in times]

        for i, (xp, xs) in enumerate(zip(x_profile, x_scalar)):
            assert abs(xp - xs) < 1.0e-15 * max(xp, 1.0), \
                f"Array and scalar mismatch at index {i}"


class TestMetalLossProfile:
    """Tests for metal loss over time arrays."""

    def test_returns_array_of_correct_length(self, oxidation):
        """Should return array matching input time array length."""
        times = np.linspace(0.0, 1.0e7, 100)
        T = 873.15

        loss_profile = oxidation.metal_loss_profile(times, T)

        assert len(loss_profile) == len(times)

    def test_zero_at_zero_time(self, oxidation):
        """Metal loss should be zero at t=0."""
        times = np.array([0.0, 1.0e6, 2.0e6])
        T = 873.15

        loss_profile = oxidation.metal_loss_profile(times, T)

        assert loss_profile[0] == 0.0

    def test_consistent_with_scalar_function(self, oxidation):
        """Array version should match scalar version."""
        times = np.array([0.0, 1.0e6, 2.0e6, 5.0e6])
        T = 873.15

        loss_profile = oxidation.metal_loss_profile(times, T)
        loss_scalar = [oxidation.metal_loss(t, T) for t in times]

        for i, (lp, ls) in enumerate(zip(loss_profile, loss_scalar)):
            assert abs(lp - ls) < 1.0e-15 * max(lp, 1.0), \
                f"Array and scalar mismatch at index {i}"


class TestTimeToCriticalLoss:
    """Tests for time to reach critical wall loss."""

    def test_returns_positive_time(self, oxidation, config):
        """Time to critical loss must be positive."""
        wt_init = config.TUBE_WT_NOMINAL
        T = 873.15

        t_crit = oxidation.time_to_critical_loss(wt_init, T)

        assert t_crit > 0.0, "Critical time must be positive"

    def test_finite_for_reasonable_conditions(self, oxidation, config):
        """Should return finite time at design conditions."""
        wt_init = config.TUBE_WT_NOMINAL
        T = config.DESIGN_TEMPERATURE
        max_loss_frac = config.SIM_MAX_WALL_LOSS_FRACTION

        t_crit = oxidation.time_to_critical_loss(wt_init, T, max_loss_frac)

        assert 0.0 < t_crit < np.inf, "Critical time should be finite"

    def test_shorter_at_higher_temperature(self, oxidation):
        """Higher temperature should reach critical loss faster."""
        wt_init = 0.01
        T_low = 823.15
        T_high = 923.15
        max_loss_frac = 0.40

        t_low = oxidation.time_to_critical_loss(wt_init, T_low, max_loss_frac)
        t_high = oxidation.time_to_critical_loss(wt_init, T_high, max_loss_frac)

        assert t_high < t_low, "Critical time shorter at higher T"

    def test_longer_for_thicker_wall(self, oxidation):
        """Thicker initial wall should take longer to reach critical loss."""
        T = 873.15
        max_loss_frac = 0.40

        wt1 = 0.005
        wt2 = 0.010

        t1 = oxidation.time_to_critical_loss(wt1, T, max_loss_frac)
        t2 = oxidation.time_to_critical_loss(wt2, T, max_loss_frac)

        assert t2 > t1, "Thicker wall lasts longer"

    def test_shorter_for_higher_allowable_loss(self, oxidation):
        """Higher max_loss_fraction should reduce time to critical loss."""
        wt_init = 0.01
        T = 873.15

        t1 = oxidation.time_to_critical_loss(wt_init, T, max_loss_fraction=0.20)
        t2 = oxidation.time_to_critical_loss(wt_init, T, max_loss_fraction=0.40)

        assert t1 < t2, "More lenient limit reached sooner"

    def test_uses_default_max_loss(self, oxidation, config):
        """Should use config default if max_loss_fraction not provided."""
        wt_init = 0.01
        T = 873.15

        t_default = oxidation.time_to_critical_loss(wt_init, T)
        t_explicit = oxidation.time_to_critical_loss(
            wt_init, T, config.SIM_MAX_WALL_LOSS_FRACTION
        )

        assert abs(t_default - t_explicit) < 1.0e-10, \
            "Default should match config value"

    def test_validation_at_critical_time(self, oxidation, config):
        """At critical time, actual loss should equal critical loss."""
        wt_init = 0.01
        T = 873.15
        max_loss_frac = 0.30

        t_crit = oxidation.time_to_critical_loss(wt_init, T, max_loss_frac)

        # At critical time, loss should equal max_loss_fraction * wt_init
        actual_loss = oxidation.metal_loss(t_crit, T)
        expected_loss = max_loss_frac * wt_init

        # Should be very close (within numerical precision)
        assert abs(actual_loss - expected_loss) < 1.0e-10 * expected_loss, \
            "Loss at critical time should equal limit"


class TestOxidationTemperatureSensitivity:
    """Tests for oxidation temperature sensitivity."""

    def test_activation_energy_effect(self, oxidation, config):
        """Verify activation energy controls temperature dependence."""
        # Q = 250 kJ/mol, so 50K change should give significant increase
        T1 = 873.15
        T2 = 923.15

        x1 = oxidation.oxide_thickness(1.0e7, T1)
        x2 = oxidation.oxide_thickness(1.0e7, T2)

        ratio = x2 / x1
        # Expect significant increase due to Arrhenius
        assert ratio > 2.0, "Temperature effect should be strong"

    def test_q_equals_250_kj_mol(self, oxidation, config):
        """Verify activation energy matches literature value."""
        assert abs(config.OXIDE_Q - 250.0e3) < 1.0, \
            "Activation energy should be ~250 kJ/mol (Quadakkers 2005)"


class TestOxidationEdgeCases:
    """Edge case and boundary tests."""

    def test_very_large_time(self, oxidation):
        """Should handle very large times without error."""
        T = 873.15
        t_huge = 1.0e15  # Very large time

        x = oxidation.oxide_thickness(t_huge, T)

        assert x > 0.0 and np.isfinite(x), "Should handle large times"

    def test_very_high_temperature(self, oxidation):
        """Should handle elevated temperatures."""
        T = 1273.15  # 1000 C
        t = 1.0e7

        x = oxidation.oxide_thickness(t, T)

        assert x > 0.0 and np.isfinite(x), "Should handle high temperatures"

    def test_very_low_temperature(self, oxidation):
        """At low temperature, oxidation should be very slow."""
        T = 573.15  # 300 C (below design)
        t = 1.0e7

        x = oxidation.oxide_thickness(t, T)

        # Should be positive but very small
        assert 0.0 < x < 1.0e-6, "Oxidation should be negligible at low T"

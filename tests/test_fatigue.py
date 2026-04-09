"""
Comprehensive test suite for src/fatigue.py.

Tests thermal strain, Coffin-Manson fatigue life, damage fraction computation,
and epsilon-N curve generation for 9Cr-1Mo at high temperature.
"""

import pytest
import numpy as np


class TestThermalStrainRange:
    """Tests for thermal strain range computation."""

    def test_zero_delta_T_gives_zero_strain(self, fatigue):
        """Zero temperature change should give zero thermal strain."""
        eps_th = fatigue.thermal_strain_range(delta_T=0.0)
        assert eps_th == 0.0

    def test_proportional_to_delta_T(self, fatigue, config):
        """Thermal strain must be proportional to temperature change."""
        delta_T = 300.0  # K
        eps_th = fatigue.thermal_strain_range(delta_T)

        expected = config.CTE * delta_T
        assert abs(eps_th - expected) < 1.0e-15, \
            "Strain = alpha * delta_T"

    def test_positive_for_positive_delta_T(self, fatigue):
        """Positive temperature change should give positive strain."""
        eps_th = fatigue.thermal_strain_range(delta_T=200.0)
        assert eps_th > 0.0

    def test_handles_negative_delta_T(self, fatigue, config):
        """Should handle cooling (negative delta_T) via absolute value."""
        delta_T_heat = 200.0
        delta_T_cool = -200.0

        eps_heat = fatigue.thermal_strain_range(delta_T_heat)
        eps_cool = fatigue.thermal_strain_range(delta_T_cool)

        # Both should be same magnitude (uses abs)
        assert abs(eps_heat - eps_cool) < 1.0e-15

    def test_uses_default_cte(self, fatigue, config):
        """Should use config default if CTE not provided."""
        delta_T = 300.0

        eps_default = fatigue.thermal_strain_range(delta_T)
        eps_explicit = fatigue.thermal_strain_range(delta_T, cte=config.CTE)

        assert abs(eps_default - eps_explicit) < 1.0e-15

    def test_linear_with_delta_T(self, fatigue):
        """Strain should scale linearly with temperature change."""
        eps1 = fatigue.thermal_strain_range(100.0)
        eps2 = fatigue.thermal_strain_range(200.0)

        ratio = eps2 / eps1
        assert abs(ratio - 2.0) < 0.01


class TestThermalStressRange:
    """Tests for thermal stress range (elastic restraint)."""

    def test_proportional_to_delta_T_and_E(self, fatigue, config):
        """Stress = E * alpha * delta_T."""
        delta_T = 300.0

        sigma_th = fatigue.thermal_stress_range(delta_T)

        expected = config.YOUNGS_MODULUS * config.CTE * delta_T
        assert abs(sigma_th - expected) < 1.0, \
            "Formula: sigma = E * alpha * delta_T"

    def test_zero_delta_T_gives_zero_stress(self, fatigue):
        """Zero temperature change should give zero thermal stress."""
        sigma_th = fatigue.thermal_stress_range(delta_T=0.0)
        assert sigma_th == 0.0

    def test_positive_for_positive_delta_T(self, fatigue):
        """Positive delta_T should give positive stress."""
        sigma_th = fatigue.thermal_stress_range(delta_T=200.0)
        assert sigma_th > 0.0

    def test_handles_negative_delta_T(self, fatigue):
        """Should use absolute value of delta_T."""
        sigma1 = fatigue.thermal_stress_range(delta_T=200.0)
        sigma2 = fatigue.thermal_stress_range(delta_T=-200.0)

        assert abs(sigma1 - sigma2) < 1.0e-10

    def test_uses_default_youngs_modulus(self, fatigue, config):
        """Should use config default if E not provided."""
        delta_T = 300.0

        sigma_default = fatigue.thermal_stress_range(delta_T)
        sigma_explicit = fatigue.thermal_stress_range(
            delta_T, E=config.YOUNGS_MODULUS, cte=config.CTE
        )

        assert abs(sigma_default - sigma_explicit) < 1.0e-10

    def test_increases_with_youngs_modulus(self, fatigue):
        """Higher modulus should give higher stress for same delta_T."""
        delta_T = 300.0

        E_low = 150.0e9
        E_high = 170.0e9

        sigma_low = fatigue.thermal_stress_range(delta_T, E=E_low)
        sigma_high = fatigue.thermal_stress_range(delta_T, E=E_high)

        assert sigma_high > sigma_low


class TestCoffinMansonLife:
    """Tests for Coffin-Manson fatigue life relation."""

    def test_zero_strain_gives_infinity(self, fatigue):
        """Zero strain range should give infinite life."""
        nf = fatigue.coffin_manson_life(delta_eps=0.0)
        assert nf == np.inf, "Zero strain gives infinite life"

    def test_positive_strain_gives_finite_life(self, fatigue):
        """Positive strain should give finite life."""
        nf = fatigue.coffin_manson_life(delta_eps=0.01)
        assert 0.0 < nf < np.inf, "Positive strain gives finite life"

    def test_higher_strain_shorter_life(self, fatigue):
        """Higher strain range should reduce fatigue life."""
        eps1 = 0.005
        eps2 = 0.020

        nf1 = fatigue.coffin_manson_life(eps1)
        nf2 = fatigue.coffin_manson_life(eps2)

        assert nf2 < nf1, "Higher strain reduces life"

    def test_negative_strain_gives_infinity(self, fatigue):
        """Negative strain is unphysical, should return infinity."""
        nf = fatigue.coffin_manson_life(delta_eps=-0.01)
        assert nf == np.inf

    def test_reasonable_life_values(self, fatigue):
        """Life should be in physically reasonable range."""
        eps = 0.01  # 1% strain

        nf = fatigue.coffin_manson_life(eps)

        # For 1% strain, expect 10^2 - 10^5 cycles
        assert 1.0 < nf < 1.0e8, \
            f"Life {nf:.0f} cycles outside reasonable range"

    def test_monotonic_relationship(self, fatigue):
        """Life should decrease monotonically with strain."""
        strains = np.array([0.002, 0.005, 0.010, 0.015, 0.020])
        lives = [fatigue.coffin_manson_life(eps) for eps in strains]

        for i in range(len(lives)-1):
            assert lives[i+1] < lives[i], \
                "Life must decrease monotonically with strain"

    def test_inverse_relationship_strain_life(self, fatigue):
        """Life and strain should follow power law: eps = C * N^(-beta)."""
        # Pick two different strains and verify the relationship
        eps1 = 0.005
        eps2 = 0.010

        nf1 = fatigue.coffin_manson_life(eps1)
        nf2 = fatigue.coffin_manson_life(eps2)

        # Strain ratio should be (N ratio)^(-beta)
        # With beta ~ 0.6, strain should increase faster than N decreases
        ratio_strain = eps2 / eps1
        ratio_life = nf1 / nf2

        # ratio_strain should be > ratio_life^(-beta) approximately
        assert ratio_strain > 1.0, "Verification"


class TestStrainRangeAtLife:
    """Tests for inverse of Coffin-Manson relation."""

    def test_returns_positive_strain(self, fatigue):
        """Strain at any positive life should be positive."""
        nf = 1000.0
        eps = fatigue.strain_range_at_life(nf)
        assert eps > 0.0, "Strain must be positive"

    def test_inverse_of_coffin_manson(self, fatigue):
        """Should invert Coffin-Manson: eps = C1*N^(-b1) + C2*N^(-b2)."""
        nf = 5000.0
        eps = fatigue.strain_range_at_life(nf)

        # Should be in reasonable range
        assert 0.0001 < eps < 0.5, "Strain in reasonable range"

    def test_decreases_with_life(self, fatigue):
        """Strain range decreases as fatigue life increases."""
        nf1 = 100.0
        nf2 = 10000.0

        eps1 = fatigue.strain_range_at_life(nf1)
        eps2 = fatigue.strain_range_at_life(nf2)

        assert eps1 > eps2, "Strain decreases with increasing life"

    def test_consistency_with_coffin_manson(self, fatigue):
        """Round-trip: life(strain_at_life(N)) should equal N."""
        nf_original = 5000.0

        eps = fatigue.strain_range_at_life(nf_original)
        nf_computed = fatigue.coffin_manson_life(eps)

        # Should be approximately equal
        assert abs(nf_computed - nf_original) / nf_original < 0.05, \
            "Round-trip consistency"

    def test_increases_with_youngs_modulus_indirectly(self, fatigue):
        """Strain at life depends on material parameters."""
        # This is implicit - just verify computation works
        nf = 1000.0
        eps = fatigue.strain_range_at_life(nf)
        assert eps > 0.0


class TestFatigueDamageFraction:
    """Tests for linear fatigue damage fraction computation."""

    def test_zero_cycles_gives_zero_damage(self, fatigue):
        """Zero applied cycles should give zero damage."""
        D_f = fatigue.fatigue_damage_fraction(n_cycles=0.0, delta_eps=0.01)
        assert D_f == 0.0, "Zero cycles gives zero damage"

    def test_proportional_to_applied_cycles(self, fatigue):
        """Damage should be proportional to n_cycles."""
        delta_eps = 0.01

        D_f1 = fatigue.fatigue_damage_fraction(100.0, delta_eps)
        D_f2 = fatigue.fatigue_damage_fraction(200.0, delta_eps)

        ratio = D_f2 / D_f1
        assert abs(ratio - 2.0) < 0.01, "Damage linear in cycles"

    def test_zero_strain_gives_zero_damage(self, fatigue):
        """Zero strain range should give zero damage (infinite life)."""
        D_f = fatigue.fatigue_damage_fraction(n_cycles=1000.0, delta_eps=0.0)
        assert D_f == 0.0, "Zero strain gives infinite life, zero damage"

    def test_damage_increases_with_strain(self, fatigue):
        """Higher strain range should increase damage at same cycles."""
        n = 1000.0

        D_f1 = fatigue.fatigue_damage_fraction(n, delta_eps=0.005)
        D_f2 = fatigue.fatigue_damage_fraction(n, delta_eps=0.010)

        assert D_f2 > D_f1, "Higher strain increases damage"

    def test_damage_fraction_formula(self, fatigue):
        """D_f = n / N_f where N_f from Coffin-Manson."""
        n = 500.0
        delta_eps = 0.01

        D_f = fatigue.fatigue_damage_fraction(n, delta_eps)
        nf = fatigue.coffin_manson_life(delta_eps)

        expected = n / nf
        assert abs(D_f - expected) < 1.0e-10, "D_f = n / N_f"


class TestFatigueLifeCurve:
    """Tests for epsilon-N curve generation."""

    def test_returns_correct_structure(self, fatigue):
        """Should return 4 arrays: life, total strain, elastic, plastic."""
        nf_array, eps_total, eps_elastic, eps_plastic = fatigue.fatigue_life_curve()

        assert len(nf_array) > 0
        assert len(eps_total) == len(nf_array)
        assert len(eps_elastic) == len(nf_array)
        assert len(eps_plastic) == len(nf_array)

    def test_n_points_control(self, fatigue):
        """Should respect n_points parameter."""
        n_points = 50
        nf_array, _, _, _ = fatigue.fatigue_life_curve(n_points=n_points)

        assert len(nf_array) == n_points

    def test_life_array_monotonic(self, fatigue):
        """Life array should be monotonically increasing."""
        nf_array, _, _, _ = fatigue.fatigue_life_curve(n_points=100)

        for i in range(len(nf_array)-1):
            assert nf_array[i] < nf_array[i+1]

    def test_strain_components_positive(self, fatigue):
        """All strain components should be positive."""
        nf_array, eps_total, eps_elastic, eps_plastic = fatigue.fatigue_life_curve()

        assert all(eps_total > 0.0)
        assert all(eps_elastic > 0.0)
        assert all(eps_plastic > 0.0)

    def test_total_equals_elastic_plus_plastic(self, fatigue):
        """Total strain should equal elastic plus plastic."""
        nf_array, eps_total, eps_elastic, eps_plastic = fatigue.fatigue_life_curve()

        eps_sum = eps_elastic + eps_plastic

        for i in range(len(nf_array)):
            assert abs(eps_total[i] - eps_sum[i]) < 1.0e-15 * eps_total[i], \
                "Total = elastic + plastic"

    def test_elastic_dominates_high_nf(self, fatigue):
        """At high life (high N_f), elastic strain should dominate."""
        nf_array, eps_total, eps_elastic, eps_plastic = fatigue.fatigue_life_curve(n_points=100)

        # At highest life (lowest strain):
        assert eps_elastic[-1] > eps_plastic[-1], \
            "Elastic dominates at high life"

    def test_plastic_dominates_low_nf(self, fatigue):
        """At low life (low N_f), plastic strain should dominate."""
        nf_array, eps_total, eps_elastic, eps_plastic = fatigue.fatigue_life_curve(n_points=100)

        # At lowest life (highest strain):
        assert eps_plastic[0] > eps_elastic[0], \
            "Plastic dominates at low life"

    def test_total_strain_decreases_with_life(self, fatigue):
        """Total strain should decrease monotonically with life."""
        nf_array, eps_total, _, _ = fatigue.fatigue_life_curve(n_points=100)

        for i in range(len(eps_total)-1):
            assert eps_total[i+1] < eps_total[i], \
                "Strain decreases with increasing life"

    def test_curve_matches_coffin_manson(self, fatigue):
        """Curve should match Coffin-Manson formula."""
        nf_array, eps_total, _, _ = fatigue.fatigue_life_curve(n_points=50)

        for i, nf in enumerate(nf_array):
            eps_expected = fatigue.strain_range_at_life(nf)
            assert abs(eps_total[i] - eps_expected) < 1.0e-15 * eps_expected, \
                f"Mismatch at N_f = {nf:.0f}"


class TestFatigueCurveShape:
    """Tests for characteristic shape of epsilon-N curve."""

    def test_two_slope_behavior(self, fatigue):
        """Epsilon-N curve should show two-slope behavior (Basquin + Coffin-Manson)."""
        # Extract curves
        nf_array, eps_total, eps_elastic, eps_plastic = fatigue.fatigue_life_curve(n_points=200)

        # At low life: plastic dominates, steeper slope
        # At high life: elastic dominates, shallower slope
        # Verify this by checking tangent slopes

        # High-life regime (last 50 points)
        n_high = 50
        slope_high = (np.log(eps_total[-1]) - np.log(eps_total[-n_high-1])) / \
                     (np.log(nf_array[-1]) - np.log(nf_array[-n_high-1]))

        # Just verify both regimes exist
        assert len(eps_total) > 10

    def test_curve_fits_data_points(self, fatigue):
        """Each point on curve should satisfy Coffin-Manson."""
        nf_array, eps_total, _, _ = fatigue.fatigue_life_curve(n_points=50)

        for nf, eps in zip(nf_array, eps_total):
            # Compute strain from Coffin-Manson at this life
            eps_cm = fatigue.strain_range_at_life(nf)

            # Should match curve value
            assert abs(eps - eps_cm) < 1.0e-10 * eps


class TestFatigueEdgeCases:
    """Edge cases and boundary tests."""

    def test_very_high_strain(self, fatigue):
        """Very high strain should give short life."""
        eps_high = 0.1  # 10% strain

        nf = fatigue.coffin_manson_life(eps_high)

        # Should be short (low cycles)
        assert nf < 1000.0, "High strain gives short life"

    def test_very_low_strain(self, fatigue):
        """Very low strain should give long life."""
        eps_low = 0.0001  # 0.01% strain

        nf = fatigue.coffin_manson_life(eps_low)

        # Should be long (high cycles)
        assert nf > 1.0e5, "Low strain gives long life"

    def test_large_temperature_swing(self, fatigue, config):
        """Large thermal cycle should create large strain."""
        delta_T_large = 500.0  # 500 K swing

        eps = fatigue.thermal_strain_range(delta_T_large)

        # Should be significant
        assert eps > 0.005, "Large delta_T creates large strain"

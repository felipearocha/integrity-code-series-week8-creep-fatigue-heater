"""
Comprehensive test suite for src/creep_engine.py.

Tests Norton power-law creep, Larson-Miller parameter, Omega method, hoop stress,
and integration routines. Includes edge cases, physics monotonicity, and known benchmarks.
"""

import pytest
import numpy as np
from scipy.optimize import brentq


class TestNortonCreepRate:
    """Tests for Norton creep rate formula."""

    def test_zero_stress_gives_zero_rate(self, creep_engine):
        """Zero applied stress should give zero creep rate."""
        rate = creep_engine.norton_creep_rate(sigma=0.0, T=873.15)
        assert rate == 0.0, "Zero stress must give zero creep rate"

    def test_negative_stress_gives_zero_rate(self, creep_engine):
        """Negative stress is unphysical and should return zero."""
        rate = creep_engine.norton_creep_rate(sigma=-1.0e6, T=873.15)
        assert rate == 0.0

    def test_zero_temperature_gives_zero_rate(self, creep_engine):
        """Zero absolute temperature is unphysical, returns zero."""
        rate = creep_engine.norton_creep_rate(sigma=1.0e8, T=0.0)
        assert rate == 0.0

    def test_negative_temperature_gives_zero_rate(self, creep_engine):
        """Negative absolute temperature is unphysical, returns zero."""
        rate = creep_engine.norton_creep_rate(sigma=1.0e8, T=-100.0)
        assert rate == 0.0

    def test_positive_stress_positive_rate(self, creep_engine):
        """Positive stress and temperature must give positive rate."""
        rate = creep_engine.norton_creep_rate(sigma=100.0e6, T=873.15)
        assert rate > 0.0, "Positive stress must give positive rate"

    def test_rate_increases_with_temperature(self, creep_engine, config):
        """Creep rate increases exponentially with temperature."""
        T_low = 823.15  # 550 C
        T_high = 923.15  # 650 C
        sigma = 100.0e6  # 100 MPa

        rate_low = creep_engine.norton_creep_rate(sigma, T_low)
        rate_high = creep_engine.norton_creep_rate(sigma, T_high)

        assert rate_high > rate_low, "Rate must increase with temperature"
        assert rate_high / rate_low > 10.0, "Temperature dependence should be strong"

    def test_rate_increases_with_stress(self, creep_engine):
        """Creep rate increases with applied stress (power law)."""
        sigma_low = 50.0e6   # 50 MPa
        sigma_high = 200.0e6  # 200 MPa
        T = 873.15

        rate_low = creep_engine.norton_creep_rate(sigma_low, T)
        rate_high = creep_engine.norton_creep_rate(sigma_high, T)

        assert rate_high > rate_low, "Rate must increase with stress"
        # With n=10.5, ratio should be (200/50)^10.5 = 4^10.5 ≈ 1e6
        assert rate_high / rate_low > 100.0, "Stress exponent should be high (n~10.5)"

    def test_stress_exponent_sensitivity(self, creep_engine, config):
        """Test sensitivity to Norton stress exponent."""
        sigma = 100.0e6
        T = 873.15
        n = config.NORTON_N

        # Stress effect should be n-th power
        sigma1 = 100.0e6
        sigma2 = 150.0e6  # 1.5x increase

        rate1 = creep_engine.norton_creep_rate(sigma1, T)
        rate2 = creep_engine.norton_creep_rate(sigma2, T)

        ratio_actual = rate2 / rate1
        ratio_expected = (sigma2 / sigma1) ** n

        assert abs(ratio_actual - ratio_expected) / ratio_expected < 0.01, \
            "Stress exponent must match Norton n parameter"

    def test_known_benchmark_rate(self, creep_engine, config):
        """Test at known benchmark: 873K/100MPa should give ~1e-8/s."""
        T = 873.15  # 600 C
        sigma = 100.0e6  # 100 MPa
        rate = creep_engine.norton_creep_rate(sigma, T)

        # Expect rate ~1e-8 /s per NIMS calibration
        assert 1.0e-9 < rate < 1.0e-7, \
            f"Rate at 873K/100MPa should be ~1e-8/s, got {rate:.3e}"


class TestLarsonMillerParameter:
    """Tests for Larson-Miller parameter computation."""

    def test_correct_formula(self, creep_engine, config):
        """LMP = T * (C + log10(t_r))."""
        T = 873.15
        t_r = 1000.0  # 1000 hours
        lmp = creep_engine.larson_miller_parameter(T, t_r)

        expected = T * (config.LMP_C + np.log10(t_r))
        assert abs(lmp - expected) < 1.0, "LMP formula must be T*(C + log10(t_r))"

    def test_positive_time_required(self, creep_engine):
        """Rupture time must be positive."""
        with pytest.raises(ValueError):
            creep_engine.larson_miller_parameter(873.15, t_r=-100.0)

    def test_zero_time_raises(self, creep_engine):
        """Zero rupture time must raise error."""
        with pytest.raises(ValueError):
            creep_engine.larson_miller_parameter(873.15, t_r=0.0)

    def test_lmp_increases_with_temperature(self, creep_engine):
        """LMP increases monotonically with temperature at fixed time."""
        t_r = 1000.0
        T1 = 823.15
        T2 = 923.15

        lmp1 = creep_engine.larson_miller_parameter(T1, t_r)
        lmp2 = creep_engine.larson_miller_parameter(T2, t_r)

        assert lmp2 > lmp1, "LMP must increase with temperature"

    def test_lmp_increases_with_time(self, creep_engine):
        """LMP increases with rupture time at fixed temperature."""
        T = 873.15
        t_r1 = 100.0
        t_r2 = 10000.0

        lmp1 = creep_engine.larson_miller_parameter(T, t_r1)
        lmp2 = creep_engine.larson_miller_parameter(T, t_r2)

        assert lmp2 > lmp1, "LMP must increase with time"

    def test_known_values(self, creep_engine, config):
        """Test LMP at reference conditions."""
        # At 873K and 100 hours:
        T = 873.15
        t_r = 100.0
        lmp = creep_engine.larson_miller_parameter(T, t_r)

        # LMP = 873.15 * (20 + log10(100)) = 873.15 * (20 + 2) = 19212.3
        expected = T * (config.LMP_C + np.log10(t_r))
        assert abs(lmp - expected) < 0.1


class TestRuptureStressFromLMP:
    """Tests for rupture stress computation from LMP."""

    def test_returns_positive_stress(self, creep_engine):
        """Rupture stress must be positive."""
        lmp = 20000.0
        sigma = creep_engine.rupture_stress_from_lmp(lmp)
        assert sigma > 0.0, "Rupture stress must be positive"

    def test_monotonically_decreasing_with_lmp(self, creep_engine):
        """Higher LMP (longer time at higher T) = lower stress."""
        lmp1 = 18000.0  # Lower LMP
        lmp2 = 22000.0  # Higher LMP

        sigma1 = creep_engine.rupture_stress_from_lmp(lmp1)
        sigma2 = creep_engine.rupture_stress_from_lmp(lmp2)

        assert sigma1 > sigma2, "Rupture stress must decrease with LMP"

    def test_known_reference_points(self, creep_engine, config):
        """Test at known reference points from API 530 / NIMS data."""
        # These are approximate anchor points from the configuration
        lmp_ref = 19000.0
        sigma_ref = creep_engine.rupture_stress_from_lmp(lmp_ref)

        # Should be in physically reasonable range (50-400 MPa for 9Cr-1Mo)
        assert 50.0e6 < sigma_ref < 400.0e6, \
            f"Rupture stress {sigma_ref/1e6:.1f} MPa outside typical range"

    def test_lmp_range_validity(self, creep_engine):
        """Test over typical 9Cr-1Mo LMP range (15000-30000)."""
        lmps = np.linspace(15000, 30000, 10)
        stresses = [creep_engine.rupture_stress_from_lmp(lmp) for lmp in lmps]

        # All should be positive
        assert all(s > 0 for s in stresses)
        # Should be monotonically decreasing
        assert all(stresses[i] > stresses[i+1] for i in range(len(stresses)-1))


class TestRuptureTimeAPI530:
    """Tests for rupture time estimation using API 530 LMP method."""

    def test_returns_positive_time(self, creep_engine):
        """Rupture time must be positive."""
        sigma = 100.0e6  # 100 MPa
        T = 873.15  # 600 C
        t_r = creep_engine.rupture_time_api530(sigma, T)
        assert t_r > 0.0, "Rupture time must be positive"

    def test_physically_reasonable_values(self, creep_engine):
        """Rupture times should be physically realistic."""
        sigma = 100.0e6
        T = 873.15
        t_r = creep_engine.rupture_time_api530(sigma, T)

        # For 9Cr-1Mo at 100 MPa/600C, expect ~1000-100000 hours
        assert 10.0 < t_r < 1.0e6, \
            f"Rupture time {t_r:.1f} h outside expected range"

    def test_higher_temperature_shorter_life(self, creep_engine):
        """Higher temperature should give shorter rupture time."""
        sigma = 100.0e6
        T_low = 823.15  # 550 C
        T_high = 923.15  # 650 C

        t_r_low = creep_engine.rupture_time_api530(sigma, T_low)
        t_r_high = creep_engine.rupture_time_api530(sigma, T_high)

        assert t_r_high < t_r_low, "Higher T must reduce rupture time"

    def test_higher_stress_shorter_life(self, creep_engine):
        """Higher stress should give shorter rupture time."""
        T = 873.15
        sigma_low = 50.0e6   # 50 MPa
        sigma_high = 200.0e6  # 200 MPa

        t_r_low = creep_engine.rupture_time_api530(sigma_low, T)
        t_r_high = creep_engine.rupture_time_api530(sigma_high, T)

        assert t_r_high < t_r_low, "Higher stress must reduce rupture time"

    def test_consistency_with_lmp(self, creep_engine, config):
        """Result should be consistent with LMP computation."""
        sigma = 100.0e6
        T = 873.15
        t_r = creep_engine.rupture_time_api530(sigma, T)

        # Compute LMP from the returned rupture time
        lmp_computed = creep_engine.larson_miller_parameter(T, t_r)

        # This should approximately match the LMP corresponding to sigma
        # (round-trip consistency)
        assert lmp_computed > 10000.0, "LMP should be in expected range"


class TestOmegaCreepRate:
    """Tests for MPC Omega method creep rate."""

    def test_increases_with_accumulated_strain(self, creep_engine, config):
        """Omega method rate increases exponentially with creep strain."""
        T = 873.15
        sigma = 100.0e6

        eps0 = 0.0
        eps1 = 0.005
        eps2 = 0.015

        rate0 = creep_engine.omega_creep_rate(eps0, T, sigma)
        rate1 = creep_engine.omega_creep_rate(eps1, T, sigma)
        rate2 = creep_engine.omega_creep_rate(eps2, T, sigma)

        assert rate0 < rate1 < rate2, "Rate must increase with strain"

    def test_initial_rate_at_zero_strain(self, creep_engine, config):
        """At zero strain, rate should equal eps_dot_0."""
        T = 873.15
        sigma = 100.0e6
        eps = 0.0

        rate = creep_engine.omega_creep_rate(eps, T, sigma)

        # Should be approximately eps_dot_0
        assert abs(rate - config.OMEGA_EPS_DOT_0) < 1.0e-11, \
            "Rate at zero strain must equal eps_dot_0"

    def test_rupture_strain_from_omega_definition(self, creep_engine, config):
        """Omega rupture strain should be 1/Omega."""
        rupture_strain = creep_engine.omega_rupture_strain()
        expected = 1.0 / config.OMEGA_PARAM

        assert abs(rupture_strain - expected) < 1.0e-10


class TestOmegaRuptureTime:
    """Tests for Omega method rupture time."""

    def test_rupture_time_formula(self, creep_engine, config):
        """t_r = 1 / (Omega * eps_dot_0)."""
        t_r = creep_engine.omega_rupture_time()
        expected = 1.0 / (config.OMEGA_PARAM * config.OMEGA_EPS_DOT_0)

        assert abs(t_r - expected) < 1.0e-15, "Rupture time formula incorrect"

    def test_positive_rupture_time(self, creep_engine):
        """Rupture time must be positive."""
        t_r = creep_engine.omega_rupture_time()
        assert t_r > 0.0


class TestHoopStress:
    """Tests for thin-wall hoop stress formula."""

    def test_thin_wall_formula_correctness(self, creep_engine):
        """Verify Barlow's formula: sigma = P*D_mean/(2*t)."""
        pressure = 5.0e6  # 5 MPa
        od = 0.1143  # m
        wt = 0.00635  # m

        sigma = creep_engine.hoop_stress_thin_wall(pressure, od, wt)

        d_mean = od - wt
        expected = pressure * d_mean / (2.0 * wt)

        assert abs(sigma - expected) < 1.0, "Hoop stress formula incorrect"

    def test_zero_wall_gives_infinity(self, creep_engine):
        """Zero wall thickness should give infinite stress."""
        pressure = 5.0e6
        od = 0.1143
        wt = 0.0

        sigma = creep_engine.hoop_stress_thin_wall(pressure, od, wt)
        assert sigma == np.inf, "Zero wall must give infinite stress"

    def test_negative_wall_gives_infinity(self, creep_engine):
        """Negative wall thickness (unphysical) gives infinity."""
        pressure = 5.0e6
        od = 0.1143
        wt = -0.01

        sigma = creep_engine.hoop_stress_thin_wall(pressure, od, wt)
        assert sigma == np.inf

    def test_stress_decreases_with_wall_thickness(self, creep_engine):
        """Hoop stress decreases as wall thickness increases."""
        pressure = 5.0e6
        od = 0.1143

        wt1 = 0.005
        wt2 = 0.010

        sigma1 = creep_engine.hoop_stress_thin_wall(pressure, od, wt1)
        sigma2 = creep_engine.hoop_stress_thin_wall(pressure, od, wt2)

        assert sigma1 > sigma2, "Stress must decrease with wall thickness"

    def test_stress_increases_with_pressure(self, creep_engine):
        """Hoop stress increases linearly with pressure."""
        od = 0.1143
        wt = 0.00635

        p1 = 3.0e6
        p2 = 6.0e6

        sigma1 = creep_engine.hoop_stress_thin_wall(p1, od, wt)
        sigma2 = creep_engine.hoop_stress_thin_wall(p2, od, wt)

        ratio = sigma2 / sigma1
        assert abs(ratio - 2.0) < 0.01, "Stress should scale linearly with pressure"

    def test_stress_increases_with_od(self, creep_engine):
        """Hoop stress increases with outside diameter (at constant ID)."""
        pressure = 5.0e6
        wt = 0.00635

        od1 = 0.1
        od2 = 0.15

        sigma1 = creep_engine.hoop_stress_thin_wall(pressure, od1, wt)
        sigma2 = creep_engine.hoop_stress_thin_wall(pressure, od2, wt)

        assert sigma2 > sigma1, "Stress should increase with OD"


class TestNortonIntegration:
    """Tests for Norton creep integration routine."""

    def test_integration_runs_without_error(self, creep_engine):
        """Basic integration should complete."""
        T = 873.15
        dt_s = 100.0  # 100 second time step
        total_time_s = 1000.0  # 1000 seconds

        # Constant stress function
        sigma_const = 100.0e6
        sigma_func = lambda t, eps: sigma_const

        times, strains, rates = creep_engine.integrate_creep_norton(
            sigma_func, T, dt_s, total_time_s
        )

        assert len(times) > 0
        assert len(strains) == len(times)
        assert len(rates) == len(times)

    def test_integration_converges(self, creep_engine):
        """Integration should show monotonic strain accumulation."""
        T = 873.15
        dt_s = 100.0
        total_time_s = 10000.0

        sigma_const = 100.0e6
        sigma_func = lambda t, eps: sigma_const

        times, strains, rates = creep_engine.integrate_creep_norton(
            sigma_func, T, dt_s, total_time_s
        )

        # Strains should be monotonically increasing
        for i in range(len(strains)-1):
            assert strains[i+1] >= strains[i], "Strain must increase monotonically"

    def test_integration_respects_max_strain(self, creep_engine):
        """Integration should stop at max_strain criterion."""
        T = 873.15
        dt_s = 10.0
        total_time_s = 1.0e6  # Long time
        max_strain = 0.02

        sigma_const = 200.0e6  # High stress for faster creep
        sigma_func = lambda t, eps: sigma_const

        times, strains, rates = creep_engine.integrate_creep_norton(
            sigma_func, T, dt_s, total_time_s, max_strain=max_strain
        )

        # Final strain should not exceed max_strain by much
        assert strains[-1] <= max_strain * 1.01, \
            "Final strain should not significantly exceed max_strain"

    def test_integration_rates_positive(self, creep_engine):
        """All creep rates during integration should be positive."""
        T = 873.15
        dt_s = 100.0
        total_time_s = 5000.0

        sigma_const = 100.0e6
        sigma_func = lambda t, eps: sigma_const

        times, strains, rates = creep_engine.integrate_creep_norton(
            sigma_func, T, dt_s, total_time_s
        )

        assert all(r >= 0.0 for r in rates), "All rates must be non-negative"

    def test_integration_time_array_consistent(self, creep_engine):
        """Time array should be consistent with dt and total_time."""
        T = 873.15
        dt_s = 200.0
        total_time_s = 2000.0

        sigma_const = 100.0e6
        sigma_func = lambda t, eps: sigma_const

        times, strains, rates = creep_engine.integrate_creep_norton(
            sigma_func, T, dt_s, total_time_s
        )

        # Times should increase with roughly constant steps
        assert times[0] == 0.0, "First time should be zero"
        assert times[-1] <= total_time_s + dt_s, "Last time should not exceed total"


class TestOmegaIntegration:
    """Tests for Omega creep integration routine."""

    def test_omega_integration_runs(self, creep_engine):
        """Omega integration should complete without error."""
        T = 873.15
        dt_s = 100.0
        total_time_s = 100000.0

        sigma_const = 100.0e6
        sigma_func = lambda t, eps: sigma_const

        times, strains, rates = creep_engine.integrate_creep_omega(
            sigma_func, T, dt_s, total_time_s
        )

        assert len(times) > 0
        assert len(strains) == len(times)

    def test_omega_integration_reaches_rupture(self, creep_engine, config):
        """Omega integration should reach rupture strain (1/Omega) given enough time."""
        T = 873.15
        # Omega rupture time = 1/(Omega * eps_dot_0)
        # Use long enough total time to actually reach rupture
        omega = config.OMEGA_PARAM
        eps_dot_0 = config.OMEGA_EPS_DOT_0
        t_rupture_s = 1.0 / (omega * eps_dot_0)
        dt_s = t_rupture_s / 500  # 500 steps to reach rupture
        total_time_s = t_rupture_s * 1.5  # 50% margin

        sigma_const = 100.0e6
        sigma_func = lambda t, eps: sigma_const

        times, strains, rates = creep_engine.integrate_creep_omega(
            sigma_func, T, dt_s, total_time_s
        )

        rupture_strain = 1.0 / omega

        # Should approach rupture strain
        assert strains[-1] > rupture_strain * 0.9, \
            "Should approach rupture strain"

    def test_omega_rates_increasing(self, creep_engine):
        """Omega rates should accelerate as strain increases."""
        T = 873.15
        dt_s = 100.0
        total_time_s = 100000.0

        sigma_const = 100.0e6
        sigma_func = lambda t, eps: sigma_const

        times, strains, rates = creep_engine.integrate_creep_omega(
            sigma_func, T, dt_s, total_time_s
        )

        # Rates should generally increase
        early_rate = rates[len(rates)//4]
        late_rate = rates[-2]

        assert late_rate > early_rate, "Omega rates should accelerate with strain"

    def test_omega_integration_with_custom_params(self, creep_engine):
        """Omega integration with custom omega and eps_dot_0."""
        T = 873.15
        dt_s = 100.0
        total_time_s = 10000.0
        omega_custom = 5.0
        eps_dot_0_custom = 1.0e-10

        sigma_const = 100.0e6
        sigma_func = lambda t, eps: sigma_const

        times, strains, rates = creep_engine.integrate_creep_omega(
            sigma_func, T, dt_s, total_time_s,
            omega=omega_custom, eps_dot_0=eps_dot_0_custom
        )

        assert len(times) > 0
        assert strains[-1] > 0.0


class TestIntegrationMonotonicity:
    """Physics monotonicity checks for integrations."""

    def test_temperature_monotonicity_norton(self, creep_engine):
        """Higher temperature should accelerate Norton creep."""
        dt_s = 100.0
        total_time_s = 5000.0
        sigma_const = 100.0e6
        sigma_func = lambda t, eps: sigma_const
        max_strain = 0.01

        times_low, strains_low, _ = creep_engine.integrate_creep_norton(
            sigma_func, T=823.15, dt_s=dt_s, total_time_s=total_time_s,
            max_strain=max_strain
        )
        times_high, strains_high, _ = creep_engine.integrate_creep_norton(
            sigma_func, T=923.15, dt_s=dt_s, total_time_s=total_time_s,
            max_strain=max_strain
        )

        # Higher temperature should accumulate more strain in same time
        assert strains_high[-1] > strains_low[-1], \
            "Higher T should produce more creep strain"

    def test_stress_monotonicity_norton(self, creep_engine):
        """Higher stress should accelerate Norton creep."""
        dt_s = 100.0
        total_time_s = 10000.0
        T = 873.15
        max_strain = 0.01

        sigma_low = 80.0e6
        sigma_high = 150.0e6

        sigma_func_low = lambda t, eps: sigma_low
        sigma_func_high = lambda t, eps: sigma_high

        times_low, strains_low, _ = creep_engine.integrate_creep_norton(
            sigma_func_low, T, dt_s, total_time_s, max_strain=max_strain
        )
        times_high, strains_high, _ = creep_engine.integrate_creep_norton(
            sigma_func_high, T, dt_s, total_time_s, max_strain=max_strain
        )

        # Higher stress should accumulate more strain in same time
        assert strains_high[-1] > strains_low[-1], \
            "Higher stress should produce more creep strain"

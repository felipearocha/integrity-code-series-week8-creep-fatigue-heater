"""
Comprehensive test suite for src/creep_fatigue.py.

Tests creep-fatigue interaction envelope (ASME III-5), damage accumulation,
and interaction assessment for 9Cr-1Mo (Gr91) steel.
"""

import numpy as np


class TestInteractionEnvelope:
    """Tests for bilinear creep-fatigue interaction envelope."""

    def test_envelope_generation(self, creep_fatigue, config):
        """Should generate envelope points for both segments."""
        df_env, dc_env = creep_fatigue.interaction_envelope(n_points=200)

        # Concatenation of two segments removes one duplicate point
        assert len(df_env) == 199
        assert len(dc_env) == 199

    def test_envelope_passes_through_critical_points(self, creep_fatigue, config):
        """Envelope should pass through (1,0), intersection, and (0,1)."""
        df_env, dc_env = creep_fatigue.interaction_envelope(n_points=500)

        # Should start near (1, 0)
        assert abs(df_env[0] - 1.0) < 0.01
        assert abs(dc_env[0] - 0.0) < 0.01

        # Should end near (0, 1)
        assert abs(df_env[-1] - 0.0) < 0.01
        assert abs(dc_env[-1] - 1.0) < 0.01

    def test_envelope_passes_through_intersection(self, creep_fatigue, config):
        """Envelope should pass through intersection point."""
        df_env, dc_env = creep_fatigue.interaction_envelope(n_points=500)
        df_int, dc_int = config.CF_INTERSECTION

        # Find point closest to intersection
        distances = np.sqrt((df_env - df_int)**2 + (dc_env - dc_int)**2)
        min_idx = np.argmin(distances)

        # Should be very close
        assert distances[min_idx] < 0.05, \
            "Envelope should pass through intersection (0.1, 0.01)"

    def test_envelope_monotonic_segments(self, creep_fatigue):
        """Each segment should be monotonic."""
        df_env, dc_env = creep_fatigue.interaction_envelope(n_points=200)

        # First segment: D_f decreases
        for i in range(len(df_env)//2 - 1):
            assert df_env[i+1] <= df_env[i], "Segment 1: D_f should decrease"

        # Second segment: D_f continues to decrease
        for i in range(len(df_env)//2, len(df_env)-1):
            assert df_env[i+1] <= df_env[i], "Segment 2: D_f should decrease"

    def test_envelope_is_bilinear(self, creep_fatigue, config):
        """Envelope should consist of two linear segments."""
        df_env, dc_env = creep_fatigue.interaction_envelope(n_points=500)
        df_int, dc_int = config.CF_INTERSECTION

        # Find intersection index
        distances = np.sqrt((df_env - df_int)**2 + (dc_env - dc_int)**2)
        int_idx = np.argmin(distances)

        # Segment 1: from (1, 0) to (df_int, dc_int)
        # Should have constant slope
        slopes_1 = []
        for i in range(int_idx - 5):
            if df_env[i+1] != df_env[i]:
                slope = (dc_env[i+1] - dc_env[i]) / (df_env[i+1] - df_env[i])
                slopes_1.append(slope)

        # Segment 2: from (df_int, dc_int) to (0, 1)
        slopes_2 = []
        for i in range(int_idx, len(df_env)-6):
            if df_env[i+1] != df_env[i]:
                slope = (dc_env[i+1] - dc_env[i]) / (df_env[i+1] - df_env[i])
                slopes_2.append(slope)

        # Slopes within each segment should be approximately constant
        if slopes_1:
            assert np.std(slopes_1) / abs(np.mean(slopes_1)) < 0.1
        if slopes_2:
            assert np.std(slopes_2) / abs(np.mean(slopes_2)) < 0.1


class TestIsWithinEnvelope:
    """Tests for point-in-envelope checking."""

    def test_origin_is_within(self, creep_fatigue):
        """Origin (0,0) should be within envelope."""
        within = creep_fatigue.is_within_envelope(D_f=0.0, D_c=0.0)
        assert within is True, "Origin must be within envelope"

    def test_corner_point_1_0_is_within(self, creep_fatigue):
        """Corner (1, 0) should be within envelope."""
        within = creep_fatigue.is_within_envelope(D_f=1.0, D_c=0.0)
        assert within is True

    def test_corner_point_0_1_is_within(self, creep_fatigue):
        """Corner (0, 1) should be within envelope."""
        within = creep_fatigue.is_within_envelope(D_f=0.0, D_c=1.0)
        assert within is True

    def test_intersection_point_is_within(self, creep_fatigue, config):
        """Intersection point should be within envelope."""
        df_int, dc_int = config.CF_INTERSECTION
        within = creep_fatigue.is_within_envelope(df_int, dc_int)
        assert within is True

    def test_point_outside_is_outside(self, creep_fatigue):
        """Point (1, 1) should be outside envelope."""
        within = creep_fatigue.is_within_envelope(D_f=1.0, D_c=1.0)
        assert within is False, "(1,1) should be outside"

    def test_point_well_outside_is_outside(self, creep_fatigue):
        """Far outside point should be outside."""
        within = creep_fatigue.is_within_envelope(D_f=0.5, D_c=0.5)
        assert within is False

    def test_negative_coordinates_within(self, creep_fatigue):
        """Negative D_f or D_c means no damage, should be within."""
        within1 = creep_fatigue.is_within_envelope(D_f=-0.1, D_c=0.5)
        within2 = creep_fatigue.is_within_envelope(D_f=0.5, D_c=-0.1)

        assert within1 is True
        assert within2 is True

    def test_segment_1_boundary(self, creep_fatigue, config):
        """Points on segment 1 boundary should be within (or very close)."""
        df_int, dc_int = config.CF_INTERSECTION

        # Point on line from (1, 0) to (df_int, dc_int)
        t = 0.5
        df_test = 1.0 - t * (1.0 - df_int)
        dc_test = 0.0 + t * dc_int

        within = creep_fatigue.is_within_envelope(df_test, dc_test)
        # Allow for numerical precision issues at boundary
        margin = creep_fatigue.envelope_margin(df_test, dc_test)
        assert within is True or abs(margin) < 1e-10, \
            "Boundary point should be within (or at boundary)"

    def test_segment_2_boundary(self, creep_fatigue, config):
        """Points on segment 2 boundary should be within."""
        df_int, dc_int = config.CF_INTERSECTION

        # Point on line from (df_int, dc_int) to (0, 1)
        t = 0.5
        df_test = df_int - t * df_int
        dc_test = dc_int + t * (1.0 - dc_int)

        within = creep_fatigue.is_within_envelope(df_test, dc_test)
        assert within is True

    def test_above_segment_1_is_outside(self, creep_fatigue, config):
        """Points above segment 1 should be outside."""
        # Just above the line from (1, 0) to (0.1, 0.01)
        df_test = 0.5
        dc_test = 0.05  # Should be above line

        within = creep_fatigue.is_within_envelope(df_test, dc_test)
        assert within is False, "Above segment 1 should be outside"

    def test_above_segment_2_is_outside(self, creep_fatigue):
        """Points above segment 2 should be outside."""
        # Above the line from (0.1, 0.01) to (0, 1)
        df_test = 0.05
        dc_test = 0.55

        within = creep_fatigue.is_within_envelope(df_test, dc_test)
        assert within is False


class TestEnvelopeMargin:
    """Tests for distance to envelope boundary."""

    def test_origin_has_positive_margin(self, creep_fatigue):
        """Origin should have positive margin (safe)."""
        margin = creep_fatigue.envelope_margin(D_f=0.0, D_c=0.0)
        assert margin > 0.0, "Origin should be safely inside"

    def test_boundary_has_zero_margin(self, creep_fatigue, config):
        """Point on boundary should have ~zero margin."""
        df_int, dc_int = config.CF_INTERSECTION

        margin = creep_fatigue.envelope_margin(df_int, dc_int)
        assert abs(margin) < 0.01, "Intersection should be on boundary"

    def test_outside_has_negative_margin(self, creep_fatigue):
        """Points outside should have negative margin."""
        margin = creep_fatigue.envelope_margin(D_f=1.0, D_c=1.0)
        assert margin < 0.0, "Outside point should have negative margin"

    def test_margin_increases_away_from_boundary(self, creep_fatigue):
        """Moving toward origin should increase margin."""
        margin1 = creep_fatigue.envelope_margin(D_f=0.1, D_c=0.1)
        margin2 = creep_fatigue.envelope_margin(D_f=0.05, D_c=0.05)

        assert margin2 > margin1, "Margin increases toward origin"

    def test_margin_formula_consistency(self, creep_fatigue, config):
        """Margin should equal D_c_limit - D_c."""
        D_f = 0.2
        D_c = 0.05

        margin = creep_fatigue.envelope_margin(D_f, D_c)

        # Compute expected
        df_int, dc_int = config.CF_INTERSECTION
        if D_f >= df_int:
            dc_limit = dc_int * (1.0 - D_f) / (1.0 - df_int)
        else:
            dc_limit = dc_int + (1.0 - dc_int) * (df_int - D_f) / df_int

        expected = dc_limit - D_c
        assert abs(margin - expected) < 1.0e-10, "Margin formula incorrect"


class TestCreepDamageFraction:
    """Tests for creep damage fraction computation."""

    def test_returns_positive_value(self, creep_fatigue):
        """Creep damage should be positive."""
        D_c = creep_fatigue.creep_damage_fraction(
            hold_time_s=1000.0, T=873.15, sigma=100.0e6
        )
        assert D_c >= 0.0

    def test_zero_hold_time_gives_zero_damage(self, creep_fatigue):
        """Zero hold time should give zero creep damage."""
        D_c = creep_fatigue.creep_damage_fraction(
            hold_time_s=0.0, T=873.15, sigma=100.0e6
        )
        assert D_c == 0.0

    def test_proportional_to_hold_time(self, creep_fatigue):
        """Damage should be linear in hold time."""
        D_c1 = creep_fatigue.creep_damage_fraction(
            hold_time_s=1000.0, T=873.15, sigma=100.0e6
        )
        D_c2 = creep_fatigue.creep_damage_fraction(
            hold_time_s=2000.0, T=873.15, sigma=100.0e6
        )

        ratio = D_c2 / D_c1
        assert abs(ratio - 2.0) < 0.01, "Damage linear in hold time"

    def test_increases_with_stress(self, creep_fatigue):
        """Higher stress should increase damage (shorter rupture life)."""
        D_c1 = creep_fatigue.creep_damage_fraction(
            hold_time_s=10000.0, T=873.15, sigma=80.0e6
        )
        D_c2 = creep_fatigue.creep_damage_fraction(
            hold_time_s=10000.0, T=873.15, sigma=150.0e6
        )

        assert D_c2 > D_c1, "Higher stress increases damage"

    def test_increases_with_temperature(self, creep_fatigue):
        """Higher temperature should increase damage."""
        D_c1 = creep_fatigue.creep_damage_fraction(
            hold_time_s=10000.0, T=823.15, sigma=100.0e6
        )
        D_c2 = creep_fatigue.creep_damage_fraction(
            hold_time_s=10000.0, T=923.15, sigma=100.0e6
        )

        assert D_c2 > D_c1, "Higher temperature increases damage"


class TestAccumulatedDamage:
    """Tests for accumulated creep and fatigue damage."""

    def test_no_cycles_no_fatigue_damage(self, creep_fatigue):
        """Zero applied cycles should give zero fatigue damage."""
        D_f, D_c, within = creep_fatigue.accumulated_damage(
            n_cycles_list=[0.0],
            delta_eps_list=[0.01],
            hold_times_s_list=[0.0],
            T_list=[873.15],
            sigma_list=[100.0e6],
        )

        assert D_f == 0.0

    def test_no_hold_time_no_creep_damage(self, creep_fatigue):
        """Zero hold time should give zero creep damage."""
        D_f, D_c, within = creep_fatigue.accumulated_damage(
            n_cycles_list=[100.0],
            delta_eps_list=[0.01],
            hold_times_s_list=[0.0],
            T_list=[873.15],
            sigma_list=[100.0e6],
        )

        assert D_c == 0.0

    def test_returns_three_values(self, creep_fatigue):
        """Should return D_f, D_c, and within flag."""
        result = creep_fatigue.accumulated_damage(
            n_cycles_list=[100.0],
            delta_eps_list=[0.01],
            hold_times_s_list=[1000.0],
            T_list=[873.15],
            sigma_list=[100.0e6],
        )

        assert len(result) == 3
        D_f, D_c, within = result
        assert isinstance(D_f, (float, np.floating))
        assert isinstance(D_c, (float, np.floating))
        assert isinstance(within, (bool, np.bool_))

    def test_multiple_cycle_types(self, creep_fatigue):
        """Should sum damage from multiple cycle types."""
        # Two fatigue events with same strain
        n_cycles_list = [100.0, 100.0]
        delta_eps_list = [0.01, 0.01]
        hold_times_s_list = [0.0, 0.0]
        T_list = [873.15, 873.15]
        sigma_list = [100.0e6, 100.0e6]

        D_f, D_c, within = creep_fatigue.accumulated_damage(
            n_cycles_list, delta_eps_list, hold_times_s_list, T_list, sigma_list
        )

        # Should be approximately double single event damage
        D_f_single, _, _ = creep_fatigue.accumulated_damage(
            [100.0], [0.01], [0.0], [873.15], [100.0e6]
        )

        assert abs(D_f - 2.0*D_f_single) < 1.0e-10 * D_f, "Damages should sum"

    def test_mixed_fatigue_creep(self, creep_fatigue):
        """Should correctly sum mixed fatigue and creep damage."""
        D_f, D_c, within = creep_fatigue.accumulated_damage(
            n_cycles_list=[500.0],
            delta_eps_list=[0.01],
            hold_times_s_list=[5000.0],
            T_list=[873.15],
            sigma_list=[100.0e6],
        )

        # Both should be positive
        assert D_f > 0.0
        assert D_c > 0.0

    def test_within_flag_accuracy(self, creep_fatigue):
        """Within flag should match is_within_envelope."""
        D_f, D_c, within = creep_fatigue.accumulated_damage(
            n_cycles_list=[100.0],
            delta_eps_list=[0.001],  # Low strain
            hold_times_s_list=[0.0],
            T_list=[873.15],
            sigma_list=[100.0e6],
        )

        # Low damage should be within
        assert within is True

        # Very high damage should be outside
        D_f_high, D_c_high, within_high = creep_fatigue.accumulated_damage(
            n_cycles_list=[100000.0],
            delta_eps_list=[0.05],  # High strain
            hold_times_s_list=[1000000.0],  # Long hold
            T_list=[873.15],
            sigma_list=[200.0e6],
        )

        assert within_high is False


class TestInteractionEnvelopePhysics:
    """Physics and monotonicity checks for interaction envelope."""

    def test_envelope_restricted_for_gr91(self, creep_fatigue, config):
        """9Cr-1Mo (Gr91) should have restrictive envelope."""
        # Intersection should be at (0.1, 0.01)
        df_int, dc_int = config.CF_INTERSECTION

        assert abs(df_int - 0.1) < 0.01, "Intersection should be ~(0.1, 0.01)"
        assert abs(dc_int - 0.01) < 0.005, "for Gr91 per ASME III-5"

    def test_higher_creep_damage_reduces_fatigue_allowance(self, creep_fatigue):
        """Higher creep damage should reduce allowable fatigue damage.

        For Gr91 the envelope is very restrictive: intersection at (0.1, 0.01).
        At D_f < 0.1 (segment 2), the creep limit is generous.
        At D_f > 0.1 (segment 1), the creep limit is tiny.
        """
        # (0.05, 0.005) should be within (segment 2, well inside)
        assert creep_fatigue.is_within_envelope(0.05, 0.005) is True

        # (0.05, 0.95) should be outside (segment 2, high creep)
        assert creep_fatigue.is_within_envelope(0.05, 0.95) is False

        # Margin should decrease as creep damage increases
        margin_low = creep_fatigue.envelope_margin(0.05, 0.1)
        margin_high = creep_fatigue.envelope_margin(0.05, 0.5)
        assert margin_low > margin_high, \
            "More creep damage should reduce margin"


class TestEnvelopeInteractionBoundary:
    """Test behavior near interaction envelope boundaries."""

    def test_creep_only_limit(self, creep_fatigue):
        """Creep-only loading should max out at D_c = 1."""
        within_just_in = creep_fatigue.is_within_envelope(D_f=0.0, D_c=0.99)
        within_just_out = creep_fatigue.is_within_envelope(D_f=0.0, D_c=1.01)

        assert within_just_in is True
        assert within_just_out is False

    def test_fatigue_only_limit(self, creep_fatigue):
        """Fatigue-only loading should max out at D_f = 1."""
        within_just_in = creep_fatigue.is_within_envelope(D_f=0.99, D_c=0.0)
        within_just_out = creep_fatigue.is_within_envelope(D_f=1.01, D_c=0.0)

        assert within_just_in is True
        assert within_just_out is False

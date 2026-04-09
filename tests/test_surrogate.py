"""
Comprehensive test suite for src/surrogate.py.

Tests GBR surrogate model training, prediction, feature importance, and cross-validation.
"""

import pytest
import numpy as np


class TestSurrogateInit:
    """Tests for surrogate model initialization."""

    def test_surrogate_creation(self, surrogate):
        """Should create surrogate model without error."""
        model = surrogate.TubeLifeSurrogate()
        assert model is not None
        assert model._fitted is False

    def test_surrogate_has_feature_names(self, surrogate):
        """Should have feature names defined."""
        assert len(surrogate.TubeLifeSurrogate.FEATURE_NAMES) == 5
        assert "temperature_K" in surrogate.TubeLifeSurrogate.FEATURE_NAMES
        assert "pressure_Pa" in surrogate.TubeLifeSurrogate.FEATURE_NAMES


class TestSurrogateFitAndPredict:
    """Tests for model fitting and prediction."""

    def test_fit_and_predict_without_error(self, surrogate):
        """Should fit and predict without error."""
        model = surrogate.TubeLifeSurrogate(n_estimators=10, random_state=42)

        # Create synthetic training data
        n_samples = 50
        X = np.random.rand(n_samples, 5) * 100 + 100  # Features in [100, 200]
        y_life = np.random.rand(n_samples) * 100000 + 10000  # Life in [10k, 110k]
        y_Df = np.random.rand(n_samples) * 0.3
        y_Dc = np.random.rand(n_samples) * 0.2

        # Fit
        model.fit(X, y_life, y_Df, y_Dc)
        assert model._fitted is True

        # Predict
        X_test = np.random.rand(5, 5) * 100 + 100
        life, Df, Dc = model.predict(X_test)

        assert len(life) == 5
        assert len(Df) == 5
        assert len(Dc) == 5

    def test_predictions_are_positive(self, surrogate):
        """Predictions should be positive (life, Df, Dc all > 0)."""
        model = surrogate.TubeLifeSurrogate(n_estimators=10, random_state=42)

        n_samples = 50
        X = np.random.rand(n_samples, 5) * 100 + 100
        y_life = np.random.rand(n_samples) * 100000 + 10000
        y_Df = np.random.rand(n_samples) * 0.3
        y_Dc = np.random.rand(n_samples) * 0.2

        model.fit(X, y_life, y_Df, y_Dc)

        X_test = np.random.rand(10, 5) * 100 + 100
        life, Df, Dc = model.predict(X_test)

        assert all(life > 0.0), "Predicted life must be positive"
        assert all(Df >= 0.0), "Predicted Df must be non-negative"
        assert all(Dc >= 0.0), "Predicted Dc must be non-negative"

    def test_predict_before_fit_raises(self, surrogate):
        """Predicting before fitting should raise error."""
        model = surrogate.TubeLifeSurrogate()

        with pytest.raises(RuntimeError):
            X_test = np.random.rand(5, 5)
            model.predict(X_test)


class TestFeatureImportance:
    """Tests for feature importance extraction."""

    def test_feature_importance_returns_dict(self, surrogate):
        """Feature importance should return dict."""
        model = surrogate.TubeLifeSurrogate(n_estimators=10, random_state=42)

        n_samples = 50
        X = np.random.rand(n_samples, 5) * 100 + 100
        y_life = np.random.rand(n_samples) * 100000 + 10000
        y_Df = np.random.rand(n_samples) * 0.3
        y_Dc = np.random.rand(n_samples) * 0.2

        model.fit(X, y_life, y_Df, y_Dc)

        imp_dict = model.feature_importance()
        assert isinstance(imp_dict, dict)

    def test_feature_importance_has_all_features(self, surrogate):
        """Feature importance should include all 5 features."""
        model = surrogate.TubeLifeSurrogate(n_estimators=10, random_state=42)

        n_samples = 50
        X = np.random.rand(n_samples, 5) * 100 + 100
        y_life = np.random.rand(n_samples) * 100000 + 10000
        y_Df = np.random.rand(n_samples) * 0.3
        y_Dc = np.random.rand(n_samples) * 0.2

        model.fit(X, y_life, y_Df, y_Dc)

        imp_dict = model.feature_importance()
        assert len(imp_dict) == 5

    def test_feature_importance_sums_to_one(self, surrogate):
        """Feature importance should sum to ~1.0."""
        model = surrogate.TubeLifeSurrogate(n_estimators=10, random_state=42)

        n_samples = 50
        X = np.random.rand(n_samples, 5) * 100 + 100
        y_life = np.random.rand(n_samples) * 100000 + 10000
        y_Df = np.random.rand(n_samples) * 0.3
        y_Dc = np.random.rand(n_samples) * 0.2

        model.fit(X, y_life, y_Df, y_Dc)

        imp_dict = model.feature_importance()
        imp_sum = sum(imp_dict.values())

        assert abs(imp_sum - 1.0) < 0.01, "Importances should sum to ~1"

    def test_feature_importance_all_positive(self, surrogate):
        """All feature importances should be positive."""
        model = surrogate.TubeLifeSurrogate(n_estimators=10, random_state=42)

        n_samples = 50
        X = np.random.rand(n_samples, 5) * 100 + 100
        y_life = np.random.rand(n_samples) * 100000 + 10000
        y_Df = np.random.rand(n_samples) * 0.3
        y_Dc = np.random.rand(n_samples) * 0.2

        model.fit(X, y_life, y_Df, y_Dc)

        imp_dict = model.feature_importance()
        assert all(imp >= 0.0 for imp in imp_dict.values())

    def test_feature_importance_before_fit_raises(self, surrogate):
        """Calling feature_importance before fit should raise."""
        model = surrogate.TubeLifeSurrogate()

        with pytest.raises(RuntimeError):
            model.feature_importance()


class TestCrossValidation:
    """Tests for cross-validation functionality."""

    def test_cross_validate_returns_scores(self, surrogate):
        """Cross-validation should return score array."""
        model = surrogate.TubeLifeSurrogate(n_estimators=10, random_state=42)

        n_samples = 100
        X = np.random.rand(n_samples, 5) * 100 + 100
        y_life = np.random.rand(n_samples) * 100000 + 10000

        model.fit(X, y_life, np.zeros(n_samples), np.zeros(n_samples))

        scores = model.cross_validate(X, y_life, cv=5)
        assert len(scores) == 5

    def test_cross_validate_r2_in_reasonable_range(self, surrogate):
        """R^2 scores should be in [-1, 1]."""
        model = surrogate.TubeLifeSurrogate(n_estimators=20, random_state=42)

        n_samples = 100
        np.random.seed(42)
        X = np.random.rand(n_samples, 5) * 100 + 100
        # Synthetic life as linear combination of features
        y_life = 10000 + X[:, 0] * 100 + X[:, 1] * 0.01 + np.random.rand(n_samples) * 1000

        model.fit(X, y_life, np.zeros(n_samples), np.zeros(n_samples))

        scores = model.cross_validate(X, y_life, cv=3)

        assert all(-1.0 <= s <= 1.0 for s in scores), "R^2 should be in [-1, 1]"

    def test_cross_validate_custom_cv(self, surrogate):
        """Should respect custom cv parameter."""
        model = surrogate.TubeLifeSurrogate(n_estimators=10, random_state=42)

        n_samples = 100
        X = np.random.rand(n_samples, 5) * 100 + 100
        y_life = np.random.rand(n_samples) * 100000 + 10000

        model.fit(X, y_life, np.zeros(n_samples), np.zeros(n_samples))

        scores_3fold = model.cross_validate(X, y_life, cv=3)
        scores_5fold = model.cross_validate(X, y_life, cv=5)

        assert len(scores_3fold) == 3
        assert len(scores_5fold) == 5


class TestParityPlot:
    """Tests for parity plot data generation."""

    def test_parity_returns_four_values(self, surrogate):
        """Parity method should return y_true, y_pred, r2, mae."""
        model = surrogate.TubeLifeSurrogate(n_estimators=10, random_state=42)

        n_samples = 50
        X = np.random.rand(n_samples, 5) * 100 + 100
        y_life = np.random.rand(n_samples) * 100000 + 10000
        y_Df = np.zeros(n_samples)
        y_Dc = np.zeros(n_samples)

        model.fit(X, y_life, y_Df, y_Dc)

        y_true, y_pred, r2, mae = model.parity_data(X, y_life)

        assert len(y_true) == n_samples
        assert len(y_pred) == n_samples
        assert isinstance(r2, (float, np.floating))
        assert isinstance(mae, (float, np.floating))

    def test_parity_r2_reasonable(self, surrogate):
        """R^2 from parity should be reasonable."""
        model = surrogate.TubeLifeSurrogate(n_estimators=20, random_state=42)

        n_samples = 100
        np.random.seed(42)
        X = np.random.rand(n_samples, 5) * 100 + 100
        y_life = 10000 + X[:, 0] * 100 + np.random.rand(n_samples) * 1000

        model.fit(X, y_life, np.zeros(n_samples), np.zeros(n_samples))

        y_true, y_pred, r2, mae = model.parity_data(X, y_life)

        assert -1.0 <= r2 <= 1.0, "R^2 should be in [-1, 1]"

    def test_parity_before_fit_raises(self, surrogate):
        """Parity before fit should raise."""
        model = surrogate.TubeLifeSurrogate()

        with pytest.raises(RuntimeError):
            X = np.random.rand(5, 5)
            y = np.random.rand(5)
            model.parity_data(X, y)


class TestBuildIsoRiskGrid:
    """Tests for iso-risk contour grid building."""

    def test_iso_risk_grid_generation(self, surrogate):
        """Should generate iso-risk grid without error."""
        model = surrogate.TubeLifeSurrogate(n_estimators=10, random_state=42)

        n_samples = 50
        X = np.random.rand(n_samples, 5) * 100 + 100
        y_life = np.random.rand(n_samples) * 100000 + 10000
        y_Df = np.zeros(n_samples)
        y_Dc = np.zeros(n_samples)

        model.fit(X, y_life, y_Df, y_Dc)

        fixed_params = {
            "temperature_K": 873.15,
            "pressure_Pa": 5.0e6,
            "wall_thickness_m": 0.006,
            "cycles_per_year": 8.0,
            "delta_T_K": 300.0,
        }

        P1, P2, life_grid, Df_grid, Dc_grid = surrogate.build_iso_risk_grid(
            model,
            "temperature_K", (800, 950),
            "pressure_Pa", (3.0e6, 7.0e6),
            fixed_params,
            n_grid=20
        )

        assert P1.shape == (20, 20)
        assert P2.shape == (20, 20)
        assert life_grid.shape == (20, 20)
        assert Df_grid.shape == (20, 20)
        assert Dc_grid.shape == (20, 20)

    def test_iso_risk_grid_has_positive_life(self, surrogate):
        """Grid life values should be positive."""
        model = surrogate.TubeLifeSurrogate(n_estimators=10, random_state=42)

        n_samples = 50
        X = np.random.rand(n_samples, 5) * 100 + 100
        y_life = np.random.rand(n_samples) * 100000 + 10000
        y_Df = np.zeros(n_samples)
        y_Dc = np.zeros(n_samples)

        model.fit(X, y_life, y_Df, y_Dc)

        fixed_params = {
            "temperature_K": 873.15,
            "pressure_Pa": 5.0e6,
            "wall_thickness_m": 0.006,
            "cycles_per_year": 8.0,
            "delta_T_K": 300.0,
        }

        P1, P2, life_grid, Df_grid, Dc_grid = surrogate.build_iso_risk_grid(
            model,
            "temperature_K", (800, 950),
            "pressure_Pa", (3.0e6, 7.0e6),
            fixed_params,
            n_grid=15
        )

        assert np.all(life_grid > 0.0), "All life values should be positive"

    def test_iso_risk_grid_shape_and_values(self, surrogate):
        """Iso-risk grid should return correct shape and finite values."""
        model = surrogate.TubeLifeSurrogate(n_estimators=10, random_state=42)

        n_samples = 100
        np.random.seed(42)
        # Generate data in physically meaningful ranges
        T_vals = np.random.uniform(833, 903, n_samples)
        P_vals = np.random.uniform(3e6, 7e6, n_samples)
        wt_vals = np.random.uniform(0.004, 0.007, n_samples)
        cyc_vals = np.random.uniform(4, 15, n_samples)
        dT_vals = np.random.uniform(200, 380, n_samples)
        X = np.column_stack([T_vals, P_vals, wt_vals, cyc_vals, dT_vals])

        y_life = 200000 - T_vals * 200 + np.random.rand(n_samples) * 5000
        y_Df = np.zeros(n_samples)
        y_Dc = np.zeros(n_samples)
        model.fit(X, y_life, y_Df, y_Dc)

        fixed_params = {
            "temperature_K": 873.15,
            "pressure_Pa": 5.0e6,
            "wall_thickness_m": 0.006,
            "cycles_per_year": 8.0,
            "delta_T_K": 300.0,
        }

        P1, P2, life_grid, Df_grid, Dc_grid = surrogate.build_iso_risk_grid(
            model,
            "temperature_K", (833.0, 903.0),
            "pressure_Pa", (3.0e6, 7.0e6),
            fixed_params,
            n_grid=15
        )

        assert life_grid.shape == (15, 15), "Grid should be n_grid x n_grid"
        assert np.all(np.isfinite(life_grid)), "All grid values should be finite"
        assert np.all(life_grid > 0), "All life values should be positive"


class TestSurrogateRobustness:
    """Tests for surrogate model robustness."""

    def test_handles_edge_case_inputs(self, surrogate):
        """Should handle edge case feature values."""
        model = surrogate.TubeLifeSurrogate(n_estimators=10, random_state=42)

        n_samples = 50
        X = np.random.rand(n_samples, 5) * 100 + 100
        y_life = np.random.rand(n_samples) * 100000 + 10000
        y_Df = np.zeros(n_samples)
        y_Dc = np.zeros(n_samples)

        model.fit(X, y_life, y_Df, y_Dc)

        # Test with very high values
        X_extreme = np.array([[1000, 1.0e7, 0.01, 20.0, 500.0]])
        life, Df, Dc = model.predict(X_extreme)

        assert np.isfinite(life[0])
        assert np.isfinite(Df[0])
        assert np.isfinite(Dc[0])

    def test_multiple_models_same_data(self, surrogate):
        """Multiple models on same data should give similar predictions."""
        n_samples = 100
        np.random.seed(42)
        X = np.random.rand(n_samples, 5) * 100 + 100
        y_life = np.random.rand(n_samples) * 100000 + 10000
        y_Df = np.zeros(n_samples)
        y_Dc = np.zeros(n_samples)

        model1 = surrogate.TubeLifeSurrogate(n_estimators=10, random_state=42)
        model1.fit(X, y_life, y_Df, y_Dc)

        model2 = surrogate.TubeLifeSurrogate(n_estimators=10, random_state=42)
        model2.fit(X, y_life, y_Df, y_Dc)

        X_test = np.random.rand(10, 5) * 100 + 100

        life1, Df1, Dc1 = model1.predict(X_test)
        life2, Df2, Dc2 = model2.predict(X_test)

        # Same random seed should give same results
        assert np.allclose(life1, life2)
        assert np.allclose(Df1, Df2)
        assert np.allclose(Dc1, Dc2)

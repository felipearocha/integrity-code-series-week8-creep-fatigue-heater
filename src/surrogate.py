"""
Gradient Boosting Regressor (GBR) surrogate model for tube life prediction.

Trained on Monte Carlo / LHS sweep results.
Predicts: failure time, dominant failure mode probability, and damage fractions.
"""

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error


class TubeLifeSurrogate:
    """GBR surrogate for creep-fatigue-oxidation tube life."""

    FEATURE_NAMES = [
        "temperature_K",
        "pressure_Pa",
        "wall_thickness_m",
        "cycles_per_year",
        "delta_T_K",
    ]

    def __init__(self, n_estimators=300, max_depth=5, learning_rate=0.05,
                 random_state=42):
        self.model_life = GradientBoostingRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=random_state,
        )
        self.model_Df = GradientBoostingRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=random_state,
        )
        self.model_Dc = GradientBoostingRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=random_state,
        )
        self._fitted = False

    def fit(self, X: np.ndarray, y_life: np.ndarray,
            y_Df: np.ndarray, y_Dc: np.ndarray):
        """
        Train surrogate models.

        Parameters
        ----------
        X : ndarray, shape (n_samples, 5)
            Feature matrix.
        y_life : ndarray
            Failure time [hours].
        y_Df : ndarray
            Final fatigue damage fraction.
        y_Dc : ndarray
            Final creep damage fraction.
        """
        self.model_life.fit(X, np.log10(np.clip(y_life, 1.0, None)))
        self.model_Df.fit(X, y_Df)
        self.model_Dc.fit(X, y_Dc)
        self._fitted = True

    def predict(self, X: np.ndarray):
        """
        Predict tube life and damage fractions.

        Returns
        -------
        life_hours : ndarray
        Df : ndarray
        Dc : ndarray
        """
        if not self._fitted:
            raise RuntimeError("Surrogate not fitted. Call fit() first.")
        log_life = self.model_life.predict(X)
        life = 10.0 ** log_life
        Df = self.model_Df.predict(X)
        Dc = self.model_Dc.predict(X)
        return life, Df, Dc

    def feature_importance(self):
        """
        Get feature importance from life model.

        Returns
        -------
        dict : feature_name -> importance
        """
        if not self._fitted:
            raise RuntimeError("Surrogate not fitted.")
        imp = self.model_life.feature_importances_
        return dict(zip(self.FEATURE_NAMES, imp))

    def cross_validate(self, X, y_life, cv=5):
        """
        Cross-validate the life model.

        Returns
        -------
        scores : ndarray
            R^2 scores for each fold.
        """
        return cross_val_score(
            self.model_life, X, np.log10(np.clip(y_life, 1.0, None)),
            cv=cv, scoring="r2"
        )

    def parity_data(self, X, y_life_true):
        """
        Generate parity plot data (predicted vs actual).

        Returns
        -------
        y_true : ndarray
        y_pred : ndarray
        r2 : float
        mae : float
        """
        if not self._fitted:
            raise RuntimeError("Surrogate not fitted.")
        log_pred = self.model_life.predict(X)
        log_true = np.log10(np.clip(y_life_true, 1.0, None))
        r2 = r2_score(log_true, log_pred)
        mae = mean_absolute_error(log_true, log_pred)
        return y_life_true, 10.0 ** log_pred, r2, mae


def build_iso_risk_grid(surrogate, param1_name, param1_range,
                         param2_name, param2_range,
                         fixed_params, n_grid=80):
    """
    Build 2D iso-risk contour data for two parameters.

    Parameters
    ----------
    surrogate : TubeLifeSurrogate
    param1_name : str
    param1_range : tuple (min, max)
    param2_name : str
    param2_range : tuple (min, max)
    fixed_params : dict
        Fixed parameter values (median from MC).
    n_grid : int
        Grid resolution per axis.

    Returns
    -------
    P1 : ndarray (n_grid, n_grid)
    P2 : ndarray (n_grid, n_grid)
    life_grid : ndarray (n_grid, n_grid)
    Df_grid : ndarray
    Dc_grid : ndarray
    """
    p1 = np.linspace(*param1_range, n_grid)
    p2 = np.linspace(*param2_range, n_grid)
    P1, P2 = np.meshgrid(p1, p2)

    X_grid = np.zeros((n_grid * n_grid, 5))
    for j, name in enumerate(TubeLifeSurrogate.FEATURE_NAMES):
        if name == param1_name:
            X_grid[:, j] = P1.ravel()
        elif name == param2_name:
            X_grid[:, j] = P2.ravel()
        else:
            X_grid[:, j] = fixed_params[name]

    life, Df, Dc = surrogate.predict(X_grid)

    return (P1, P2,
            life.reshape(n_grid, n_grid),
            Df.reshape(n_grid, n_grid),
            Dc.reshape(n_grid, n_grid))

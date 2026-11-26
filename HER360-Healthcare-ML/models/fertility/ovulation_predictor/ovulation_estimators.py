"""
HER360 Fertility - Ovulation Predictor: estimator definitions (v1)

Kept in a module (not in a notebook) so that:
  - every notebook uses identical definitions,
  - a saved model can be loaded later by the API.
"""
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor


class GlobalMedian(BaseEstimator, RegressorMixin):
    """Always predict the training median."""
    def fit(self, X, y):
        self.m_ = float(np.median(y))
        return self

    def predict(self, X):
        return np.full(len(X), self.m_)


class CalendarOffset(BaseEstimator, RegressorMixin):
    """prev_cycle_mean + offset, where offset is learned from the training data."""
    def fit(self, X, y):
        self.offset_ = float(np.median(np.asarray(y) - X["prev_cycle_mean"].to_numpy()))
        return self

    def predict(self, X):
        return X["prev_cycle_mean"].to_numpy() + self.offset_


def _imputer():
    return SimpleImputer(strategy="median", add_indicator=True)


def build_models():
    """The five models compared in notebook 04 (same settings)."""
    return {
        "global_median": GlobalMedian(),
        "calendar_offset": CalendarOffset(),
        "linear_regression": make_pipeline(_imputer(), LinearRegression()),
        "random_forest": make_pipeline(_imputer(), RandomForestRegressor(
            n_estimators=300, min_samples_leaf=10, max_features=0.7,
            random_state=42, n_jobs=-1)),
        "gradient_boosting": make_pipeline(_imputer(), GradientBoostingRegressor(
            n_estimators=150, learning_rate=0.03, max_depth=2,
            subsample=0.8, min_samples_leaf=10, random_state=42)),
    }

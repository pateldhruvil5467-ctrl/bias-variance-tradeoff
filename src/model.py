"""
Final polynomial regression model.

This module provides a reusable implementation of the final model
selected during the bias-variance trade-off analysis.
"""

from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures


POLYNOMIAL_DEGREE = 6


def create_model(degree: int = POLYNOMIAL_DEGREE) -> Pipeline:
    """
    Create the polynomial regression pipeline.

    Parameters
    ----------
    degree : int
        Polynomial feature degree.

    Returns
    -------
    Pipeline
        Configured polynomial regression pipeline.
    """
    if degree < 1:
        raise ValueError("Polynomial degree must be at least 1.")

    return Pipeline(
        [
            ("polynomial_features", PolynomialFeatures(degree=degree)),
            ("regression", LinearRegression()),
        ]
    )


def train_model(X, y, degree: int = POLYNOMIAL_DEGREE) -> Pipeline:
    """
    Train the polynomial regression model.

    Parameters
    ----------
    X : array-like
        Training features.
    y : array-like
        Training targets.
    degree : int
        Polynomial feature degree.

    Returns
    -------
    Pipeline
        Fitted polynomial regression pipeline.
    """
    model = create_model(degree)
    model.fit(X, y)
    return model


def predict(model: Pipeline, X):
    """
    Generate predictions using a trained model.

    Parameters
    ----------
    model : Pipeline
        Fitted polynomial regression pipeline.
    X : array-like
        Input features.

    Returns
    -------
    array-like
        Model predictions.
    """
    return model.predict(X)
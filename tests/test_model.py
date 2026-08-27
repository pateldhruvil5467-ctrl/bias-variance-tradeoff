"""
Tests for the final polynomial regression model.
"""

import numpy as np
import pytest

from src.model import create_model, train_model, predict


def test_create_model():
    """Verify that the model pipeline is created correctly."""
    model = create_model()

    assert model is not None
    assert "polynomial_features" in model.named_steps
    assert "regression" in model.named_steps


def test_create_model_invalid_degree():
    """Verify that invalid polynomial degrees are rejected."""
    with pytest.raises(ValueError):
        create_model(degree=0)


def test_train_model():
    """Verify that the model can be trained."""
    X = np.array([[-2], [-1], [0], [1], [2]])
    y = np.array([4, 1, 0, 1, 4])

    model = train_model(X, y, degree=2)

    assert model is not None


def test_prediction_shape():
    """Verify that predictions have the expected shape."""
    X_train = np.array([[-2], [-1], [0], [1], [2]])
    y_train = np.array([4, 1, 0, 1, 4])

    X_test = np.array([[-1.5], [0.5], [1.5]])

    model = train_model(X_train, y_train, degree=2)
    predictions = predict(model, X_test)

    assert predictions.shape == (3,)


def test_predictions_are_finite():
    """Verify that model predictions contain no NaN or infinite values."""
    X_train = np.array([[-2], [-1], [0], [1], [2]])
    y_train = np.array([4, 1, 0, 1, 4])

    X_test = np.array([[-1.5], [0.5], [1.5]])

    model = train_model(X_train, y_train, degree=2)
    predictions = predict(model, X_test)

    assert np.all(np.isfinite(predictions))
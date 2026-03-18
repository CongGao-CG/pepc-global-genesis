"""Core prediction function for single-basin numpy arrays."""

from importlib import resources
from pathlib import Path
from typing import Tuple

import joblib
import numpy as np

from .basins import BASINS


def _get_data_dir() -> Path:
    """Get the path to the package data directory."""
    return resources.files("pepc_global_genesis") / "data"


def _load_model(basin: str):
    """Load the trained SVC model for a given basin."""
    if basin not in BASINS:
        raise ValueError(f"Invalid basin '{basin}'. Must be one of: {BASINS}")
    model_path = _get_data_dir() / f"genesis_SVC_{basin}.joblib"
    return joblib.load(model_path)


def _load_scaler(basin: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load the scaling parameters (mean, std) for a given basin."""
    if basin not in BASINS:
        raise ValueError(f"Invalid basin '{basin}'. Must be one of: {BASINS}")
    scale_path = _get_data_dir() / f"genesis_predictors_{basin}_scale.npy"
    scale_data = np.load(scale_path)
    return scale_data[0], scale_data[1]


def predict_genesis(
    av850: np.ndarray,
    shr: np.ndarray,
    rh600: np.ndarray,
    pi: np.ndarray,
    basin: str,
) -> np.ndarray:
    """
    Predict tropical cyclone genesis using trained SVC models.

    Given four environmental predictor variables as numpy arrays of the same shape,
    this function returns a binary array (0 or 1) of the same shape indicating
    predicted tropical cyclone genesis for a specified basin.

    Parameters
    ----------
    av850 : np.ndarray
        Absolute vorticity at 850 hPa (s^-1).
    shr : np.ndarray
        Vertical wind shear between 200 hPa and 850 hPa (m/s).
    rh600 : np.ndarray
        Relative humidity at 600 hPa (%).
    pi : np.ndarray
        Potential intensity (m/s).
    basin : str
        Basin name. Must be one of: 'AS', 'BoB', 'WNP', 'ENP', 'NA', 'SI', 'SP'.

    Returns
    -------
    np.ndarray
        Binary array (0 or 1) indicating predicted genesis.
        Same shape as input arrays.

    Raises
    ------
    ValueError
        If basin is invalid or if input array shapes don't match.

    Examples
    --------
    >>> import numpy as np
    >>> from pepc_global_genesis import predict_genesis
    >>>
    >>> # Create predictor arrays (shape: lat x lon or time x lat x lon)
    >>> av850 = np.random.rand(12, 72, 144) * 1e-4
    >>> shr = np.random.rand(12, 72, 144) * 20
    >>> rh600 = np.random.rand(12, 72, 144) * 50 + 30
    >>> pi = np.random.rand(12, 72, 144) * 20 + 50
    >>>
    >>> # Predict genesis for North Atlantic
    >>> genesis_flag = predict_genesis(av850, shr, rh600, pi, basin='NA')
    """
    # Validate basin
    if basin not in BASINS:
        raise ValueError(f"Invalid basin '{basin}'. Must be one of: {BASINS}")

    # Validate input shapes
    if not (av850.shape == shr.shape == rh600.shape == pi.shape):
        raise ValueError(
            f"All input arrays must have the same shape. "
            f"Got: av850={av850.shape}, shr={shr.shape}, "
            f"rh600={rh600.shape}, pi={pi.shape}"
        )

    # Stack predictors along feature dimension
    original_shape = av850.shape
    X = np.stack(
        [
            av850.ravel(),
            shr.ravel(),
            rh600.ravel(),
            pi.ravel(),
        ],
        axis=-1,
    ).astype(np.float32)

    # Load model and scaler
    means, stds = _load_scaler(basin)
    model = _load_model(basin)

    # Scale and predict
    X_scaled = (X - means) / stds
    y_pred = model.predict(X_scaled)

    # Reshape to original shape
    y_pred = y_pred.reshape(original_shape)

    return y_pred

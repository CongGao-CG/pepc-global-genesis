"""
PepC-Global Genesis - Tropical Cyclone Genesis Prediction

A Python package for predicting tropical cyclone genesis
using trained Support Vector Classification (SVC) models.
"""

from .basins import BASINS, BASIN_GENESIS_RANGES, get_basin_names
from ._core import predict_genesis
from .global_predict import predict_genesis_global

__version__ = "1.2.2"
__all__ = [
    "predict_genesis",
    "predict_genesis_global",
    "get_basin_names",
    "BASINS",
    "BASIN_GENESIS_RANGES",
]

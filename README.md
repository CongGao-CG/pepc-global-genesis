# PepC-Global Genesis

A Python package for predicting tropical cyclone genesis using trained Support Vector Classification (SVC) models.

## Installation

```bash
pip install --upgrade pepc-global-genesis
```

Or from source:

```bash
git clone https://github.com/CongGao-CG/pepc-global-genesis.git
cd pepc-global-genesis
pip install .
```

## Quick Start

```python
import numpy as np
from pepc_global_genesis import predict_genesis

# Create predictor arrays (all must have the same shape)
# Shape can be (lat, lon) or (time, lat, lon), etc.
av850 = np.random.uniform(-1e-4, 1e-4, size=(10, 10))  # Absolute vorticity at 850 hPa (s^-1)
shr = np.random.uniform(0, 20, size=(10, 10))          # Vertical wind shear between 200 hPa and 850 hPa (m s^-1)
rh600 = np.random.uniform(30, 80, size=(10, 10))       # Relative humidity at 600 hPa (%)
pi = np.random.uniform(0, 100, size=(10, 10))          # Potential intensity (m s^-1)

# Predict genesis for a specific basin
basin = 'NA'  # North Atlantic
genesis_flag = predict_genesis(av850, shr, rh600, pi, basin=basin)

# genesis_flag is a numpy array of 0s and 1s with the same shape as inputs
print(f"Predicted genesis: {np.sum(genesis_flag)} grid points")
```

## Available Basins

| Basin Code | Name                  | Latitude Range    | Longitude Range     |
|------------|-----------------------|-------------------|---------------------|
| `AS`       | Arabian Sea           | 5° to 22.5°N      | 50° to 77.5°E       |
| `BoB`      | Bay of Bengal         | 5° to 22.5°N      | 80° to 100°E        |
| `WNP`      | Western North Pacific | 5° to 30°N        | 102.5°E to 180°     |
| `ENP`      | Eastern North Pacific | 5° to 25°N        | 177.5° to 75°W      |
| `NA`       | North Atlantic        | 5° to 30°N        | 97.5° to 2.5°W      |
| `SI`       | South Indian          | 30° to 5°S        | 20° to 145°E        |
| `SP`       | South Pacific         | 30° to 5°S        | 147.5°E to 100°W    |

## API Reference

### `predict_genesis(av850, shr, rh600, pi, basin)`

Predict tropical cyclone genesis.

**Parameters:**
- `av850` (np.ndarray): Absolute vorticity at 850 hPa (s^-1)
- `shr` (np.ndarray): Vertical wind shear between 200-850 hPa (m/s)
- `rh600` (np.ndarray): Relative humidity at 600 hPa (%)
- `pi` (np.ndarray): Potential intensity (m/s)
- `basin` (str): Basin name (one of: 'AS', 'BoB', 'WNP', 'ENP', 'NA', 'SI', 'SP')

**Returns:**
- `np.ndarray`: Binary array (0 or 1) indicating predicted genesis, same shape as inputs

**Raises:**
- `ValueError`: If basin is invalid or input array shapes don't match

### `get_basin_names()`

Get the list of valid basin names.

**Returns:**
- `list[str]`: List of 7 basin names

### `BASINS`

List of valid basin names.

## Input Variables

| Variable | Description                          | Typical Units |
|----------|--------------------------------------|---------------|
| av850    | Absolute vorticity at 850 hPa        | s^-1          |
| shr      | Vertical wind shear between 200 hPa and 850 hPa | m s^-1        |
| rh600    | Relative humidity at 600 hPa         | %             |
| pi       | Potential intensity                  | m s^-1        |

## Model Details

This package contains pre-trained SVC models for 7 tropical cyclone basins. The models were trained on ERA5 monthly data using four environmental predictors known to influence tropical cyclone genesis:

1. **Absolute vorticity (av850)**: Measures rotation in the lower-level atmosphere
2. **Wind shear (shr)**: Vector difference between upper-level and lower-level winds
3. **Relative humidity (rh600)**: Mid-level moisture
4. **Potential intensity (pi)**: Upper limit of tropical cyclone intensity

Each basin has its own trained model and scaling parameters stored within the package.

## License

MIT License

## Citation

If you use this package in your research, please cite:

Gao, Cong, Ning Lin. "PepC-Global: A Basin-Tuned Probabilistic Tropical Cyclone Model with Enhanced Out-of-Sample Skill and Climate-Sensitive Over-Land Decay". Journal of Advances in Modeling Earth Systems (JAMES), under review.

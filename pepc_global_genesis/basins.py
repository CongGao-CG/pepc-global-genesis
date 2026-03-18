"""Basin definitions for tropical cyclone genesis prediction."""

from typing import Dict, List


def get_basin_names() -> List[str]:
    """
    Get the list of valid basin names for tropical cyclone genesis prediction.

    Returns:
        List of 7 basin names.

    Basins:
        - AS: Arabian Sea (5° to 22.5°N, 50° to 77.5°E)
        - BoB: Bay of Bengal (5° to 22.5°N, 80° to 100°E)
        - WNP: Western North Pacific (5° to 30°N, 102.5°E to 180°)
        - ENP: Eastern North Pacific (5° to 25°N, 182.5° to 285°E)
        - NA: North Atlantic (5° to 30°N, 262.5° to 357.5°E)
        - SI: South Indian (30° to 5°S, 20° to 145°E)
        - SP: South Pacific (30° to 5°S, 147.5° to 260°E)
    """
    return ['AS', 'BoB', 'WNP', 'ENP', 'NA', 'SI', 'SP']


BASINS = get_basin_names()

# Longitude/latitude ranges for genesis prediction (all longitudes in 0–360°E).
# Order matters: ENP before NA so that NA overwrites ENP in the overlap region
# after per-basin exclusions are applied.
BASIN_GENESIS_RANGES: Dict[str, Dict] = {
    'AS':  dict(grid_lat=(5, 22.5),  grid_lon=(50, 77.5)),
    'BoB': dict(grid_lat=(5, 22.5),  grid_lon=(80, 100)),
    'WNP': dict(grid_lat=(5, 30),    grid_lon=(102.5, 180)),
    'ENP': dict(grid_lat=(5, 25),    grid_lon=(182.5, 285)),
    'NA':  dict(grid_lat=(5, 30),    grid_lon=(262.5, 357.5)),
    'SI':  dict(grid_lat=(-30, -5),  grid_lon=(20, 145)),
    'SP':  dict(grid_lat=(-30, -5),  grid_lon=(147.5, 260)),
}

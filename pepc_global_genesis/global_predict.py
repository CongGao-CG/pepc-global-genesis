"""Global tropical cyclone genesis prediction on gridded xarray DataArrays."""

from importlib import resources

import numpy as np
import xarray as xr

from .basins import BASINS, BASIN_GENESIS_RANGES
from ._core import predict_genesis

# Expected 2.5-degree grid coordinates
_LON_2P5 = np.arange(0, 360, 2.5)
_LAT_2P5_N2S = np.arange(90, -90.1, -2.5)   # 90 to -90
_LAT_2P5_S2N = np.arange(-90, 90.1, 2.5)    # -90 to 90


def _detect_latlon_dims(da):
    """Auto-detect latitude and longitude dimension names."""
    lat_name = lon_name = None
    for name in da.dims:
        low = name.lower()
        if low in ('lat', 'latitude'):
            lat_name = name
        elif low in ('lon', 'longitude'):
            lon_name = name
    if lat_name is None or lon_name is None:
        raise ValueError(
            f"Cannot detect lat/lon dimensions from {list(da.dims)}. "
            f"Expected 'lat'/'latitude' and 'lon'/'longitude'."
        )
    return lat_name, lon_name


def _validate_grid(lon_vals, lat_vals):
    """Validate coordinates are on the expected 2.5-degree grid."""
    if not np.allclose(lon_vals, _LON_2P5):
        raise ValueError(
            f"Longitude must be 0, 2.5, 5, ..., 357.5 "
            f"(got {lon_vals[:4]}...{lon_vals[-2:]})"
        )
    if not (np.allclose(lat_vals, _LAT_2P5_N2S) or
            np.allclose(lat_vals, _LAT_2P5_S2N)):
        raise ValueError(
            f"Latitude must be 90, 87.5, ..., -90 or -90, -87.5, ..., 90 "
            f"(got {lat_vals[:4]}...{lat_vals[-2:]})"
        )


def _load_ocean_mask(lat_vals):
    """Load ocean mask and align to the input latitude order.

    Returns a boolean array of shape (73, 144) where True = ocean.
    """
    mask_path = (resources.files("pepc_global_genesis")
                 / "data" / "ocean_mask_remapmean_2p5.nc")
    ds = xr.open_dataset(mask_path, decode_times=False)
    ocean = ds["ocean"].squeeze().values  # (73, 144), lat 90 -> -90
    ds.close()
    # Flip if input latitude runs S -> N
    if lat_vals[0] < lat_vals[-1]:
        ocean = ocean[::-1]
    return ocean > 0.5


def _apply_exclusions(lat2d, lon2d, mask, basin):
    """Apply ENP/NA overlap exclusion rules.

    Adapted from ENP_NA_extra_mask.py.  In the overlap region (lon 262.5-285),
    specific grid cells are removed from each basin so that every cell is
    assigned to at most one of ENP or NA.
    """
    if basin == "NA":
        mask = mask & ~(
            (lat2d >= 5) & (lat2d <= 15) &
            (lon2d >= 262.5) & (lon2d <= 272.5)
        )
        mask = mask & ~(
            (lat2d >= 5) & (lat2d <= 7.5) &
            (lon2d >= 275) & (lon2d <= 282.5)
        )
        mask = mask & ~(
            (lat2d >= 12.5) & (lat2d <= 15) &
            (lon2d >= 274) & (lon2d <= 276)
        )
    elif basin == "ENP":
        mask = mask & ~(
            (lat2d >= 20) & (lat2d <= 25) &
            (lon2d >= 265) & (lon2d <= 285)
        )
        mask = mask & ~(
            (lat2d >= 15) & (lat2d <= 17.5) &
            (lon2d >= 270) & (lon2d <= 285)
        )
        mask = mask & ~(
            (lat2d >= 10) & (lat2d <= 12.5) &
            (lon2d >= 275) & (lon2d <= 285)
        )
        mask = mask & ~(
            (lat2d >= 21.5) & (lat2d <= 23.5) &
            (lon2d >= 261.5) & (lon2d <= 263.5)
        )
    return mask


def _build_basin_map(lat, lon):
    """Map each (lat, lon) cell to a basin name, or empty string if outside.

    Processing order: ENP is assigned before NA, so in the small number of
    overlap cells that survive both basins' exclusions, NA takes precedence.
    """
    lon2d, lat2d = np.meshgrid(lon, lat)
    basin_map = np.full((len(lat), len(lon)), '', dtype='U3')

    for basin, cfg in BASIN_GENESIS_RANGES.items():
        lat_lo, lat_hi = cfg['grid_lat']
        lon_lo, lon_hi = cfg['grid_lon']
        mask = (
            (lat2d >= lat_lo) & (lat2d <= lat_hi) &
            (lon2d >= lon_lo) & (lon2d <= lon_hi)
        )
        if basin in ('ENP', 'NA'):
            mask = _apply_exclusions(lat2d, lon2d, mask, basin)
        basin_map[mask] = basin

    return basin_map


def predict_genesis_global(av850, shr, rh600, pi):
    """Predict TC genesis globally on gridded xarray DataArrays.

    Parameters
    ----------
    av850 : xr.DataArray
        Absolute vorticity at 850 hPa (s^-1).
    shr : xr.DataArray
        Vertical wind shear between 200 and 850 hPa (m/s).
    rh600 : xr.DataArray
        Relative humidity at 600 hPa (%).
    pi : xr.DataArray
        Potential intensity (m/s).

    All inputs must share the same dimensions and be on a 2.5-degree global
    grid (lon: 0, 2.5, ..., 357.5; lat: 90 to -90 or -90 to 90).
    Longitudes in -180..180 are converted to 0..360 internally.

    Returns
    -------
    xr.Dataset
        Two variables:

        - **genesis** (float): 1 = genesis, 0 = no genesis, NaN = not
          predicted (land or outside all basins).
        - **basin** (str): cell classification — one of the 7 basin names
          (``'AS'``, ``'BoB'``, ``'WNP'``, ``'ENP'``, ``'NA'``, ``'SI'``,
          ``'SP'``), ``'ocean_outside'`` (ocean but outside all basins),
          ``'land_outside'`` (land outside all basins), or
          ``'land_inside'`` (land inside a basin's lat/lon range).
    """
    lat_name, lon_name = _detect_latlon_dims(av850)

    lat_vals = av850[lat_name].values.astype(np.float64)
    lon_vals = av850[lon_name].values.astype(np.float64)

    # Convert -180..180 to 0..360, sorting to 0, 2.5, ..., 357.5
    lon_converted = False
    if np.any(lon_vals < 0):
        lon_vals = lon_vals % 360
        lon_sort_idx = np.argsort(lon_vals)
        lon_vals = lon_vals[lon_sort_idx]
        lon_converted = True

    _validate_grid(lon_vals, lat_vals)

    ocean_mask = _load_ocean_mask(lat_vals)
    basin_map = _build_basin_map(lat_vals, lon_vals)

    # Move lat and lon to the last two axes for uniform indexing
    other_dims = [d for d in av850.dims if d not in (lat_name, lon_name)]
    target_order = other_dims + [lat_name, lon_name]
    perm = [list(av850.dims).index(d) for d in target_order]

    av850_np = av850.values.transpose(perm)
    shr_np = shr.values.transpose(perm)
    rh600_np = rh600.values.transpose(perm)
    pi_np = pi.values.transpose(perm)

    # Reorder lon axis if we converted from -180..180
    if lon_converted:
        av850_np = av850_np[..., lon_sort_idx]
        shr_np = shr_np[..., lon_sort_idx]
        rh600_np = rh600_np[..., lon_sort_idx]
        pi_np = pi_np[..., lon_sort_idx]

    result = np.full(av850_np.shape, np.nan, dtype=np.float64)

    for basin in BASINS:
        spatial_mask = (basin_map == basin) & ocean_mask
        if not spatial_mask.any():
            continue
        lat_idx, lon_idx = np.where(spatial_mask)

        a_sub = av850_np[..., lat_idx, lon_idx]
        s_sub = shr_np[..., lat_idx, lon_idx]
        r_sub = rh600_np[..., lat_idx, lon_idx]
        p_sub = pi_np[..., lat_idx, lon_idx]

        # Mask out cells where any predictor is NaN
        valid = (np.isfinite(a_sub) & np.isfinite(s_sub) &
                 np.isfinite(r_sub) & np.isfinite(p_sub))

        if not valid.any():
            continue

        if valid.all():
            pred = predict_genesis(a_sub, s_sub, r_sub, p_sub, basin=basin)
            result[..., lat_idx, lon_idx] = pred.astype(np.float64)
        else:
            # Predict only where all four predictors are finite
            pred_flat = np.full(a_sub.shape, np.nan, dtype=np.float64)
            v = valid.ravel()
            if v.any():
                pred_flat.ravel()[v] = predict_genesis(
                    a_sub.ravel()[v],
                    s_sub.ravel()[v],
                    r_sub.ravel()[v],
                    p_sub.ravel()[v],
                    basin=basin,
                ).astype(np.float64)
            result[..., lat_idx, lon_idx] = pred_flat

    # Build basin label array (2D: lat x lon)
    in_basin = basin_map != ''
    label = np.where(
        ocean_mask & in_basin, basin_map,
        np.where(ocean_mask & ~in_basin, 'ocean_outside',
                 np.where(~ocean_mask & in_basin, 'land_inside',
                          'land_outside'))
    )

    # Undo lon reordering to match original input order
    if lon_converted:
        inv_lon_idx = np.argsort(lon_sort_idx)
        result = result[..., inv_lon_idx]
        label = label[:, inv_lon_idx]

    # Transpose back to original dimension order
    inv_perm = [0] * len(perm)
    for i, p in enumerate(perm):
        inv_perm[p] = i
    result = result.transpose(inv_perm)

    genesis_da = xr.DataArray(
        result,
        coords=av850.coords,
        dims=av850.dims,
        name='genesis',
    )
    basin_da = xr.DataArray(
        label,
        coords={lat_name: av850[lat_name], lon_name: av850[lon_name]},
        dims=[lat_name, lon_name],
        name='basin',
    )

    return xr.Dataset({'genesis': genesis_da, 'basin': basin_da})

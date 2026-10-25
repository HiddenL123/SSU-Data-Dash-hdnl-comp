"""Coordinate transforms for experimental spatial model families.

Shared geometry lives in features/, alongside the reusable feature providers.
These features depend only on listing-time inputs and use a fixed Toronto
reference origin; no validation labels are consulted or fitted.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

FEATURE_SET = "Xpt_spatiotemporal__selected_binary"
SPLINE_GEO_SUFFIX = "__st_spacetime_splines+geo_context"

def coords(homes):
    lat = pd.to_numeric(homes.latitude, errors="coerce").to_numpy(dtype=float)
    lon = pd.to_numeric(homes.longitude, errors="coerce").to_numpy(dtype=float)
    dates = pd.to_datetime(homes.date_listed, errors="coerce")
    t = dates.dt.to_period("M").astype("int64").to_numpy(dtype=float)
    if (not np.isfinite(np.column_stack((lat, lon, t))).all() or
        (np.abs(lat) > 90).any() or (np.abs(lon) > 180).any()):
        raise ValueError("Valid lat/lon and listing date required")
    # Approximate local projection: 1 degree longitude at Toronto latitude,
    # 1 degree latitude, and months, in km-equivalent scaled units.
    return np.column_stack(((lon + 79.4) * 80.0, (lat - 43.7) * 111.0,
                            (t - pd.Period("2020-01", freq="M").ordinal) * 2.5))

"""Public weather feed adapters (METAR/CLI/GOES-19/NEXRAD). NYS Mesonet optional/blocked."""

from __future__ import annotations

# Buckets verified 2026-09-16 against registry.opendata.aws:
GOES_EAST_BUCKET = "noaa-goes19"  # GOES-East operational since 2025-04-04
NEXRAD_L2_BUCKET = "unidata-nexrad-level2"  # NOT deprecated noaa-nexrad-level2
NEXRAD_L3_BUCKET = "unidata-nexrad-level3"

# Central Park target + neighbor for wind-direction features
NYC_LAT = 40.77898
NYC_LON = -73.96925
LGA_LAT = 40.7772
LGA_LON = -73.8726
# Closest WSR-88D to NYC metro
RADAR_SITE = "KOKX"  # Upton, NY — Level II key prefix
RADAR_L3_PREFIX = "OKX"  # Level III product ID prefix

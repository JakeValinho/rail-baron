"""A rough outline of the country to draw behind the geographic map.

The board is not a true map: it is stretched and squeezed to make room for
the track. So the outlines below are given in latitude and longitude and then
bent to fit, using the cities as pins. Each city has a real position and a
board position; a smooth warp (a thin-plate spline) carries one onto the
other, and the coastline goes along for the ride.

The outlines are hand-drawn and coarse. They are scenery, not data.
"""

from __future__ import annotations

import numpy as np

from .board import Board
from .layout import geographic_layout

# (latitude, longitude) of every city on the board.
CITY_LAT_LON = {
    "New York": (40.71, -74.01), "Washington DC": (38.90, -77.04), "Pittsburgh": (40.44, -80.00),
    "Albany": (42.65, -73.75), "Boston": (42.36, -71.06), "Philadelphia": (39.95, -75.17),
    "Buffalo": (42.89, -78.88), "Portland ME": (43.66, -70.26), "Baltimore": (39.29, -76.61),
    "Charlotte": (35.23, -80.84), "Norfolk": (36.85, -76.29), "Chattanooga": (35.05, -85.31),
    "Atlanta": (33.75, -84.39), "Charleston": (32.78, -79.93), "Miami": (25.76, -80.19),
    "Jacksonville": (30.33, -81.66), "Richmond": (37.54, -77.44), "Knoxville": (35.96, -83.92),
    "Tampa": (27.95, -82.46), "Mobile": (30.69, -88.04), "Cleveland": (41.50, -81.69),
    "Cincinnati": (39.10, -84.51), "Chicago": (41.88, -87.63), "Detroit": (42.33, -83.05),
    "Columbus": (39.96, -83.00), "Indianapolis": (39.77, -86.16), "Milwaukee": (43.04, -87.91),
    "St. Louis": (38.63, -90.20), "Memphis": (35.15, -90.05), "Shreveport": (32.53, -93.75),
    "Dallas": (32.78, -96.80), "Little Rock": (34.75, -92.29), "New Orleans": (29.95, -90.07),
    "Birmingham": (33.52, -86.80), "San Antonio": (29.42, -98.49), "Louisville": (38.25, -85.76),
    "Houston": (29.76, -95.37), "Nashville": (36.16, -86.78), "Fort Worth": (32.76, -97.33),
    "Kansas City": (39.10, -94.58), "Oklahoma City": (35.47, -97.52),
    "Minneapolis/St. Paul": (44.98, -93.27), "Denver": (39.74, -104.99), "Des Moines": (41.59, -93.62),
    "Omaha": (41.26, -95.93), "Pueblo": (38.25, -104.61), "Fargo": (46.88, -96.79),
    "Spokane": (47.66, -117.43), "Salt Lake City": (40.76, -111.89), "Seattle": (47.61, -122.33),
    "Portland OR": (45.52, -122.68), "Rapid City": (44.08, -103.23), "Casper": (42.85, -106.31),
    "Pocatello": (42.87, -112.45), "Billings": (45.78, -108.50), "Butte": (46.00, -112.53),
    "San Diego": (32.72, -117.16), "Los Angeles": (34.05, -118.24),
    "San Francisco/Oakland": (37.77, -122.42), "Reno": (39.53, -119.81), "Sacramento": (38.58, -121.49),
    "Las Vegas": (36.17, -115.14), "Phoenix": (33.45, -112.07), "El Paso": (31.76, -106.49),
    "Tucumcari": (35.17, -103.72),
}

# Outlines as (latitude, longitude). Where a border runs through a lake or
# behind another shape it is drawn loosely, because it is covered up.
USA = [
    (48.38, -124.72), (48.15, -123.70), (48.12, -122.75), (49.00, -122.76),            # Puget Sound
    (49.00, -95.15), (49.38, -95.15), (48.60, -93.40), (48.00, -89.60),                # 49th parallel
    (47.80, -87.50), (46.90, -84.90), (46.50, -84.35), (46.00, -83.60), (45.30, -82.50),   # through the lakes
    (43.60, -82.15), (43.00, -82.42), (42.30, -83.10), (41.70, -82.70), (42.30, -80.50),
    (42.80, -79.00), (43.26, -79.07), (43.60, -78.70), (43.65, -76.80), (44.10, -76.35),
    (44.70, -75.50), (45.00, -74.75), (45.00, -71.50), (45.30, -70.90), (45.90, -70.30),   # St. Lawrence, Maine
    (46.40, -70.05), (47.45, -69.20), (47.35, -68.30), (47.07, -67.80), (45.90, -67.80),
    (45.60, -67.45), (45.15, -67.15), (44.80, -66.95),
    (44.40, -68.00), (44.00, -69.10), (43.65, -70.20), (43.10, -70.65), (42.65, -70.60),   # New England
    (42.35, -70.95), (41.95, -70.60), (41.75, -70.00), (42.07, -70.20), (41.67, -69.95),
    (41.55, -70.60), (41.50, -71.10), (41.35, -71.60), (41.07, -71.86), (40.60, -73.50),
    (40.45, -74.00), (39.40, -74.40), (38.93, -74.90), (39.40, -75.50), (38.80, -75.10),   # mid-Atlantic
    (38.00, -75.30), (37.10, -75.95), (38.50, -76.30), (39.30, -76.30), (38.30, -76.50),
    (37.00, -76.30), (36.90, -76.00),
    (35.25, -75.50), (34.60, -76.60), (33.85, -78.00), (33.20, -79.20), (32.75, -79.85),   # the Carolinas
    (32.00, -80.85), (30.70, -81.45), (30.30, -81.40), (28.40, -80.60), (26.90, -80.05),   # Florida
    (25.75, -80.12), (25.15, -80.60), (25.15, -81.10), (25.90, -81.70), (26.50, -82.10),
    (27.60, -82.75), (28.20, -82.80), (29.10, -83.00), (29.90, -83.70), (30.05, -84.30),
    (29.65, -85.30), (30.30, -86.50), (30.30, -87.50), (30.25, -88.00), (30.35, -88.60),   # Gulf coast
    (30.30, -89.30), (30.15, -89.60), (30.00, -89.30), (29.20, -89.20), (29.10, -90.20),
    (29.20, -91.20), (29.55, -92.20), (29.75, -93.30), (29.70, -93.85), (29.30, -94.80),
    (28.60, -96.00), (27.80, -97.10), (26.00, -97.15),
    (25.90, -97.50), (26.40, -99.10), (27.50, -99.50), (28.70, -100.50), (29.45, -101.30),  # Rio Grande
    (29.80, -102.40), (29.00, -103.10), (29.55, -104.40), (30.60, -105.00), (31.75, -106.50),
    (31.78, -108.20), (31.33, -108.20), (31.33, -111.07), (32.50, -114.80), (32.72, -114.72),
    (32.53, -117.12),
    (32.70, -117.25), (33.20, -117.40), (33.60, -117.90), (33.75, -118.40), (34.00, -118.80),  # Pacific
    (34.40, -119.70), (34.45, -120.47), (35.20, -120.90), (36.30, -121.90), (36.60, -121.95),
    (36.95, -122.05), (37.50, -122.50), (37.80, -122.52), (38.00, -123.00), (38.95, -123.73),
    (40.44, -124.40), (41.75, -124.20), (42.84, -124.56), (44.60, -124.07), (46.20, -123.98),
    (47.00, -124.17),
]

CANADA = [
    (49.00, -123.20), (50.50, -127.50), (54.50, -130.50), (56.00, -131.00),
    (56.00, -88.00), (54.00, -82.30), (52.90, -82.20), (51.30, -80.50), (51.40, -79.00),   # James Bay
    (54.00, -79.30), (56.00, -77.00), (56.00, -60.00), (54.00, -57.50), (52.00, -55.70),
    (51.40, -57.20), (50.20, -60.00), (50.20, -66.40), (49.30, -68.10), (47.00, -70.60),   # St. Lawrence
    (48.50, -68.50), (49.20, -64.40), (48.00, -65.50), (47.00, -64.90), (46.20, -64.50),
    (45.90, -63.50), (46.90, -60.40), (45.30, -61.00), (44.60, -63.50), (43.45, -65.60),   # Nova Scotia
    (44.30, -66.20), (45.30, -64.50), (45.75, -64.60), (45.25, -66.05), (45.10, -67.10),
    (44.80, -66.95),
    (45.50, -69.50), (44.50, -74.00), (42.50, -78.00), (41.50, -84.00), (45.00, -93.00),   # closed off behind the USA
    (48.00, -110.00), (48.50, -122.00),
]

VANCOUVER_ISLAND = [
    (48.35, -123.50), (48.55, -124.60), (50.75, -128.40), (50.90, -127.60), (49.70, -124.90), (48.90, -123.60),
]

MEXICO = [
    (32.53, -117.12), (30.40, -115.95), (28.00, -114.10), (26.70, -113.50), (24.80, -112.20),   # Baja
    (22.90, -110.00), (23.40, -109.40), (24.20, -110.30), (26.00, -111.30), (28.00, -112.80),
    (30.00, -114.60), (31.30, -114.90), (31.75, -114.75),
    (31.30, -113.50), (29.00, -112.20), (27.90, -110.90), (25.60, -109.10), (23.20, -106.45),   # the mainland
    (21.50, -105.30), (20.50, -105.50), (19.00, -104.50), (18.50, -95.50),
    (19.20, -96.10), (20.50, -97.20), (21.30, -97.40), (22.20, -97.80), (24.00, -97.70),        # Gulf coast
    (25.90, -97.15),
    (30.00, -100.00), (33.00, -108.00), (34.00, -116.00),                                       # closed off behind the USA
]

LAKES = {
    "Superior": [
        (46.75, -92.10), (47.30, -91.30), (48.00, -89.60), (48.40, -89.20), (48.80, -88.20), (48.75, -86.50),
        (47.95, -84.90), (46.90, -84.50), (46.50, -84.60), (46.45, -85.00), (46.68, -85.97), (46.45, -86.65),
        (46.55, -87.40), (46.85, -87.70), (47.45, -87.80), (47.20, -88.60), (46.87, -89.30), (46.60, -90.40),
        (46.95, -90.75), (46.70, -91.60),
    ],
    "Michigan": [
        (45.85, -84.75), (45.95, -85.60), (45.90, -86.00), (45.70, -86.90), (45.10, -87.55), (44.55, -87.95),
        (45.25, -87.00), (44.10, -87.55), (43.04, -87.80), (42.30, -87.75), (41.88, -87.52), (41.62, -87.25),
        (41.75, -86.80), (42.40, -86.30), (43.20, -86.30), (44.00, -86.50), (44.75, -86.10), (45.20, -85.50),
        (45.40, -85.00), (45.75, -84.75),
    ],
    "Huron": [
        (45.85, -84.70), (46.00, -84.00), (46.30, -83.60), (46.20, -82.50), (46.00, -81.50), (45.90, -80.70),
        (45.00, -79.90), (44.50, -80.10), (44.60, -80.95), (45.25, -81.60), (44.50, -81.40), (43.75, -81.75),
        (43.00, -82.42), (43.85, -82.65), (44.05, -83.00), (43.65, -83.85), (44.30, -83.30), (45.05, -83.40),
        (45.65, -84.45),
    ],
    "Erie": [
        (41.70, -83.45), (42.05, -83.15), (42.00, -82.50), (42.25, -81.90), (42.65, -81.00), (42.55, -80.10),
        (42.85, -79.60), (42.86, -78.98), (42.50, -79.30), (42.15, -80.10), (41.78, -81.30), (41.58, -81.70),
        (41.42, -82.60),
    ],
    "Ontario": [
        (43.25, -79.80), (43.65, -79.40), (43.90, -78.20), (44.00, -77.00), (44.20, -76.50), (43.95, -76.20),
        (43.50, -76.40), (43.30, -77.60), (43.35, -78.70), (43.25, -79.07),
    ],
}

SMOOTHING = 2.0         # how far the warp may miss a city, traded against how much it bends
STEP = 0.4              # outlines are cut into pieces no longer than this, in degrees, so they bend smoothly


def _flat(lat_lon) -> np.ndarray:
    """Latitude and longitude as flat x, y, with east-west shrunk as it is at mid-country."""
    a = np.asarray(lat_lon, dtype=float)
    return np.column_stack([a[:, 1] * np.cos(np.radians(38.0)), a[:, 0]])


def _kernel(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    r2 = ((a[:, None, :] - b[None, :, :]) ** 2).sum(axis=2)
    return np.where(r2 > 0, 0.5 * r2 * np.log(np.maximum(r2, 1e-12)), 0.0)


def fit_warp(board: Board):
    """A function from (latitude, longitude) rows to positions on the geographic map."""
    geo = geographic_layout(board)
    names = [c for c in board.cities if c in CITY_LAT_LON]
    src = _flat([CITY_LAT_LON[c] for c in names])
    dst = np.array([geo[board.cities[c].node] for c in names])
    n = len(names)
    # Thin-plate spline: a flat fit plus one smooth bump per city.
    top = np.hstack([_kernel(src, src) + SMOOTHING * np.eye(n), np.ones((n, 1)), src])
    bottom = np.hstack([np.ones((1, n)), np.zeros((1, 3))]), np.hstack([src.T, np.zeros((2, 3))])
    system = np.vstack([top, *bottom])
    weights = np.linalg.solve(system, np.vstack([dst, np.zeros((3, 2))]))

    def warp(lat_lon) -> np.ndarray:
        p = _flat(lat_lon)
        return np.hstack([_kernel(p, src), np.ones((len(p), 1)), p]) @ weights

    return warp


def _densify(outline) -> np.ndarray:
    pts = np.asarray(outline, dtype=float)
    out = []
    for a, b in zip(pts, np.roll(pts, -1, axis=0)):
        pieces = max(1, int(np.ceil(np.abs(b - a).max() / STEP)))
        out.extend(a + (b - a) * k / pieces for k in range(pieces))
    return np.array(out)


def land_shapes(board: Board) -> list[dict]:
    """Shapes to draw behind the map, back to front, in geographic map units.

    Each is {"kind": "neighbour" | "land" | "lake", "name", "points": [[x, y], ...]}.
    The result is kept on the board, so asking twice costs nothing.
    """
    cached = getattr(board, "_land_shapes", None)
    if cached is not None:
        return cached
    warp = fit_warp(board)
    outlines = [("neighbour", "Canada", CANADA), ("neighbour", "Vancouver Island", VANCOUVER_ISLAND),
                ("neighbour", "Mexico", MEXICO), ("land", "United States", USA)]
    outlines += [("lake", name, shore) for name, shore in LAKES.items()]
    shapes = [{"kind": kind, "name": name, "points": np.round(warp(_densify(outline)), 1).tolist()}
              for kind, name, outline in outlines]
    board._land_shapes = shapes
    return shapes

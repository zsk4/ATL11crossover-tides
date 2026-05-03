# Zachary Katz
# zachary_katz@mines.edu
# 05 August 2025

"""
Download ATL 11 granule if necessary
Extract example with multiple crossing tracks

NOTE ProcessPoolExecutor calls the whole .py file again and thus needs some parts (variables used in the paralle process)
to be rerun each time, but some parts (the parallel process call) to definitley not be run each time.
Results in weird ordering in file, to fix and make more readable in future version

v0.1 6 August 2025
    Initial parallelization of project
    Wait to refactor until new ATL11 version comes out and I need to refactor anyways
"""

# Imports
import earthaccess
from pathlib import Path
import h5py
from astropy.time import Time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shapely
from shapely.geometry import Point, Polygon, shape
import shapefile
import cartopy.crs as ccrs
from pyproj import CRS, Transformer
import re
import scipy
from itertools import combinations
import matplotlib.ticker as ticker
import os

from concurrent.futures import ProcessPoolExecutor
from scipy.optimize import least_squares
import xarray as xr
from numpy.polynomial import Polynomial
from utide import solve

# WORKER
# User-defined path
data_path = f'/home/m10921061/scratch/data/ATL11_Crossovers_v006'
# Get new RINGS Gl
rings_iceshelves_path = '/home/m10921061/scratch/background/RINGS_2022/RINGS_iceshelves_2022.shp'
auth = earthaccess.login(strategy='netrc')

# Earthaccess unable to handle crossing 180 deg so download in two chunks
short_name = 'ATL11'
bbox_west = (-180,-86, -140,-75)
bbox_east = (155,-86,180, -75)

st = '2021-01-01'
ed = '2025-06-01'
# Download ATL11 if necessary
results = earthaccess.search_data(short_name = 'ATL11', 
bounding_box = bbox_east,
temporal = (st,ed),
version='006',)
print(len(results))
earthaccess.download(results, data_path, provider='POCLOUD')

results = earthaccess.search_data(short_name = 'ATL11', 
bounding_box = bbox_west,
temporal = (st,ed),
version='006',)
print(len(results))
earthaccess.download(results, data_path, provider='POCLOUD')

files = list(Path(data_path).glob('*.h5'))

def plot_shapefile(
    records: list[shapefile._Record],
    shapes: list[shapefile.Shape],
    ax ,
    colors: list[str],
    transform,
    fill: bool = False,
) -> None:
    """
    Plots the given records and shapes on axis ax.

    Parameters
    ----------
    records : list[shapefile._Record]
        Shapely record containing shape classification
    shapes : list[shapefile.Shape]
        Shapely shape points
    ax : cartopy.mpl.geoaxes.GeoAxes
        Axes to polot on
    colors : list[str]
    [Grounded ice color, Ice Shelf color]; Must be length 2
    """
    for record, shape in zip(records, shapes):
        classification = record[field_names.index("Id_text")]
        points = shape.points
        parts = list(shape.parts)
        parts.append(
            len(points)
        )  # Append the end index of the last part of the shapefile
        for i in range(len(parts) - 1):
            part = points[parts[i] : parts[i + 1]]
            if (
                classification == "Isolated island"
                or classification == "Ice rise or connected island"
                or classification == "Grounded ice or land"
            ):
                if fill:
                    ax.fill(*zip(*part), color=colors[0], zorder=2,transform=transform)
                else:
                    ax.plot(*zip(*part), color=colors[0], linewidth=1.5, zorder=2,transform=transform)
            elif classification == "Ice shelf":
                if fill:
                    ax.fill(*zip(*part), color=colors[1], zorder=2,transform=transform)
                else:
                    ax.plot(*zip(*part), color=colors[1], linewidth=1.5, zorder=2,transform=transform)
            else:
                print(f"Unknown classification: {classification}")

# Load ross polygon
ice_shelf_polygons = []
grounded_polygons = []
sf = shapefile.Reader(rings_iceshelves_path)
fields = sf.fields[1:]  # Skip deletion flag
field_names = [field[0] for field in fields]

records = sf.records()
shapes = sf.shapes()

for record, shape in zip(records, shapes):
    rec_dict = dict(zip(field_names, record))
    classification = rec_dict["Type"]
    points = shape.points
    parts = list(shape.parts)
    parts.append(len(points))  # Append the end index of the last part
    for i in range(len(parts) - 1):
        part = points[parts[i] : parts[i + 1]]
        polygon = shapely.Polygon(part)
        if (
            classification == "Isolated island"
            or classification == "Ice rise or connected island"
            or classification == "Grounded ice or land"
        ):
            if polygon.is_valid:
                grounded_polygons.append(polygon)
        elif classification == "Ice shelf":
            if polygon.is_valid:
                ice_shelf_polygons.append(polygon)
        else:
            print(f"Unknown classification: {classification}")

# Filter records and shapes to bbox
bbox = [-200000,-1800000,0,-800000] #Ross
filtered_records = []
filtered_shapes = []
for record, _shape in zip(records, shapes):
    shape_bbox = _shape.bbox
    filtered_records.append(record)
    filtered_shapes.append(_shape)

shapefiles = [shapely.geometry.shape(s.__geo_interface__) for s in filtered_shapes]

def ll2xy(lon: list[float], lat: list[float]) -> tuple[list[float], list[float]]:
    """
    Transform coordinates from input geodetic coordinates (lon, lat)
    to output Antarctic Polar Stereographic coordinates (x, y).
    Can also take single floats.

    Parameters
    ----------
    lon: list[float]
        Geodetic longitude in EPSG:4326
    lat: list[float]
        Geodetic latitude in EPSG:4326

    Returns
    -------
    x: list[float]
    Antarctic Polar Stereographic (EPSG:3031) x
    y: list[float]
    Antarctic Polar Stereographic (EPSG:3031) y
    """

    crs_ll = CRS("EPSG:4326")
    crs_xy = CRS("EPSG:3031")
    ll_to_xy = Transformer.from_crs(crs_ll, crs_xy, always_xy=True)
    x, y = ll_to_xy.transform(lon, lat)
    return x, y

from shapely.geometry import Point, Polygon, shape
ross = shape(filtered_shapes[0].__geo_interface__)

# Tidal constituents [hr]
HR_IN_DAY = 24
SEC_IN_HR = 3600

M2 = 12.4206012
S2 = 12
N2 = 12.65834751
K2 = 11.96723606

K1 = 23.9344721
O1 = 25.81933871
P1 = 24.06588766
Q1 = 26.868350

constituents = {
    "M2": M2,
    "S2": S2,
    "N2": N2,
    "K2": K2,
    "K1": K1,
    "O1": O1,
    "P1": P1,
    "Q1": Q1,
}
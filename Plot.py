# Radius Crossovers

# Zachary Katz
# zachary_katz@wendian.mines.edu

# 03 February 2025
# Intermediate Crossover Stage 
# Stationary ATL11 tracks, append all crossovers from different tracks within a given Radius

# Imports
print("IMPORTING MODULES")

import earthaccess
from pathlib import Path
import h5py
from astropy.time import Time
import matplotlib
matplotlib.use("Agg")
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
from shapely.geometry import Point, Polygon, shape
from scipy.optimize import least_squares
import xarray as xr
from numpy.polynomial import Polynomial
from utide import solve
from scipy.spatial import cKDTree

print ("IMPORTS COMPLETE")

### CHECK AND EXIT IF NOT ON NODE ###
if "SLURM_JOB_ID" not in os.environ:
    print("Not running using Slurm. Exiting.")
    sys.exit(1)
else:
    print(f"Running Slurm Job ID: {os.environ['SLURM_JOB_ID']}")


print("SETUP")
# WORKER
# User-defined path
########## SET DATA PATH
data_path = f'/home/m10921061/scratch/output/ATL11_Crossover_Tide_FULL/ross_ATL11_tides.nc'
# Get new RINGS Gl
########## SET RINGS GL PATH
rings_iceshelves_path = '/home/m10921061/scratch/background/RINGS_2022/RINGS_iceshelves_2022.shp'
########## SET OUTPUT DIR NAME
output_dir = f'./OceanSciencesPosterPlots'
os.makedirs(output_dir, exist_ok=True)


# Earthaccess unable to handle crossing 180 deg so download in two chunks
short_name = 'ATL11'

########### SET BOUNDING BOXES FOR ANTARCTIC REGIONS
bbox_amery = (60.31,-76.08, 77.00,-67.27)

def xy2ll(x: list[float], y: list[float]) -> tuple[list[float], list[float]]:
    """
    Transform coordinates from Antarctic Polar Stereographic coordinates (x, y)
    to output geodetic coordinates (lon, lat).
    Can also take single floats.

    Parameters
    ----------
    x: list[float]
       Antarctic Polar Stereographic (EPSG:3031) x
    y: list[float]
       Antarctic Polar Stereographic (EPSG:3031) y

    Returns
    -------
    lon: list[float]
         Geodetic longitude in EPSG:4326
    lat: list[float]
         Geodetic latitude in EPSG:4326
    """
    crs_ll = CRS("EPSG:4326")
    crs_xy = CRS("EPSG:3031")
    xy_to_ll = Transformer.from_crs(crs_xy, crs_ll, always_xy=True)
    lon, lat = xy_to_ll.transform(x, y)
    return lon, lat

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
        classification = record[field_names.index("Type")]
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

# Load ROSS polygon
print("Loading RINGS Shapefile")
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

print("FILTERING")
# Filter records and shapes to bbox
bbox = [-156095.6916323168,-604782.339773115,-156095.6816323168,-604782.329773115] #Ross (GZ05)
#bbox = [2100000, 700000, 2100010, 700010]  # Amery
filtered_records = []
filtered_shapes = []
for record, _shape in zip(records, shapes):
    shape_bbox = _shape.bbox
    if not (shape_bbox[2] < bbox[0] or shape_bbox[0] > bbox[2] or
            shape_bbox[3] < bbox[1] or shape_bbox[1] > bbox[3]):
        filtered_records.append(record)
        filtered_shapes.append(_shape)

shapefiles = [shapely.geometry.shape(s.__geo_interface__) for s in filtered_shapes]

from shapely.geometry import Point, Polygon, shape # BAD, Overwrites shape var
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

def to_scalar(val):
    return np.array([np.asarray(v).item() for v in val])

def pad_numeric(val, fill=np.nan):
    maxlen = max(len(v) for v in val)
    out = np.full((len(val), maxlen), fill)
    for i, v in enumerate(val):
        out[i, :len(v)] = v
    return out

def pad_datetime(val):
    maxlen = max(len(v) for v in val)
    out = np.full((len(val), maxlen), np.datetime64('NaT', 'ns'))

    for i, v in enumerate(val):
        out[i, :len(v)] = np.array(v, dtype='datetime64[ns]')

    return out


"""
out_ds = xr.Dataset(
    data_vars={
        'x':    (['point'], x),
        'y':    (['point'], y),
        'lat':  (['point'], lat),
        'lon':  (['point'], lon),
        'res':  (['point'], res),

        'amp':   (['point','constituent'], amp),
        'phase': (['point','constituent'], phase),

        'amp_pyTMD': (['point','constituent'], tide_analyses['amp_pyTMD']),
        'ph_pyTMD':  (['point','constituent'], tide_analyses['ph_pyTMD']),

        't1':      (['point','time'], t1),
        't2':      (['point','time'], t2),
        'dt_days': (['point','time'], dt_days),
        'h1':      (['point','time'], h1),
        'h2':      (['point','time'], h2),
        'dh':      (['point','time'], dh),

        'datetime':            (['point','time_xover'], datetime),
        'xovers':              (['point','time_xover'], xovers),
        'xovers_undetrended':  (['point','time_xover'], xovers_u),
    },
    coords={
        'point': np.arange(len(x)),
        'constituent': np.arange(amp.shape[1]),
        'time': np.arange(t1.shape[1]),
        'time_xover': np.arange(datetime.shape[1]),
    }
)
"""
# LOAD DATASET
print("Loading Dataset")
data = f'/home/m10921061/scratch/output/OceanSciencesPoster/ross_ATL11_tides.nc'

ds = xr.open_dataset(data)
print(ds)

ps71_projection = ccrs.Stereographic(central_latitude=-90, central_longitude=0, true_scale_latitude=-71)
fig, ax = plt.subplots(
    figsize=(10, 10),
    subplot_kw={'projection': ps71_projection},
)
const = 'M2'
i = 1
n=-1
plot_shapefile(filtered_records, filtered_shapes, ax, ["lightgray", "lightgray"], ps71_projection, fill=True)

xs = np.array(ds['x'][:n])
ys = np.array(ds['y'][:n])
amps = np.array([a[i] for a in ds['amp'][:n]])
amps_pyTMD = np.array([a[i] for a in ds['amp_pyTMD'][:n]])
amp_diff = amps - amps_pyTMD
col = ax.scatter(xs, ys, s=0.1, zorder=10, c=amps, cmap='viridis',vmin=hmin,vmax=hmax)
cbar = fig.colorbar(col)

KM_SCALE = 1e3
ax.xaxis.set_visible(True)
ax.yaxis.set_visible(True)
ticks_x = ticker.FuncFormatter(lambda x, pos: "{0:g}".format(x / KM_SCALE))
ax.xaxis.set_major_formatter(ticks_x)
ticks_y = ticker.FuncFormatter(lambda x, pos: "{0:g}".format(x / KM_SCALE))
ax.yaxis.set_major_formatter(ticks_y)
ax.set_xlabel("X (PS71) [km]", size=25)
ax.set_ylabel("Y (PS71) [km]", size=25)
ax.tick_params(labelsize=20)
ax.tick_params(size=4)

cbar.ax.tick_params(labelsize=25)
cbar.set_label(f"{const} Amplitude [m]", fontsize=28, color="black")

fig.savefig(f"{output_dir}/{const}_ha_amp.png", bbox_inches="tight", dpi=300, transparent=True)
plt.close(fig)
"""
# Plot amps
print("Plotting Amplitudes")
for i, const in enumerate(constituents): # Loop over amplitudes
    if const == 'K1' or const == 'O1':
        hmin = 0.3
        hmax = 0.6  
    else:
        hmin = 0
        hmax = 0.3
    ps71_projection = ccrs.Stereographic(central_latitude=-90, central_longitude=0, true_scale_latitude=-71)
    fig, ax = plt.subplots(
        figsize=(10, 10),
        subplot_kw={'projection': ps71_projection},
    )
    plot_shapefile(filtered_records, filtered_shapes, ax, ["lightgray", "lightgray"], ps71_projection, fill=True)
    n = -1

    xs = np.array(ds['x'][:n])
    ys = np.array(ds['y'][:n])
    amps = np.array([a[i] for a in ds['amp'][:n]])
    amps_pyTMD = np.array([a[i] for a in ds['amp_pyTMD'][:n]])
    amp_diff = amps - amps_pyTMD
    col = ax.scatter(xs, ys, s=0.1, zorder=10, c=amps, cmap='viridis',vmin=hmin,vmax=hmax)
    cbar = fig.colorbar(col)

    KM_SCALE = 1e3
    ax.xaxis.set_visible(True)
    ax.yaxis.set_visible(True)
    ticks_x = ticker.FuncFormatter(lambda x, pos: "{0:g}".format(x / KM_SCALE))
    ax.xaxis.set_major_formatter(ticks_x)
    ticks_y = ticker.FuncFormatter(lambda x, pos: "{0:g}".format(x / KM_SCALE))
    ax.yaxis.set_major_formatter(ticks_y)
    ax.set_xlabel("X (PS71) [km]", size=25)
    ax.set_ylabel("Y (PS71) [km]", size=25)
    ax.tick_params(labelsize=20)
    ax.tick_params(size=4)

    cbar.ax.tick_params(labelsize=25)
    cbar.set_label(f"{const} Amplitude [m]", fontsize=28, color="black")
    fig.savefig(f"{output_dir}/{const}_ha_amp.png", bbox_inches="tight", dpi=300, transparent=True)
    plt.close(fig)

# Plot Difference from CATS
print("Plotting Differences from CATS")
hmin = -0.5
hmax = 0.5
for i, const in enumerate(constituents): # Loop over amplitudes
    ps71_projection = ccrs.Stereographic(central_latitude=-90, central_longitude=0, true_scale_latitude=-71)
    fig, ax = plt.subplots(
        figsize=(10, 10),
        subplot_kw={'projection': ps71_projection},
    )
    plot_shapefile(filtered_records, filtered_shapes, ax, ["lightgray", "lightgray"], ps71_projection, fill=True)

    n = -1
    xs = np.array(ds['x'][:n])
    ys = np.array(ds['y'][:n])
    amps = np.array([a[i] for a in ds['amp'][:n]])
    amps_pyTMD = np.array([a[i] for a in ds['amp_pyTMD'][:n]])
    amp_diff = amps - amps_pyTMD
    col = ax.scatter(xs, ys, s=0.1, zorder=10, c=amp_diff, cmap='seismic',vmin=hmin,vmax=hmax)
    cbar = fig.colorbar(col)

    KM_SCALE = 1e3
    ax.xaxis.set_visible(True)
    ax.yaxis.set_visible(True)
    ticks_x = ticker.FuncFormatter(lambda x, pos: "{0:g}".format(x / KM_SCALE))
    ax.xaxis.set_major_formatter(ticks_x)
    ticks_y = ticker.FuncFormatter(lambda x, pos: "{0:g}".format(x / KM_SCALE))
    ax.yaxis.set_major_formatter(ticks_y)
    ax.set_xlabel("X (PS71) [km]", size=25)
    ax.set_ylabel("Y (PS71) [km]", size=25)
    ax.tick_params(labelsize=20)
    ax.tick_params(size=4)

    cbar.ax.tick_params(labelsize=25)
    cbar.set_label(f"Data - Model {const} Amplitude [m]", fontsize=28, color="black")
    fig.savefig(f"{output_dir}/{const}_ha_CATSDiff.png", bbox_inches="tight", dpi=300, transparent=True)
    plt.close(fig)

# Plot Histogram
print("Plotting Histograms of Differences from CATS")

hmin = -0.5
hmax = 0.5
# 2x4 subplots
fig, axs = plt.subplots(4, 2, figsize=(10, 12),sharex=True, sharey=True)
ax_list = axs.flatten()
for i, const in enumerate(constituents): # Loop over amplitudes
    n = -1
    xs = np.array(ds['x'][:n])
    ys = np.array(ds['y'][:n])
    amps = np.array([a[i] for a in ds['amp'][:n]])
    amps_pyTMD = np.array([a[i] for a in ds['amp_pyTMD'][:n]])
    amp_diff = amps - amps_pyTMD
    ax = ax_list[i]
    ax.hist(amp_diff, bins=100, range=(hmin, hmax), color='#21314d')
    ax.set_xlabel("Data - Model [m]", size=25)
    ax.set_ylabel("Counts", size=25)
    ax.tick_params(labelsize=20)
    ax.tick_params(size=4)
    ax.label_outer()
    # Remove right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    fig.tight_layout()

    # Add constitutent as text in upper left corner
    ax.text(0.05, 0.95, const, transform=ax.transAxes, fontsize=28, verticalalignment='top', color='#21314d')

    fig.savefig(f"{output_dir}/histo.png", bbox_inches="tight", dpi=300, transparent=True)
"""
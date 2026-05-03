# Radius Crossovers

# Zachary Katz
# zachary_katz@wendian.mines.edu

# 03 February 2025
# Intermediate Crossover Stage 
# Stationary ATL11 tracks, append all crossovers from different tracks within a given Radius

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
from shapely.geometry import Point, Polygon, shape
from scipy.optimize import least_squares
import xarray as xr
from numpy.polynomial import Polynomial
from utide import solve
from scipy.spatial import cKDTree

### CHECK AND EXIT IF NOT ON NODE ###
if "SLURM_JOB_ID" not in os.environ:
    print("Not running using Slurm. Exiting.")
    sys.exit(1)
else:
    print(f"Running Slurm Job ID: {os.environ['SLURM_JOB_ID']}")

# WORKER
# User-defined path
########## SET DATA PATH
data_path = f'/home/m10921061/scratch/data/ATL11_Crossovers_v006_Amery'
# Get new RINGS Gl
########## SET RINGS GL PATH
rings_iceshelves_path = '/home/m10921061/scratch/background/RINGS_2022/RINGS_iceshelves_2022.shp'
########## LOGIN TO EARTHDATA
auth = earthaccess.login(strategy='netrc')

########## SET MAX DAYS BETWEEN CROSSOVER PAIRS AND RADIUS TO QUERY
day_xover = 45  # Max days between crossover pairs
########## SET RADIUS TO QUERY
rad_select = 5000 # Meters of radius to query
########## SET OUTPUT DIR NAME
output_dir = f'/home/m10921061/scratch/output/ATL11_Crossover_Tide_{day_xover}dayPairsSpatialAvg{rad_select}meters_v006_Amery'


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

st = '2021-01-01'
ed = '2025-06-01'

########### Download ATL11 if necessary
#results = earthaccess.search_data(short_name = 'ATL11', 
#bounding_box = bbox_east,
#temporal = (st,ed),
#version='006',)
#print(len(results))
#earthaccess.download(results, data_path, provider='POCLOUD')

results = earthaccess.search_data(short_name = 'ATL11', 
bounding_box = bbox_amery,
temporal = (st,ed),
version='006',)
print(len(results))
earthaccess.download(results, data_path, provider='POCLOUD')

files = list(Path(data_path).glob('*.h5'))

# Test Using Smaller Chunks Here!
#######################################
#####################################
###################################
#################################
###############################

chunk_size = 5
#num_chunks = 20
num_chunks = (len(files) + chunk_size - 1) // chunk_size 

###############################
#################################
###################################
#####################################
#######################################
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

# Load AMERY polygon
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
#bbox = [-156095.6916323168,-604782.339773115,-156095.6816323168,-604782.329773115] #Ross (GZ05)
bbox = [2100000, 700000, 2100010, 700010]  # Amery
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

# Each reference track pair is a group
group = ['/pt1/','/pt2/','/pt3/']

def process_chunk(chunk_idx):
    chunk_df = []
    start_idx = chunk_idx * chunk_size
    end_idx = min(start_idx + chunk_size, len(files))
    file_chunk = files[start_idx:end_idx]

    print(f"Processing chunk {chunk_idx+1}/{num_chunks} with {len(file_chunk)} files...")

    # Make array of ATL11 data
    print("ATL11 Data Array")
    # Loop over files
    for file in file_chunk:
        # Loop over pairs
        for i, g in enumerate(group):
            data = {}
            # Load variables into dictionary list
            with h5py.File(file, 'r') as fi:
                data['t_ref'] = fi['/ancillary_data/atlas_sdp_gps_epoch'][:] # ICESat-2 reference epoch
                data['filename'] = file
                data['group'] = g

                # Crossing track data
                data['x_q_flag'] = fi[g+'crossing_track_data/atl06_quality_summary'][:] #0 likely no problem; 1 problems #See table 4-4 in IceSat2 ATL11 ATBD
                data['x_lat'] = fi[g+'crossing_track_data/latitude'][:] # Latitude [degrees]
                data['x_lon'] = fi[g+'crossing_track_data/longitude'][:] # Longitude [degrees]
                data['x_h_corr'] = fi[g+'crossing_track_data/h_corr'][:] # Mean corrected height [m]
                data['x_t_dt'] = fi[g+'crossing_track_data/delta_time'][:] # GPS seconds since reference epoch
                data['x_cycle_number'] = fi[g+'crossing_track_data/cycle_number'][:] # Cycle
                data['x_rgt'] = fi[g+'crossing_track_data/rgt'][:] # Reference ground track
                data['x_spot_crossing'] = fi[g+'crossing_track_data/spot_crossing'][:] # Spot  number
                data['x_ref_pt'] = fi[g+'crossing_track_data/ref_pt'][:] # Reference point
                data['x_x'], data['x_y'] = ll2xy(data['x_lon'],data['x_lat'])
                data["x_dac"] = fi[g + "crossing_track_data/dac"][:]

                time_temp = data['t_ref'] + data['x_t_dt']
                data['x_time'] = Time(time_temp, format='gps').iso # Convert to readable time
                
                # Main track data
                data['lat'] = fi[g+'/latitude'][:] # Latitude [degrees]
                data['lon'] = fi[g+'/longitude'][:] # Longitude [degrees]
                data['h_corr'] = fi[g+'/h_corr'][:] # Mean corrected height [m]
                data['t_dt'] = fi[g+'/delta_time'][:] # GPS seconds since reference epoch
                data['q_flag'] = fi[g+'/quality_summary'][:] #0 likely no problem; 1 problems #See table 4-4 in IceSat2 ATL11 ATBD
                data['cycle_number'] = fi[g+'/cycle_number'][:] # Cycle
                data['ref_pt'] = fi[g+'/ref_pt'][:] # Reference point
                time_temp = data['t_ref'] + data['t_dt']
                data['time'] = Time(time_temp, format='gps').iso # Convert to readable time
                data["dac"] = fi[g + "/cycle_stats/dac"][:]


                data['x'], data['y'] = ll2xy(data['lon'],data['lat'])
                

        print("Downscaling to Ross Ice Shelf")
        # Downscale data so only in ross polygon
        inside_mask = []
        x_inside_mask = []
        for xi, yi in zip(data['x'], data['y']):
            ######### TEST BOX HERE AND IN NEXT FOR
            # small_box [120579.55227226857, -950647.6134126243, 48837.606640782804, -1036344.6830128166]
            point = Point(xi, yi)
            inside_mask.append(ross.contains(point))
        data['inside_mask'] = inside_mask

        for xi, yi in zip(data['x_x'], data['x_y']):
            point = Point(xi, yi)
            x_inside_mask.append(ross.contains(point))

        data['x_inside_mask'] = x_inside_mask

        print("Creating Crossover Table")
        # Loop over each track and create crossover table
        #print(np.sum(data['inside_mask']), "points inside ross")
        for ref_pt in data['ref_pt'][data['inside_mask']]:
            
            mskx = (data['x_ref_pt'] == ref_pt) & (data['x_t_dt'] < 2e300)
            msk = (data['ref_pt'] == ref_pt)
            mskd_lat = data['lat'][msk]
            mskd_lon = data['lon'][msk]
            mskd_xlat = data['x_lat'][mskx]
            mskd_xlon = data['x_lon'][mskx]
            mskd_xt = data['x_time'][mskx]
            mskd_xcycle = data['x_cycle_number'][mskx]
            mskd_xrgt = data['x_rgt'][mskx]
            mskd_xdac = data["x_dac"][mskx]

            mskd_t = data['time'][msk]
            mskd_dac = data["dac"][msk]

            mskd_tdt = data['t_dt'][msk]
            msk_t = mskd_tdt < 2e300
            mskd_t = mskd_t[msk_t]
            mskd_dac = mskd_dac[msk_t]

            mskd_h = data['h_corr'][msk]
            mskd_h = mskd_h[msk_t]
            mskd_xh = data['x_h_corr'][mskx]
            
            mskd_q = data['q_flag'][msk]
            mskd_q = mskd_q[msk_t]
            mskd_xq = data['x_q_flag'][mskx]

            msk_t = msk_t.flatten()
            mskd_cycle = data['cycle_number'][msk_t]

            mskd_lon = np.ones(len(mskd_t)) * mskd_lon
            mskd_lat = np.ones(len(mskd_t)) * mskd_lat

            mskd_xx, mskd_xy = ll2xy(mskd_xlon,mskd_xlat)
            mskd_x, mskd_y = ll2xy(mskd_lon,mskd_lat)

            match = re.search(r'ATL11_(\d{4})', str(data['filename']))
            mskd_rgt = np.ones(len(mskd_t),dtype=int) * int(match.group(1))
            
            #print("Masked_xlat length:", len(mskd_xlat))
            if len(mskd_xlat) > 5:
                df1 = pd.DataFrame(
                {
                    'lat': mskd_xlat,
                    'lon': mskd_xlon,
                    'x': mskd_xx,
                    'y': mskd_xy,
                    'time': mskd_xt,
                    'height': mskd_xh,
                    'q_flag': mskd_xq,
                    'cycle_number': mskd_xcycle,
                    'rgt': mskd_xrgt,
                    "dac": mskd_xdac,
                }
                )
                df2 = pd.DataFrame(
                    {
                        'lat': mskd_lat,
                        'lon': mskd_lon,
                        'x': mskd_x,
                        'y': mskd_y,
                        'time': mskd_t,
                        'height': mskd_h,
                        'q_flag': mskd_q,
                        'cycle_number': mskd_cycle,
                        'rgt': mskd_rgt,
                        "dac": mskd_dac,
                        
                    }
                )
                df = pd.concat([df1,df2], ignore_index=True)
                df = df.sort_values(by=['time']).reset_index(drop=True)

                df["datetime"] = df["time"]
                t = pd.to_datetime(df["time"], format="%Y-%m-%d %H:%M:%S.%f", errors="raise")

                df["time"] = (t - t.iloc[0]).dt.total_seconds() / SEC_IN_HR
                chunk_df.append(df)
    
    return chunk_df

def tide_model(t, periods, parameters):
    """
    Generate the tide from synthetic data.

    Parameters
    ----------
    t : List
        List of sampling times in hours
    periods : dict
        Dictionary of tidal constituent periods in hours
    parameters : list
        List in form [A1, A2, ... , phi_1, phi_2, ...]
        where A is the amplitude in m and phi is the phase shift in radians

    Returns
    -------
    modeled : list
        Tides at time t as estimated by the model
    """

    assert len(parameters) == 2 * len(
        periods
    ), "Parameters must be twice the number of periods"

    model = np.zeros_like(t, dtype=float)
    n = len(periods)
    for i in range(n):
        A = parameters[i]
        phi = parameters[i + n]
        model += A * np.cos(2 * np.pi * t / periods[i] - phi)
    return model

def residuals(parameters, t, data, periods):
    """
    Residual function for scipy's least_squares
    Scipy does the squaring for us.
    """
    return tide_model(t, periods, parameters) - data

periods = [constituents[c] for c in constituents]


def tide_fitting(df, initial_guess):
    mskd_q = df['q_flag'] == 0
    mskd_h = df['height'].values[mskd_q]
    mskd_dac = df['dac'].values[mskd_q]
    t = df["time"].values[mskd_q]
    #y = scipy.signal.detrend(mskd_h - mskd_dac, type='linear') # Detrend assumes linear spacing -- whoops!
    h_dac_corrected = mskd_h - mskd_dac
    poly, stats = Polynomial.fit(t, h_dac_corrected, 1,full=True) # Linear detrending
    fitted_y = poly(h_dac_corrected)
    y_detrended = h_dac_corrected #- fitted_y # We'reversed not detrending if we're using differences...

    residual = stats[0]

    #initial_guess = [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    #fit = least_squares(residuals, initial_guess, args=(t, y_detrended, periods))
    #for n in range(8):
    #    if fit.x[n] < 0:
    #        fit.x[n + 8] += np.pi
    print("STARTING LEAST SQUARES")
    #print(t)
    #print(y_detrended)
    #print(df['lat'][0])
    x, y = ll2xy(df['lon'][0],df['lat'][0])

    if len(t) > 16:
        # Make pairs of points and complete best fit instead of using utide!
        #datetimes = pd.to_datetime(df["datetime"]).values[mskd_q]
        datetimes = pd.to_datetime(
            df["datetime"].str.strip(),
            format="%Y-%m-%d %H:%M:%S.%f",
            errors="coerce"
        ).values[mskd_q]
        y_data = np.asarray(y_detrended)
        t_ns = datetimes.astype("datetime64[ns]").astype("int64")
        n = len(datetimes)
        i, j = np.triu_indices(n, k=1)
        t1 = datetimes[i]
        t2 = datetimes[j]
        dt_ns = np.abs(t_ns[j] - t_ns[i])
        h1 = y_data[i]
        h2 = y_data[j]
        delta_h = h2 - h1
        dt_max = np.int64(pd.Timedelta(days=day_xover).value)
        mask = dt_ns < dt_max
        
        t1 = t1[mask]
        t2 = t2[mask]
        h1 = h1[mask]
        h2 = h2[mask]
        delta_h = delta_h[mask]
        dt = dt_ns[mask] / 1e9 / 3600 / 24  # days

        omega = np.array(list(constituents.values()), dtype=float)
        M = len(omega)
        N  = len(delta_h)  
        #print("Number of pairs used in fit:", N, "example dt", np.unique(dt)[:])

        # Print table of t1,t2,dt,h1,h2,delta_h for all
        table = pd.DataFrame({
            't1': t1,
            't2': t2,
            'dt_days': dt,
            'h1': h1,
            'h2': h2,
            'delta_h': delta_h,
        })
        
        #print(table)
        
        if N > 16:
            # Design Matrix
            G = np.zeros((N, 2*M))

            t1_hours = t1.astype("datetime64[ns]").astype(float) / 1e9 / 3600  # seconds → hours
            t2_hours = t2.astype("datetime64[ns]").astype(float) / 1e9 / 3600
            for i, wi in enumerate(omega):
                omega_i = 2 * np.pi / wi  # radians per hour
                Delta = np.exp(1j * omega_i * t2_hours) - np.exp(1j * omega_i * t1_hours)
                G[:, 2*i]   =  2 * np.real(Delta)
                G[:, 2*i+1] = -2 * np.imag(Delta)

            # Least squares
            m, *_ = np.linalg.lstsq(G, delta_h, rcond=None)
            c = m[0::2] + 1j * m[1::2] # Complex coefficients

            a = 2 * np.real(c)
            b = -2 * np.imag(c)

            amps = np.sqrt(a**2 + b**2) # amp
            phases = np.arctan2(b, a) # phase
            #print("Fitted Amplitudes (m):", amps, "Fitted Phases (rad):", phases)
            # Check conditioning
            u, s, vt = np.linalg.svd(G, full_matrices=False)
            #print("Condition number:", s[0] / s[-1])  
            

            #soln = solve(t[:], y_detrended[:], lat=df['lat'][0], method="ols", conf_int="none", constit=['M2','S2','N2','K2','K1','O1','P1','Q1'])
            
            
            #amps = np.array([soln['A'][soln['name'] == key][0] for key in constituents])
            #phases = np.array([soln['g'][soln['name'] == key][0] for key in constituents])

            # combine amplitudes first, then phases
            result = np.concatenate([amps, phases])

            amp = np.abs(result[:8])
            phase = result[8:] % (2 * np.pi) * 180 / np.pi 

            return {
            'x': x,
            'y': y,
            'amp': amp,
            'phase': phase,
            'lat': df['lat'][0],
            'lon': df['lon'][0],
            'res': residual,
            't1': t1,
            't2': t2,
            'dt_days': dt,
            'h1': h1,
            'h2': h2,
            'dh': delta_h,
            'datetime': df["datetime"].values[mskd_q],
            'xovers': y_detrended,
            'xovers_undetrended': h_dac_corrected,
            }
        else:
            return {
            'x': x,
            'y': y,
            'amp': -1000*np.ones(8),
            'phase': -1000*np.ones(8),
            'lat': df['lat'][0],
            'lon': df['lon'][0],
            'res': residual,
            't1': -1000*np.ones(5),
            't2': -1000*np.ones(5),
            'dt_days': -1000*np.ones(5),
            'h1': -1000*np.ones(5),
            'h2': -1000*np.ones(5),
            'dh': -1000*np.ones(5),
            'datetime': df["datetime"].values[mskd_q],
            'xovers': y_detrended,
            'xovers_undetrended': h_dac_corrected,
            }   
    else:
        return {
        'x': x,
        'y': y,
        'amp': -1000*np.ones(8),
        'phase': -1000*np.ones(8),
        'lat': df['lat'][0],
        'lon': df['lon'][0],
        'res': residual,
        't1': -1000*np.ones(5),
        't2': -1000*np.ones(5),
        'dt_days': -1000*np.ones(5),
        'h1': -1000*np.ones(5),
        'h2': -1000*np.ones(5),
        'dh': -1000*np.ones(5),
        'datetime': df["datetime"].values[mskd_q],
        'xovers': y_detrended,
        'xovers_undetrended': h_dac_corrected,
        }

    x, y = ll2xy(df['lon'][0],df['lat'][0])
    return {
    'x': x,
    'y': y,
    'amp': amp,
    'phase': phase,
    'lat': df['lat'][0],
    'lon': df['lon'][0],
    'res': residual,
    't1': t1,
    't2': t2,
    'dt_days': dt,
    'h1': h1,
    'h2': h2,
    'dh': delta_h,
    'xovers': y_detrended,
    'xovers_undetrended': h_dac_corrected,
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

print("Processing Chunks in Parallel")
with ProcessPoolExecutor(max_workers=12) as executor:
    results = list(executor.map(process_chunk, range(num_chunks)))

dfs = [df for chunk in results for df in chunk]

lats = [df['lat'].iloc[0] for df in dfs]
lons = [df['lon'].iloc[0] for df in dfs]

print(f'Numeber of Dataframes made by chunking method: {len(dfs)}')

coords = np.array([[df.iloc[0]["x"], df.iloc[0]["y"]] for df in dfs])

tree = cKDTree(coords)

radius = rad_select # meters
neighbors = tree.query_ball_tree(tree, r=radius)
dists, _ = tree.query(tree.data, k=len(tree.data)) 

# With ChatGPT, Feb 3rd 2026
merged_dfs = []
for i, idx_list in enumerate(neighbors):
    ref_rgt = set(dfs[i]["rgt"].unique())

    # remove self
    idx_list = [j for j in idx_list if j != i]

    # sort neighbors by distance to center (closest first)
    idx_list = sorted(idx_list, key=lambda j: np.linalg.norm(tree.data[i] - tree.data[j]))

    selected = []
    selected_rgts = set()

    for j in idx_list:
        rgt_j = set(dfs[j]["rgt"].unique())

        # must be disjoint from center
        if not rgt_j.isdisjoint(ref_rgt):
            continue

        # must not overlap with already selected neighbors
        if not rgt_j.isdisjoint(selected_rgts):
            continue

        selected.append(dfs[j])
        selected_rgts.update(rgt_j)

    # Always include the center df
    merged = pd.concat([dfs[i]] + selected, ignore_index=True)
    merged_dfs.append(merged)



# Calculate CATS tide at each location for comparison
print("CATS For Comparison")
import pyTMD
tide_dir = "/home/m10921061/scratch/background" # Path to tide model
tide_mod = "CATS2008-v2023"
print("Setting up model")
model = pyTMD.io.model(tide_dir, format="netcdf").elevation(tide_mod)
constituents_pyTMD = pyTMD.io.OTIS.read_constants(
    model.grid_file,
    model.model_file,
    model.projection,
    type=model.type,
    grid=model.format,
)

c = constituents_pyTMD.fields

print("Extracting Tide Harmonics")
amp_pyTMD, ph_pyTMD, D = pyTMD.io.OTIS.interpolate_constants(
    np.atleast_1d(lons),
    np.atleast_1d(lats),
    constituents_pyTMD,
    type=model.type,
    method="spline",
    extrapolate=True,
)

# Downscale to just the 8 constitutents used in my simple Harmonic Analysis

amp_pyTMD = [ap[:8] for ap in amp_pyTMD]
ph_pyTMD = [p[:8] for p in ph_pyTMD]

initial_guesses = []
for amp, ph in zip(amp_pyTMD, ph_pyTMD):
    amp = amp * np.pi / 180
    init = np.concatenate((amp,ph))
    initial_guesses.append(init)


# Try sinusoid fit
print("Fitting a Sinusoid")
with ProcessPoolExecutor() as executor:
    results = list(executor.map(tide_fitting, merged_dfs, initial_guesses))

print(f'Number of results from fitting: {len(results)}')

results = [r for r in results if r is not None]


tide_analyses = {
'xs': [r['x'] for r in results],
'ys': [r['y'] for r in results],
'amplitudes': [r['amp'] for r in results],
'phases': [r['phase'] for r in results],
'lats': [r['lat'] for r in results],
'lons': [r['lon'] for r in results],
'detrend_residuals': [r['res'] for r in results],
'datetime': [r['datetime'] for r in results],
't1': [r['t1'] for r in results],
't2': [r['t2'] for r in results],
'dt_days': [r['dt_days'] for r in results],
'h1': [r['h1'] for r in results],
'h2': [r['h2'] for r in results],
'dh': [r['dh'] for r in results],
'xovers': [r['xovers'] for r in results],
'xovers_undetrended': [r['xovers_undetrended'] for r in results],
}

tide_analyses['amp_pyTMD'] = amp_pyTMD
tide_analyses['ph_pyTMD'] = ph_pyTMD

x   = to_scalar(tide_analyses['xs'])
y   = to_scalar(tide_analyses['ys'])
lat = to_scalar(tide_analyses['lats'])
lon = to_scalar(tide_analyses['lons'])

res = np.array([np.asarray(v).squeeze() for v in tide_analyses['detrend_residuals']])

amp   = np.stack(tide_analyses['amplitudes'])  # (point, 8)
phase = np.stack(tide_analyses['phases'])

t1      = pad_numeric(tide_analyses['t1'])
t2      = pad_numeric(tide_analyses['t2'])
dt_days = pad_numeric(tide_analyses['dt_days'])
h1      = pad_numeric(tide_analyses['h1'])
h2      = pad_numeric(tide_analyses['h2'])
dh      = pad_numeric(tide_analyses['dh'])

datetime = pad_datetime(tide_analyses['datetime'])
xovers   = pad_numeric(tide_analyses['xovers'])
xovers_u = pad_numeric(tide_analyses['xovers_undetrended'])


# Remove points with not enough data before saving and plotting
valid_mask = amp[:,0] > 0
x   = x[valid_mask]
y   = y[valid_mask]
lat = lat[valid_mask]
lon = lon[valid_mask]
res = res[valid_mask]
amp   = amp[valid_mask,:]
phase = phase[valid_mask,:]
t1      = t1[valid_mask,:]
t2      = t2[valid_mask,:]
dt_days = dt_days[valid_mask,:]
h1      = h1[valid_mask,:]
h2      = h2[valid_mask,:]
dh      = dh[valid_mask,:]
datetime = datetime[valid_mask,:]
xovers   = xovers[valid_mask,:]
xovers_u = xovers_u[valid_mask,:]

# Huge full array not needed
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
out_ds = xr.Dataset(
    data_vars={
        'x':    (['point'], x),
        'y':    (['point'], y),
        'lat':  (['point'], lat),
        'lon':  (['point'], lon),
        'res':  (['point'], res),

        'amp':   (['point','constituent'], amp),
        'phase': (['point','constituent'], phase),

        'amp_pyTMD': (['point','constituent'], np.array(tide_analyses['amp_pyTMD'])[valid_mask,:]),
        'ph_pyTMD':  (['point','constituent'], np.array(tide_analyses['ph_pyTMD'])[valid_mask,:]),
    },
    coords={
        'point': np.arange(len(x)),
        'constituent': np.arange(amp.shape[1]),
    }
)

out_ds.to_netcdf(f"{output_dir}/ross_ATL11_tides.nc")

# Plot amps
print("Plotting Amplitudes")
for i, const in enumerate(constituents): # Loop over amplitudes
    if const == 'K1' or const == 'O1':
        hmin = 0.1
        hmax = 0.4  
    else:
        hmin = 0
        hmax = 0.4
    ps71_projection = ccrs.Stereographic(central_latitude=-90, central_longitude=0, true_scale_latitude=-71)
    fig, ax = plt.subplots(
        figsize=(10, 10),
        subplot_kw={'projection': ps71_projection},
    )
    plot_shapefile(filtered_records, filtered_shapes, ax, ["lightgray", "lightgray"], ps71_projection, fill=True)
    n = -1

    xs = np.array(out_ds['x'][:n])
    ys = np.array(out_ds['y'][:n])
    amps = np.array([a[i] for a in out_ds['amp'][:n]])
    amps_pyTMD = np.array([a[i] for a in out_ds['amp_pyTMD'][:n]])
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
    xs = np.array(out_ds['x'][:n])
    ys = np.array(out_ds['y'][:n])
    amps = np.array([a[i] for a in out_ds['amp'][:n]])
    amps_pyTMD = np.array([a[i] for a in out_ds['amp_pyTMD'][:n]])
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
    cbar.set_label(f"{const} - CATS Amplitude [m]", fontsize=28, color="black")
    fig.savefig(f"{output_dir}/{const}_ha_CATSDiff.png", bbox_inches="tight", dpi=300, transparent=True)

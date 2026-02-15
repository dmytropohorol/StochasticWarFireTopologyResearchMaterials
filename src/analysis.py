# %% [CELL 1]
# --- IMPORTS ---
# This cell will fail if libraries listed below is not downloaded
# on your local machine. Check the documentation for the assistance.

# Powerful data structures for data analysis, time series, and statistics
import pandas as pd

# Geographic pandas extensions
import geopandas as gpd

# Python plotting package
import matplotlib.pyplot as plt

# Python package for creating and manipulating graphs and networks
import networkx as nx

# Fundamental package for array computing in Python
import numpy as np

# Fundamental algorithms for scientific computing in Python
from scipy.spatial import cKDTree

# The Gudhi library is an open source library for 
# Computational Topology and Topological Data Analysis (TDA).
import gudhi

# We need a clustering algorithm to group individual dots into "events"
from sklearn.cluster import DBSCAN 

print("Libraries loaded successfully!")

# %% [CELL 2] 
# --- LOAD & PROJECT DATA ---

# CRS: EPSG:3034 - ETRS89-extended / LCC Europe
# We are not using EPSG:3035 because its scope is statistical analysis,
# and we are not using EPSG:25832 because it's used for conformal 
# mapping at scales larger than 1:500,000
METRIC_CRS = "EPSG:3035" #TODO: test compared to EPSG:32636

DATA_PATH = "../data/viirs-jpss1_Ukraine_combined.csv" 

try:
    # Reading a csv file
    data_frame = pd.read_csv(
        DATA_PATH, 
        low_memory=True, 
        parse_dates=['acq_date'],
        dtype={
            'latitude': float, 
            'longitude': float, 
            'bright_ti4': float,
            'scan': float,
            'track': float,
            'acq_time': int,
            'satellite': str,
            'instrument': str,
            'confidence': str,
            'bright_ti5': float,
            'frp': float,
            'daynight': str,
            'type': int
            }
        )
    print(f"Original rows: {len(data_frame)}")

    # Filtering for only high confidence fire detections
    if 'confidence' in data_frame.columns:
        data_frame = data_frame[data_frame['confidence'] != 'l']
        print(f"High confidence rows: {len(data_frame)}")
    
    print("Projecting coordinates to meters...")
    
    # Reprojecting angles to geometry
    geometry = gpd.points_from_xy(data_frame['longitude'], data_frame['latitude'])
    geo_data_frame = gpd.GeoDataFrame(data_frame, geometry=geometry, crs="EPSG:4326") # 4326 = GPS

    # Converting to meters
    geo_data_frame_meters = geo_data_frame.to_crs(METRIC_CRS)

    # Writing new x and y coordinates into a DataFrame
    data_frame['x'] = geo_data_frame_meters.geometry.x
    data_frame['y'] = geo_data_frame_meters.geometry.y

    # Freeing the memory (csv could be extreamly big, so we need to free arrays explicitly)
    del geo_data_frame, geo_data_frame_meters, geometry

    print("Data projected to meters successfuly.")

    # Visualtion
    plt.figure(figsize=(20, 20))
    plt.scatter(data_frame.x, data_frame.y, s=0.1, c='red', alpha=0.5)
    plt.title(f"Projected Fire Points ({len(data_frame)} total)")
    plt.axis('equal')
    plt.show()

except Exception as e:
    print(e)
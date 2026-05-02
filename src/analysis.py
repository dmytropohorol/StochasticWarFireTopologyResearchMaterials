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
from sklearn.cluster import DBSCAN, HDBSCAN
from sklearn import metrics

import seaborn as sns
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.windows import from_bounds

print("Great success")


# %% [CELL 2] 
# --- LOAD & PROJECT DATA ---

# CRS: EPSG:3034 - ETRS89-extended / LCC Europe
# We are not using EPSG:3035 because its scope is statistical analysis,
# and we are not using EPSG:25832 because it's used for conformal 
# mapping at scales larger than 1:500,000
METRIC_CRS = "EPSG:3035" #TODO: test compared to EPSG:32636

DATA_PATH = "../data/viirs-jpss1/Ukraine/viirs-jpss1_2018-2024_Ukraine_formatted.csv" 

# Reading a csv file
data_frame = pd.read_csv(
    DATA_PATH, 
    low_memory=True, 
    parse_dates=['acq_date'],
    dayfirst=True
    )

# Filter data frame based on hotspot confidence and type (0 = presumed vegetation fire)
# and delete these columns out of necessity
data_frame = data_frame[data_frame['confidence'] != 'n']
data_frame = data_frame[data_frame['type'] == 0]
data_frame = data_frame.drop('confidence', axis=1)
data_frame = data_frame.drop('type', axis=1)

print(f"Hotspots in data frame: {len(data_frame)}")

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
data_frame = data_frame.drop('latitude', axis=1)
data_frame = data_frame.drop('longitude', axis=1)

print("Great success")



# %% [CELL 3] 
# --- REPROJECT RASTER TO METRIC FOR LAND COVER ---

OUTPUT_TIFF_3035 = "../data/land_cover_ukraine_3035.tif"

# Re-mapping dictionary for Copernicus classes https://zenodo.org/records/4723921
LAND_USE_MAP = {
    20: "Shrubs", 30: "Fields", 40: "Agriculture", 50: "Urban", 60: "Bare", 
    80: "Water", 111: "Forest", 112: "Forest", 113: "Forest", 114: "Forest", 
    115: "Forest", 116: "Forest", 121: "Forest_Open", 122: "Forest_Open", 
    123: "Forest_Open", 124: "Forest_Open", 125: "Forest_Open", 126: "Forest_Open"
}

with rasterio.open(OUTPUT_TIFF_3035) as src:
    coord_pairs = zip(data_frame['x'], data_frame['y'])
    data_frame['land_use_code'] = [val[0] for val in src.sample(coord_pairs)]

data_frame['land_use'] = data_frame['land_use_code'].map(LAND_USE_MAP).fillna("Other")

data_frame = data_frame.drop('land_use_code', axis=1)

print("Great success")


# %% [CELL 4] 
# --- DATA INTEGRITY AND ANALYSIS ---

# Checking if data is valid
plt.figure(figsize=(12, 6))
sns.heatmap(data_frame.isnull(), cbar=False, cmap='viridis', yticklabels=False)
plt.title('Heatmap of Missing Values in Dataset')
plt.show()

# Reviewing the data values
data_frame.hist(bins=30, figsize=(16, 12), edgecolor='black')
plt.suptitle('Histograms of All Features', fontsize=16)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
# countplot is better for categorical data like 'land_use'
sns.countplot(
    data=data_frame, 
    y='land_use', 
    order=data_frame['land_use'].value_counts().index,
    hue='land_use', 
    palette='viridis', 
    legend=False
)

plt.title('Distribution of Hotspots by Land Use Type')
plt.xlabel('Number of Hotspots')
plt.ylabel('Land Use Type')
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.show()

print("Great success")



# %% [CELL 5] 
# --- DATA CLUSTERIZATION ---

SPATIAL_EPS = 6000  # meters
DATE_EPS = 3        # days

# Because epsilon value in DBSCAN is the same for all axises, 
# we need to transform the date value

# Creating a new column for 3d axis — time. We need to filter not only by the position
# of the hotspot, but the time it happend.
time_delta = data_frame['acq_date'] - data_frame['acq_date'].min()
data_frame['days_since_start'] = time_delta.dt.total_seconds() / (24 * 60 * 60)

# Based on our spatial epsilon we need to change the date epsilon
time_scale_factor = SPATIAL_EPS / DATE_EPS
data_frame['z_time'] = data_frame['days_since_start'] * time_scale_factor
data_frame = data_frame.drop('days_since_start', axis=1)

X = data_frame[['x', 'y', 'z_time']].to_numpy()
hdb = HDBSCAN(
    min_cluster_size=5, 
    min_samples=None, 
    copy=True, 
    n_jobs=-1
).fit(X)

labels = hdb.dbscan_clustering(cut_distance=SPATIAL_EPS) # hdb.labels_
probabilities = hdb.probabilities_

n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)
print("Estimated number of clusters: %d" % n_clusters_)
print("Estimated number of noise points: %d" % n_noise_)
print(f"Silhouette Coefficient: {metrics.silhouette_score(X, labels):.3f}")

unique_labels = set(labels)

plt.figure(figsize=(12.8, 7.2), dpi=300) #TODO: Move to global config
colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = [0, 0, 0, 1]
        # Set zorder low so noise stays in the background
        layer = 1 
    else:
        # Set zorder higher so cluster points plot on top of noise
        layer = 2 

    # Mask for all members of the current class
    class_member_mask = (labels == k)
    xy = X[class_member_mask]

    if k == -1:
        # Plot noise all at once exactly as you had it
        plt.plot(
            xy[:, 0],
            xy[:, 1],
            marker=".",
            linestyle="none",    
            color=tuple(col),    
            markersize=1,
            zorder=layer         
        )
    else:
        cluster_probs = probabilities[class_member_mask]
        
        rgba_colors = np.zeros((len(cluster_probs), 4))
        rgba_colors[:, :3] = col[:3]      # Base RGB color
        rgba_colors[:, 3] = cluster_probs # Alpha = Probability (Edge points will fade out)

        plt.scatter(
            xy[:, 0],
            xy[:, 1],
            marker=".",
            c=rgba_colors,
            s=1, # Equivalent to markersize=1
            zorder=layer
        )

plt.show()

print("Great success")



# %% [CELL 4] 

# If a lot of points are located near the clusters, that is strange. x2 of radius i.e.
# If i can somehow check Spatiometral Velocity of points not just of the cluster, but
# the points around it too, i can theoretically detect if that fire is natural or not
# or it was ignated. The reason is that if the fire happens, it should move under some
# logic, and ignation should not appear kilometers away. In addition, it explains the
# shelling problem. When shelling, bombs or etc, are left in the forest, OR is activelly
# in use, it will make THE SAME hotspot activities. If for example



# %% [CELL 4] 

# Linearity analysis and Wind direction to linearity



# %% [CELL 4] 

# Final ML war-fires vs everything else distinguisher. 



# %% [CELL 4] 

# Picture, stats, infographics etc generation
# %% [CELL 1]
# --- IMPORTS ---
# This cell will fail if libraries listed below is not downloaded
# on your local machine. Check the documentation for the assistance.

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import HDBSCAN
from sklearn.neighbors import KDTree
import seaborn as sns
import rasterio
import numpy as np
import pandas as pd
import xarray as xr
print("Great success")



# %% [CELL 2] 
# --- LOAD & PROJECT DATA ---

RAW_HOTSPOT_DATA = "../data/viirs-jpss1/AllCountries/viirs-jpss1_2018-2024_AllCountries11.csv" 

# CRS: EPSG:3034 - ETRS89-extended / LCC Europe
# We are not using EPSG:3035 because its scope is statistical analysis,
# and we are not using EPSG:25832 because it's used for conformal 
# mapping at scales larger than 1:500,000
METRIC_CRS = "EPSG:3035"

# Reading a csv file
data_frame = pd.read_csv(
    RAW_HOTSPOT_DATA, 
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

# For data clusterization
SPATIAL_EPS = 5000  # meters
DATE_EPS = 3        # days


# --- 3D SPATIOTEMPORAL CLUSTERING ---

# HDBSCAN uses an isotropic distance metric (it treats all axes equally).
# To prevent time from dominating the spatial coordinates, we must scale 
# the temporal axis (Z) so that DATE_EPS mathematically equals SPATIAL_EPS.

# Convert absolute datetime to continuous days elapsed since the first detection
time_delta = data_frame['acq_date'] - data_frame['acq_date'].min()
data_frame['days_since_start'] = time_delta.dt.total_seconds() / 86400.0  # 86400 seconds in a day

# Scale the time axis: 1 unit of Z will now equal 1 unit of X/Y space
time_scale_factor = SPATIAL_EPS / DATE_EPS
data_frame['z_time'] = data_frame['days_since_start'] * time_scale_factor
data_frame = data_frame.drop('days_since_start', axis=1)

X_cluster = data_frame[['x', 'y', 'z_time']].to_numpy()

print("Running 3D Spatiotemporal Pass for Clusters...")
# Run HDBSCAN on the normalized 3D (X, Y, Scaled_Time) matrix
hdb = HDBSCAN(
    min_cluster_size=5, 
    min_samples=None, 
    copy=True, 
    n_jobs=-1
).fit(X_cluster)

# Force a flat cluster extraction using the spatial epsilon cut distance
cluster_labels = hdb.dbscan_clustering(cut_distance=SPATIAL_EPS)
probabilities = hdb.probabilities_

# Save the 3D Time-Cluster labels to the dataframe for later feature extraction
data_frame['cluster_id'] = cluster_labels


# --- CYCLICAL SEASONALITY ---
# Map Day of Year to a 2D circle to avoid the Dec 31st/Jan 1st boundary gap
days_in_year = 365.25
day_of_year = data_frame['acq_date'].dt.dayofyear
data_frame['day_sin'] = np.sin(2 * np.pi * day_of_year / days_in_year)
data_frame['day_cos'] = np.cos(2 * np.pi * day_of_year / days_in_year)


# --- TIME-CLUSTER METRICS ---
# We ignore noise points (-1) when calculating cluster aggregates
valid_time_clusters = data_frame[data_frame['cluster_id'] != -1]

# cluster_point_count (Intensity)
tc_counts = valid_time_clusters.groupby('cluster_id').size()
data_frame['cluster_point_count'] = data_frame['cluster_id'].map(tc_counts).fillna(1)

# cluster_duration_days (Duration)
tc_duration = valid_time_clusters.groupby('cluster_id')['acq_date'].agg(
    lambda x: (x.max() - x.min()).total_seconds() / 86400.0
)
data_frame['cluster_duration_days'] = data_frame['cluster_id'].map(tc_duration).fillna(0)


# --- CREATE SPATIAL MEGACLUSTERS ---
# We must collapse the time axis to find the 2D spatial boundaries
X_megacluster = data_frame[['x', 'y']].to_numpy()

print("Running 2D Spatial Pass for Megaclusters...")
hdb_spatial = hdb = HDBSCAN(
    min_cluster_size=5, 
    min_samples=None, 
    copy=True, 
    n_jobs=-1
).fit(X_megacluster)
megacluster_labels = hdb_spatial.labels_
data_frame['mega_cluster_id'] = megacluster_labels
valid_megaclusters = data_frame[data_frame['mega_cluster_id'] != -1]


# --- MEGACLUSTER METRICS ---
# megacluster_point_count (Historical Volume)
mc_counts = valid_megaclusters.groupby('mega_cluster_id').size()
data_frame['megacluster_point_count'] = data_frame['mega_cluster_id'].map(mc_counts).fillna(1)

# megacluster_cluster_count (Repeated Strikes)
# How many unique, valid 3D time-clusters happened inside this 2D boundary?
mc_repeats = valid_megaclusters[valid_megaclusters['cluster_id'] != -1].groupby('mega_cluster_id')['cluster_id'].nunique()
data_frame['megacluster_cluster_count'] = data_frame['mega_cluster_id'].map(mc_repeats).fillna(1)


# --- CIRCULAR VARIANCE (Agriculture vs Shelling) ---
def calc_circular_variance(dates):
    doy = dates.dt.dayofyear
    angles = 2 * np.pi * doy / days_in_year
    # Calculate Mean Resultant Vector Length (R)
    R = np.sqrt(np.sum(np.cos(angles))**2 + np.sum(np.sin(angles))**2) / len(angles)
    return 1.0 - R

print("Calculating Circular Variance...")
mc_circ_var = valid_megaclusters.groupby('mega_cluster_id')['acq_date'].apply(calc_circular_variance)
# Fill missing or noise points with 0 (treating them as singular events)
data_frame['circular_variance'] = data_frame['mega_cluster_id'].map(mc_circ_var).fillna(0)


# --- MEGACLUSTER EXPANSION RATIO (Trench Line Metric) ---
print("Calculating Spatial Density Gradient via KDTree...")
RADIUS_3KM = 3000 
RADIUS_6KM = 12000

tree = KDTree(X_megacluster)

# count_only=True runs at C-speed, avoiding memory overhead of returning actual indices
counts_3km = tree.query_radius(X_megacluster, r=RADIUS_3KM, count_only=True)
counts_6km = tree.query_radius(X_megacluster, r=RADIUS_6KM, count_only=True)

# Calculate ratio. np.maximum prevents division by zero, 
# though a point is always its own neighbor so counts_3km >= 1
data_frame['megacluster_expansion_ratio'] = counts_6km / np.maximum(counts_3km, 1)


# --- Linear Analysis ---
# TODO: For each cluster check the wind data and find the angle compared to
# eigen value of cluster

# --- Spcial Velocity Analysis ---
# TODO: For each cluster, megacluster and bigger clusters detect how fire moves


print("Feature Extraction Complete.")



# %% [CELL 5]
# --- DATA INTEGRITY AND ANALYSIS ---


# Checking if data is valid
plt.figure(figsize=(12.8, 7.2), dpi=300)
sns.heatmap(data_frame.isnull(), cbar=False, cmap='viridis', yticklabels=False)
plt.title('Heatmap of Missing Values in Dataset')
plt.show()

# Reviewing the data values
data_frame.hist(bins=30, figsize=(16, 12), edgecolor='black')
plt.suptitle('Histograms of All Features', fontsize=16)
plt.tight_layout()
plt.show()

plt.figure(figsize=(12.8, 7.2), dpi=300)
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

def visualize_hdbscan(labels):
    plt.figure(figsize=(12.8, 7.2), dpi=300)
    unique_labels = set(labels)
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
        xy = X_cluster[class_member_mask]

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

visualize_hdbscan(cluster_labels)

visualize_hdbscan(megacluster_labels)

print("Great success")



# %% [CELL 8] 
# --- MACHINE LEARNING: ANOMALY DETECTION ---

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

print("Preparing features for Machine Learning...")

# 1. Define the exact feature vector
# CRITICAL: Do NOT include x, y, time_cluster_id, mega_cluster_id, or acq_date.
ml_features = [
    'cluster_point_count', 
    'cluster_duration_days', 
    'megacluster_point_count', 
    'megacluster_cluster_count', 
    'megacluster_expansion_ratio', 
    'circular_variance', 
    'day_sin', 
    'day_cos',
    'bright_ti5',
    'frp',
    'temperature_2m_C',
    'total_precip_m',
    'soil_moisture',
    'wind_speed_m_s',
    'wind_direction_deg',
    'wind_gust',
]

# We will ignore 'land_use' for the first pass to avoid categorical encoding complexities.
# We only want to analyze data that survived the noise filter for feature generation
# (Though noise points are now size=1, they are valid events).
ml_data = data_frame.copy()

# 2. Split the dataset temporally (Peacetime vs War)
# We train ONLY on peacetime data to learn the baseline.
PEACETIME_END = '2022-02-24'
peacetime_mask = ml_data['acq_date'] < PEACETIME_END


#data_frame = data_frame.drop('acq_date', axis=1)



train_data = ml_data[peacetime_mask][ml_features]
test_data = ml_data[~peacetime_mask][ml_features]

print(f"Training on {len(train_data)} peacetime events.")
print(f"Testing on {len(test_data)} wartime events.")

# 3. Scale the features
# Even though tree-based models handle unscaled data okay, scaling ensures
# features like cluster_point_count (1000s) don't overshadow circular_variance (0-1).
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(train_data)
X_test_scaled = scaler.transform(test_data)

# 4. Train the Isolation Forest
print("Training Isolation Forest on Peacetime Baseline...")
iso_forest = IsolationForest(
    n_estimators=15000,        # Number of trees
    max_samples='auto',      
    contamination=0.05,      # Assume 2% of peacetime fires were genuine outliers
    random_state=42,
    n_jobs=-1
)
iso_forest.fit(X_train_scaled)

# 5. Predict on the War-Time Data
print("Scoring Wartime Data...")
# Predict returns 1 (normal) or -1 (anomaly)
predictions = iso_forest.predict(X_test_scaled)
# Score samples returns a continuous score. Lower score = More Anomalous.
anomaly_scores = iso_forest.score_samples(X_test_scaled) 

# Map results back to the testing dataframe
results_df = ml_data[~peacetime_mask].copy()
results_df['is_anomaly'] = predictions
results_df['anomaly_score'] = anomaly_scores

print("Prediction Complete.")


# --- VISUALIZATION: DID IT GUESS CORRECTLY? ---
print("Generating Anomaly Map...")

# We want to isolate the absolute most extreme anomalies (e.g., the bottom 5% of scores)
threshold = results_df['anomaly_score'].quantile(0.1)
extreme_anomalies = results_df[results_df['anomaly_score'] < threshold]
normal_war_fires = results_df[results_df['anomaly_score'] >= threshold]

plt.figure(figsize=(14, 10), dpi=300)

# Plot the "normal" wartime fires (agriculture, natural) in the background
plt.scatter(
    normal_war_fires['x'], 
    normal_war_fires['y'], 
    c='lightgray', 
    s=2, 
    alpha=0.5, 
    label='Classified: Normal Behavior'
)

# Plot the extreme anomalies on top
plt.scatter(
    extreme_anomalies['x'], 
    extreme_anomalies['y'], 
    c='red', 
    s=10, 
    alpha=0.8, 
    label='Classified: Extreme Anomaly (War)'
)

plt.title('Isolation Forest Predictions: Top 5% Spatiotemporal Anomalies', fontsize=16)
plt.xlabel('X (Meters - EPSG:3035)')
plt.ylabel('Y (Meters - EPSG:3035)')
plt.legend(loc='upper left')
plt.axis('equal') # Ensure the map isn't stretched
plt.grid(True, linestyle='--', alpha=0.3)
plt.show()

# Print the stats of the worst anomalies
print("\n--- Average Features of Extreme Anomalies vs Normal Fires ---")
comparison = results_df.groupby(results_df['anomaly_score'] < threshold)[ml_features].mean().T
comparison.columns = ['Normal', 'Anomaly']
print(comparison.round(2))


# %% [CELL 9] 

# Picture, stats, infographics etc generation

# Get air quality and detect a shelling using it
# Use deepstate to distinguish war-fires between everything else
import os
import requests
import rasterio
from rasterio.windows import from_bounds
from rasterio.warp import calculate_default_transform, reproject, Resampling
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')

# --- CONFIG ---
DOWNLOAD_URL = "https://zenodo.org/records/3939050/files/PROBAV_LC100_global_v3.0.1_2019-nrt_Discrete-Classification-map_EPSG-4326.tif?download=1"
RAW_TIFF = os.path.join(DATA_DIR, "PROBAV_LC100_global_v3.0.1_2019-nrt_Discrete-Classification-map_EPSG-4326.tif")
UKR_TIFF = os.path.join(DATA_DIR, "land_cover_ukraine_3035.tif")

os.makedirs(os.path.dirname(RAW_TIFF), exist_ok=True)

# --- DOWNLOAD RAW TIFF IF MISSING ---
if os.path.exists(UKR_TIFF):
    print(f"File {UKR_TIFF} already exists.")
    exit()

if not os.path.exists(RAW_TIFF):
    print("Downloading global land cover TIFF (this might take a few minutes)...")
    with requests.get(DOWNLOAD_URL, stream=True) as r:
        r.raise_for_status()
        with open(RAW_TIFF, 'wb') as f:
            # Write in 8KB chunks so we don't overwhelm memory
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    print("Download complete.")


# --- REPROJECT & CROP ---
print("Cropping & Reprojecting global TIFF to EPSG:3035...")
with rasterio.open(RAW_TIFF) as src:
    win = from_bounds(22.0, 44.0, 40.5, 52.5, src.transform)
    win_transform = src.window_transform(win)
    
    dst_transform, w, h = calculate_default_transform(
        src.crs, 'EPSG:3035', win.width, win.height, 
        left=22.0, bottom=44.0, right=40.5, top=52.5
    )
    
    kwargs = src.meta.copy()
    kwargs.update({'crs': 'EPSG:3035', 'transform': dst_transform, 'width': w, 'height': h, 'compress': 'lzw'})
    
    with rasterio.open(UKR_TIFF, 'w', **kwargs) as dst:
        reproject(
            source=src.read(1, window=win), 
            destination=rasterio.band(dst, 1), 
            src_transform=win_transform, 
            src_crs=src.crs, 
            dst_transform=dst_transform, 
            dst_crs='EPSG:3035', 
            resampling=Resampling.nearest
        )
print(f"Great Success. Created {UKR_TIFF}.")
import ee
import json
import os
import argparse
from datetime import datetime, timedelta
import geopandas as gpd

def init_gee():
    """Initializes Google Earth Engine. Requires running `earthengine authenticate` first."""
    try:
        ee.Initialize(project='ee-fires-shapes') # User might need to change project name or just use ee.Initialize()
    except Exception as e:
        print("Earth Engine initialization failed. Please run 'earthengine authenticate' in your terminal.")
        try:
            # Fallback for default initialization
            ee.Initialize()
        except Exception as e:
            print("Fallback initialization also failed. Ensure you have an active GEE project.")
            raise e

def extract_burned_area(lat, lon, fire_date_str, buffer_meters=5000, output_file="burned_area.geojson"):
    """
    Extracts a burned area polygon using Sentinel-2 and dNBR.
    
    :param lat: Latitude of the fire cluster center
    :param lon: Longitude of the fire cluster center
    :param fire_date_str: Date of the fire (YYYY-MM-DD)
    :param buffer_meters: Radius around the center to analyze
    :param output_file: Output path for the GeoJSON
    """
    init_gee()
    
    print(f"Analyzing fire at {lat}, {lon} on {fire_date_str}...")
    
    fire_date = datetime.strptime(fire_date_str, "%Y-%m-%d")
    
    # 1. Define Temporal Windows
    # Pre-fire: 1 month before the fire up to 1 day before
    pre_start = (fire_date - timedelta(days=30)).strftime("%Y-%m-%d")
    pre_end = (fire_date - timedelta(days=1)).strftime("%Y-%m-%d")
    
    # Post-fire: 1 day after the fire up to 15 days after
    post_start = (fire_date + timedelta(days=1)).strftime("%Y-%m-%d")
    post_end = (fire_date + timedelta(days=15)).strftime("%Y-%m-%d")
    
    print(f"  Pre-fire window: {pre_start} to {pre_end}")
    print(f"  Post-fire window: {post_start} to {post_end}")

    # 2. Define Spatial ROI
    point = ee.Geometry.Point([lon, lat])
    roi = point.buffer(buffer_meters).bounds()

    # 3. Sentinel-2 Cloud Masking Function
    def mask_s2_clouds(image):
        qa = image.select('QA60')
        # Bits 10 and 11 are clouds and cirrus, respectively.
        cloudBitMask = 1 << 10
        cirrusBitMask = 1 << 11
        # Both flags should be set to zero, indicating clear conditions.
        mask = qa.bitwiseAnd(cloudBitMask).eq(0).And(qa.bitwiseAnd(cirrusBitMask).eq(0))
        return image.updateMask(mask).divide(10000)

    # 4. Fetch and Process Pre-Fire Image
    s2 = ee.ImageCollection("COPERNICUS/S2_HARMONIZED")
    
    pre_fire = (s2.filterBounds(roi)
                .filterDate(pre_start, pre_end)
                .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
                .map(mask_s2_clouds)
                .median()
                .clip(roi))

    # 5. Fetch and Process Post-Fire Image
    post_fire = (s2.filterBounds(roi)
                 .filterDate(post_start, post_end)
                 .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
                 .map(mask_s2_clouds)
                 .median()
                 .clip(roi))

    # 6. Calculate NBR (Normalized Burn Ratio)
    # NBR = (NIR - SWIR2) / (NIR + SWIR2) -> Sentinel-2 bands B8A and B12
    # Note: B8 (NIR) and B12 (SWIR2) is standard.
    pre_nbr = pre_fire.normalizedDifference(['B8', 'B12']).rename('NBR')
    post_nbr = post_fire.normalizedDifference(['B8', 'B12']).rename('NBR')

    # 7. Calculate dNBR (Difference NBR)
    dnbr = pre_nbr.subtract(post_nbr).rename('dNBR')

    # 8. Thresholding (Algorithm Baseline)
    # Values > 0.1 indicate low severity burn. > 0.27 is moderate.
    # We use > 0.1 to capture the full polygon for ML later.
    burn_mask = dnbr.gt(0.1)

    # 9. Convert Raster to Vector (Polygons) in the Cloud
    print("  Vectorizing burned area mask in Earth Engine...")
    vectors = burn_mask.selfMask().reduceToVectors(
        geometry=roi,
        crs=pre_fire.select('B8').projection(),
        scale=10,  # 10m resolution for Sentinel-2 NIR
        geometryType='polygon',
        eightConnected=True,
        labelProperty='burn_class',
        reducer=ee.Reducer.countEvery(),
        maxPixels=1e9
    )

    # 10. Download GeoJSON
    print("  Downloading GeoJSON feature collection...")
    try:
        # Get the feature collection as a dictionary
        # Note: If this fails due to large size, we would use ee.batch.Export
        geojson_dict = vectors.getInfo() 
        
        # Add metadata to the features
        for feature in geojson_dict['features']:
            feature['properties']['fire_date'] = fire_date_str
            feature['properties']['center_lat'] = lat
            feature['properties']['center_lon'] = lon
            feature['properties']['source'] = 'Sentinel-2 dNBR'
            
        with open(output_file, 'w') as f:
            json.dump(geojson_dict, f)
            
        print(f"Success! Burned area saved to {output_file}")
        
    except Exception as e:
        print(f"Failed to fetch geometry from Earth Engine: {e}")
        print("The polygon might be too large or complex for getInfo(). Consider exporting to Google Drive.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract burned area polygons using GEE.")
    parser.add_argument("--lat", type=float, required=True, help="Latitude of the fire.")
    parser.add_argument("--lon", type=float, required=True, help="Longitude of the fire.")
    parser.add_argument("--date", type=str, required=True, help="Date of the fire (YYYY-MM-DD).")
    parser.add_argument("--buffer", type=int, default=5000, help="Buffer size in meters (default 5000).")
    parser.add_argument("--out", type=str, default="data/generated/test_burn.geojson", help="Output GeoJSON file.")
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    
    extract_burned_area(args.lat, args.lon, args.date, args.buffer, args.out)

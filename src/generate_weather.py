import pandas as pd
import numpy as np
import xarray as xr
import os
import warnings

# Suppress standard xarray/cfgrib warnings for cleaner console output
warnings.filterwarnings("ignore")

# ==========================================
# CONFIGURATION - UPDATE THESE PATHS
# ==========================================
CSV_PATH = "data/viirs-jpss1/AllCountries/viirs-jpss1_2018-2024_AllCountries1st.csv"
NC_PATH = "data/era5_eastern_europe_*/*.nc"

def append_weather_to_csv(csv_path, nc_path):
    print(f"Loading CSV data from: {csv_path}")
    # Load the CSV. Crucial: Parse 'acq_date' as datetime immediately!
    data_frame = pd.read_csv(CSV_PATH, low_memory=False)    

    data_frame['_temp_time'] = pd.to_datetime(data_frame['acq_date'], dayfirst=True)

    initial_rows = len(data_frame)
    print(f"Loaded {initial_rows} hotspots.")

    # ---------------------------------------------------------
    # 1. FIX VIIRS TIME FORMATTING
    # ---------------------------------------------------------
    print("Parsing VIIRS timestamps...")
    # Pad with zeros so '936' becomes '0936'
    time_str = data_frame['acq_time'].astype(str).str.zfill(4)

    # Extract hours and minutes
    hours = time_str.str[:2].astype(int)
    minutes = time_str.str[2:].astype(int)

    # Combine _temp_time with the parsed hours/minutes
    data_frame['datetime_utc'] = data_frame['_temp_time'] + \
                                 pd.to_timedelta(hours, unit='h') + \
                                 pd.to_timedelta(minutes, unit='m')

    # ERA5 is hourly, so we round the fire time to the nearest exact hour
    data_frame['era5_time_match'] = data_frame['datetime_utc'].dt.round('h')


    # ---------------------------------------------------------
    # 2. LOAD ALL WEATHER FILES AS ONE VIRTUAL DATABASE
    # ---------------------------------------------------------
    print("Loading continuous GRIB/NetCDF database...")
    # Using the cfgrib engine as required by your downloaded format
    ds = xr.open_mfdataset(nc_path, combine='by_coords', engine='netcdf4')


    # ---------------------------------------------------------
    # 3. VECTORIZED EXTRACTION
    # ---------------------------------------------------------
    print("Extracting weather profiles for all hotspots simultaneously...")
    lats = xr.DataArray(data_frame['latitude'].values, dims='points')
    lons = xr.DataArray(data_frame['longitude'].values, dims='points')
    times = xr.DataArray(data_frame['era5_time_match'].values, dims='points')

    # Extract data for all points in one single C-level operation
    weather_data = ds.sel(latitude=lats, longitude=lons, valid_time=times, method='nearest')


    # ---------------------------------------------------------
    # 4. MAP THE EXTRACTED DATA BACK TO DATAFRAME
    # ---------------------------------------------------------
    print("Mapping variables and calculating wind vectors...")

    # Convert Kelvin to Celsius
    data_frame['temperature_2m_C'] = weather_data['t2m'].values - 273.15

    # Total precipitation (meters) and Soil Moisture (m³/m³)
    data_frame['total_precip_m'] = weather_data['tp'].values
    data_frame['soil_moisture'] = weather_data['swvl1'].values

    # Calculate exact Wind Speed & Direction from the U and V vectors
    u10 = weather_data['u10'].values
    v10 = weather_data['v10'].values

    data_frame['wind_speed_m_s'] = np.sqrt(u10**2 + v10**2)
    # Standard meteorological wind direction (0=North, 90=East, etc.)
    data_frame['wind_direction_deg'] = (270 - np.rad2deg(np.arctan2(v10, u10))) % 360

    data_frame['wind_gust'] = weather_data['fg10'].values
    data_frame['cloud_base'] = weather_data['cbh'].values

    # ---------------------------------------------------------
    # 5. CLEANUP & SAVE
    # ---------------------------------------------------------
    # Drop temporary datetime match column, but KEEP latitude/longitude
    data_frame = data_frame.drop(columns=['era5_time_match'])
    
    # Close the xarray dataset to free up memory/file locks
    ds.close()

    print(f"Overwriting original CSV with new weather columns...")
    # Save back to the exact same CSV path, modifying it in place
    data_frame = data_frame.drop(columns=['_temp_time'])

    data_frame.to_csv(CSV_PATH, index=False)
    
    print("Great success! Weather merged and CSV updated.")


if __name__ == "__main__":
    # Ensure the files actually exist before running
    if not os.path.exists(CSV_PATH):
        print(f"Error: Could not find CSV file at {CSV_PATH}")
    else:
        append_weather_to_csv(CSV_PATH, NC_PATH)
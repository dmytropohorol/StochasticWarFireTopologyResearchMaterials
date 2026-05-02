import os
import sys
import time
import requests
import pandas as pd
from datetime import datetime
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter

# --- CONFIGURATION ---
INPUT_FILE_FOLDER = 'data/custom_pre_war'
INPUT_FILE_NAME = 'viirs-jpss1_Ukraine_pre_war_only_high_confidence'
CHUNK_SIZE = 100  # Number of coordinates per API call

# --- SETUP ROBUST SESSION ---
session = requests.Session()
retries = Retry(
    total=3,                # Try 3 times total
    backoff_factor=3,       # Wait 3s, then 9s, then 27s between retries
    status_forcelist=[429, 500, 502, 503, 504], # Retry on these errors
    allowed_methods=["GET"]
)
session.mount('https://', HTTPAdapter(max_retries=retries))

# --- PREPARE DATA ---  
data_frame = pd.read_csv(f'{INPUT_FILE_FOLDER}/{INPUT_FILE_NAME}.csv')

# Ensure valid date format for grouping
# (This creates a helper column solely for grouping purposes)
# DO NOT TRUST MICROSOFT EXCEL. It shows grouping_date the same as acq_date, but they are not the same.
data_frame['grouping_date'] = pd.to_datetime(data_frame['acq_date'], dayfirst=True).dt.strftime('%Y-%m-%d')

# Prepare empty columns
data_frame['wind_speed_10m_mean'] = None
data_frame['wind_direction_10m_dominant'] = None

# --- PROCESS BY DATE ---
# We group by the formatted date string so all rows for "2022-01-01" stay together
grouped = data_frame.groupby('grouping_date')

print(f"Found {len(grouped)} unique days to process.")

url = "https://archive-api.open-meteo.com/v1/archive"
try:
    for date_str, group in grouped:
        # 'group' is a mini-dataframe containing ALL rows for this specific date
        
        # We slice this day's group into smaller chunks (e.g., 50 rows)
        # to avoid making the API URL too long
        for i in range(0, len(group), CHUNK_SIZE):
            chunk = group.iloc[i : i + CHUNK_SIZE]
            
            # Create comma-separated lists of lat/lon
            lat_str = ",".join(chunk['latitude'].astype(str))
            lon_str = ",".join(chunk['longitude'].astype(str))
            
            params = {
                "latitude": lat_str,
                "longitude": lon_str,
                "start_date": date_str,
                "end_date": date_str,
                "daily": "wind_speed_10m_mean,wind_direction_10m_dominant",
                "timezone": "Africa/Cairo"
            }
            
            try:
                # timeout=20 means "give up if server doesn't respond in 20 seconds"
                # The session.get() will automatically retry if it times out.
                r = session.get(url, params=params, timeout=20)
                r.raise_for_status()
                data = r.json()
                
                # The API returns a LIST of results if multiple coords are sent.
                # Usually data is a list of dicts, or a dict containing lists.
                # Open-Meteo returns a LIST of objects when multiple coords are used.
                if isinstance(data, list):
                    results = data
                else:
                    # If only 1 coordinate was sent, it returns a single dict, wrap in list
                    results = [data]
                    
                # Map results back to the dataframe
                # The API returns results in the same order we sent the coordinates
                current_res_idx = 0
                for index, row in chunk.iterrows():
                    daily = results[current_res_idx].get('daily', {})
                    
                    ws = daily.get('wind_speed_10m_mean', [None])[0]
                    wd = daily.get('wind_direction_10m_dominant', [None])[0]
                    
                    data_frame.at[index, 'wind_speed_10m_mean'] = ws
                    data_frame.at[index, 'wind_direction_10m_dominant'] = wd
                    
                    current_res_idx += 1
                
                print(f"Success: {date_str} (Batch of {len(chunk)} rows)")
                
            except Exception as e:
                print(f"FAILED: {date_str} batch. Error: {e}")
                
except KeyboardInterrupt:
    print("\n Script stopped by user! Saving progress...")

    data_frame.drop(columns=['grouping_date'], inplace=True)

    os.makedirs(f"{INPUT_FILE_FOLDER}/backup", exist_ok=True)
    current_timesptamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file = f'{INPUT_FILE_FOLDER}/backup/{INPUT_FILE_NAME}_{current_timesptamp}.csv'

    data_frame.to_csv(file, index=False)

    print(f"Progress saved to {file}")

    sys.exit()

# --- CLEANUP AND SAVE ---
data_frame.drop(columns=['grouping_date'], inplace=True)

os.makedirs(f"{INPUT_FILE_FOLDER}/generated", exist_ok=True)
current_timesptamp = datetime.now().strftime("%Y%m%d_%H%M%S")
data_frame.to_csv(f'{INPUT_FILE_FOLDER}/generated/{INPUT_FILE_NAME}_{current_timesptamp}.csv', index=False)

print("Processing complete.")
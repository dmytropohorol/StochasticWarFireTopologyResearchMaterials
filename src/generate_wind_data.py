import os
import sys
import time
import requests
import pandas as pd
from datetime import datetime
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter

# --- CONFIGURATION ---
INPUT_FILE_FOLDER = 'data/viirs-jpss1/AllCountries'
INPUT_FILE_NAME = 'viirs-jpss1_2018-2024_AllCountries3iter'
CHUNK_SIZE = 50

HOURLY_VARIABLES = [
    "wind_speed_10m", "wind_direction_10m", "wind_gusts_10m",
    "soil_moisture_0_to_7cm", "relative_humidity_2m", "cloud_cover",
    "temperature_2m", "vapour_pressure_deficit", "precipitation"
]

# --- SETUP ROBUST SESSION ---
session = requests.Session()
retries = Retry(
    total=6,
    backoff_factor=2,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["GET"] 
)
session.mount('https://', HTTPAdapter(max_retries=retries))

# --- HELPER FUNCTION FOR TIME ROUNDING ---
def calculate_target_time(row):
    acq_time = row['acq_time']
    if pd.isna(acq_time): 
        return f"{row['grouping_date']}T12:00"
        
    val = int(acq_time)
    hours = val // 100
    minutes = val % 100
    
    if minutes > 30:
        hours += 1
        
    if hours >= 24:
        hours = 23
        
    return f"{row['grouping_date']}T{hours:02d}:00"


# --- PREPARE DATA ---  
data_frame = pd.read_csv(f'{INPUT_FILE_FOLDER}/{INPUT_FILE_NAME}.csv')

data_frame['grouping_date'] = pd.to_datetime(data_frame['acq_date'], dayfirst=True).dt.strftime('%Y-%m-%d')
data_frame['target_time_str'] = data_frame.apply(calculate_target_time, axis=1)

for var in HOURLY_VARIABLES:
    data_frame[var] = None

print(f"Loaded {len(data_frame)} rows. Grouping by day and processing...")

# --- SETUP LIVE OUTPUT FILE ---
os.makedirs(f"{INPUT_FILE_FOLDER}/generated", exist_ok=True)
current_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
LIVE_OUTPUT_FILE = f'{INPUT_FILE_FOLDER}/generated/{INPUT_FILE_NAME}_LIVE_{current_timestamp}.csv'
print(f"Live data will be constantly written to: {LIVE_OUTPUT_FILE}")

url = "https://archive-api.open-meteo.com/v1/archive"
grouped_by_date = data_frame.groupby('grouping_date')
total_processed = 0

try:
    for date_str, group in grouped_by_date:
        
        for i in range(0, len(group), CHUNK_SIZE):
            chunk = group.iloc[i : i + CHUNK_SIZE]
            
            lat_str = ",".join(chunk['latitude'].astype(str).tolist())
            lon_str = ",".join(chunk['longitude'].astype(str).tolist())
            hourly_str = ",".join(HOURLY_VARIABLES)
            
            params = {
                "latitude": lat_str,
                "longitude": lon_str,
                "start_date": date_str,
                "end_date": date_str,
                "hourly": hourly_str,
                "timezone": "GMT"
            }
            
            try:
                r = session.get(url, params=params, timeout=30)
                r.raise_for_status()
                data = r.json()
                
                results = data if isinstance(data, list) else [data]
                
                for (index, row), res in zip(chunk.iterrows(), results):
                    target_time_str = row['target_time_str']
                    hourly_data = res.get('hourly', {})
                    times = hourly_data.get('time', [])
                    
                    try:
                        date_idx = times.index(target_time_str)
                        for var in HOURLY_VARIABLES:
                            data_frame.at[index, var] = hourly_data.get(var, [])[date_idx]
                            
                    except (ValueError, IndexError):
                        pass
                
                total_processed += len(chunk)
                print(f"Success: {total_processed}/{len(data_frame)} processed | Date: {date_str} | Chunk Size: {len(chunk)}", flush=True)
                
                # Extract only the 50 rows we just finished modifying
                finished_chunk = data_frame.loc[chunk.index].drop(columns=['grouping_date', 'target_time_str'], errors='ignore')
                
                # Append them to the live CSV. If the file doesn't exist yet, it writes the column headers.
                finished_chunk.to_csv(LIVE_OUTPUT_FILE, mode='a', header=not os.path.exists(LIVE_OUTPUT_FILE), index=False)

                time.sleep(5)

            except Exception as e:
                print(f"FAILED on Date: {date_str}. Error: {e}")
                
except KeyboardInterrupt:
    print("\n Script stopped by user! The Live CSV is already safely on your hard drive.")
    sys.exit()

print(f"Processing complete. Final data is ready at {LIVE_OUTPUT_FILE}")
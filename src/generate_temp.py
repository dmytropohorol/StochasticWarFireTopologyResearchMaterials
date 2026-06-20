import cdsapi
import os

c = cdsapi.Client(
    url="https://cds.climate.copernicus.eu/api", 
    key="6a7ea6ab-b83b-4fcc-9fba-d44aabff2a0f"
)

output_dir = 'D:/geospatial_researches/FiresShapesNewWork/data/'
os.makedirs(output_dir, exist_ok=True)

years = ['2018']
months = ['09', '10', '11', '12']
days = [str(i).zfill(2) for i in range(1, 32)]
times = [f"{str(i).zfill(2)}:00" for i in range(24)]

print("Initiating monthly chunking download pipeline...")

for year in years:
    for month in months:
        filename = f'{output_dir}era5_eastern_europe_{year}_{month}.nc'
        
        if os.path.exists(filename):
            print(f"[{year}-{month}] already exists. Skipping...")
            continue
            
        print(f"Requesting data for {year}-{month}...")
        
        try:
            c.retrieve(
                'reanalysis-era5-single-levels',
                {
                    'product_type': 'reanalysis',
                    'variable': [
                        # Replaced speed/direction with raw U/V vectors
                        '10m_u_component_of_wind', '10m_v_component_of_wind', 
                        '10m_wind_gust_since_previous_post_processing',
                        'volumetric_soil_water_layer_1', '2m_temperature', 'total_precipitation',
                        'cloud_base_height' 
                    ],
                    'year': year,
                    'month': month,
                    'day': days,
                    'time': times,
                    'area': [56.0, 20.0, 44.0, 41.0], 
                    'data_format': 'netcdf', # Updated key
                },
                filename
            )
            print(f"Successfully downloaded {year}-{month}!")
            
        except Exception as e:
            print(f"\n[!] Pipeline halted on {year}-{month}.")
            print(f"Error details: {e}")
            print("You can rerun the script later and it will resume from this exact spot.")
            exit()
            
print("\nAll 84 months downloaded successfully!")
import os
import glob
import zipfile

data_dir = 'D:/geospatial_researches/FiresShapesNewWork/data/'

# Find all the "fake" .nc files
fake_nc_files = glob.glob(os.path.join(data_dir, 'era5_eastern_europe_*.zip'))

print(f"Found {len(fake_nc_files)} disguised zip files. Extracting...")

for bad_file in fake_nc_files:
    # 2. Create a specific folder for this month's data
    extract_folder = bad_file.replace('.zip', '')
    os.makedirs(extract_folder, exist_ok=True)
    
    # 3. Extract the two actual NetCDF files into that folder
    with zipfile.ZipFile(bad_file, 'r') as zip_ref:
        zip_ref.extractall(extract_folder)
        
    # Optional: delete the zip file to save space
    os.remove(bad_file) 

print("Extraction complete! Your NetCDF files are ready.")
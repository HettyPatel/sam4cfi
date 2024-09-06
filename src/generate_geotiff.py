import os
import rasterio
import numpy as np

def process_files(label_dir, water_farm_dir, npy_dir, output_dir):
    
    os.makedirs(output_dir, exist_ok=True)

    # List all label files and npy files
    label_files = [f for f in os.listdir(label_dir) if f.endswith('_RAW_LABEL.tif')]
    water_farm_files = [f for f in os.listdir(water_farm_dir) if f.endswith('_RAW_LABEL_FARM.tif')]
    npy_files = [f for f in os.listdir(npy_dir) if f.endswith('.npy')]

    for label_file in label_files:
        # Extract the identifier from the label file name (e.g., T10SEH_2018)
        id_ = label_file.split('_RAW_LABEL.tif')[0]

        # Find all corresponding npy files for this identifier
        matching_npy_files = [f for f in npy_files if f.startswith(id_)]
        matching_water_farm_file = next((f for f in water_farm_files if f.startswith(id_)), None)

        if matching_npy_files and matching_water_farm_file:
            for npy_file in matching_npy_files:
                # Load the npy file
                npy_file_path = os.path.join(npy_dir, npy_file)
                data = np.load(npy_file_path)  # Shape: (24, 10, h, w)

                # Extract row and column from the npy file name
                parts = npy_file.split('_')
                row = int(parts[2]) * data.shape[2]  # Height of npy data
                col = int(parts[3]) * data.shape[3]  # Width of npy data

                # Load the transformation matrix and label data from the label file
                label_file_path = os.path.join(label_dir, label_file)
                with rasterio.open(label_file_path) as dataset:
                    transform = dataset.transform
                    crs = dataset.crs
                    label_data = dataset.read(1)  # Read the label data

                # Extract the corresponding label data section
                label_data_cut = label_data[row:row+data.shape[2], col:col+data.shape[3]]

                # Load and align the water farm file
                water_farm_file_path = os.path.join(water_farm_dir, matching_water_farm_file)
                with rasterio.open(water_farm_file_path) as dataset:
                    water_farm_data = dataset.read(1)  # Read the water farm data

                # Extract the corresponding water farm data section
                water_farm_data_cut = water_farm_data[row:row+data.shape[2], col:col+data.shape[3]]

                # Combine the original npy data (240 bands) with the label and water farm data (2 bands)
                combined_data = np.concatenate(
                    (data.reshape(240, data.shape[2], data.shape[3]), 
                     label_data_cut[np.newaxis, :, :], 
                     water_farm_data_cut[np.newaxis, :, :]), 
                    axis=0
                )  # Shape: (242, h, w)

                # Write the data to a GeoTIFF file
                geotiff_file_name = f"{os.path.splitext(npy_file)[0]}_combined.tif"
                geotiff_file_path = os.path.join(output_dir, geotiff_file_name)

                # Write the data to a single GeoTIFF file
                with rasterio.open(
                    geotiff_file_path,
                    'w',
                    driver='GTiff',
                    height=combined_data.shape[1],  # Height
                    width=combined_data.shape[2],  # Width
                    count=combined_data.shape[0],  # Total number of bands (240 + 2)
                    dtype=combined_data.dtype,
                    crs=crs,
                    transform=transform
                ) as dst:
                    for band_index in range(combined_data.shape[0]):
                        dst.write(combined_data[band_index], band_index + 1)

                print(f"GeoTIFF file created at {geotiff_file_path}")

# Define the directories
label_dir = '/data/hpate061/CalCROP21/ACCEPTABLE_GRIDS/LABEL_MAPS'
water_farm_dir = '/data/hpate061/CalCROP21/ACCEPTABLE_GRIDS/WATER_FARM_GEOTIFF'
npy_dir = '/data/hpate061/CalCROP21/ACCEPTABLE_GRIDS/IMAGE_GRIDS'
output_dir = '/data/hpate061/CalCROP21/ACCEPTABLE_GRIDS/GeoTIFF'

# Process the files
process_files(label_dir, water_farm_dir, npy_dir, output_dir)

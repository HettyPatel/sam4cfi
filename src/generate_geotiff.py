'''
Script to iterate through each '.npy' file and its corresponding label GeoTIFF file, pre-process the data,
and write the output to a new GeoTIFF file. 
'''
import os
import rasterio
from rasterio.transform import Affine
import numpy as np

def process_files(label_dir, npy_dir, output_dir):
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # List all label files and npy files
    label_files = [f for f in os.listdir(label_dir) if f.endswith('_RAW_LABEL.tif')]
    npy_files = [f for f in os.listdir(npy_dir) if f.endswith('.npy')]

    for label_file in label_files:
        # Extract the identifier from the label file name (e.g., T10SEH_2018)
        id_ = label_file.split('_RAW_LABEL.tif')[0]

        # Find all corresponding npy files for this identifier
        matching_npy_files = [f for f in npy_files if f.startswith(id_)]

        for npy_file in matching_npy_files:
            # Load the label file
            label_file_path = os.path.join(label_dir, label_file)
            npy_file_path = os.path.join(npy_dir, npy_file)

            # Load the npy file
            data = np.load(npy_file_path)  # Shape: (24, 10, 1098, 1098)

            # Extract row and column from the npy file name
            parts = npy_file.split('_')
            row = int(parts[2]) * 1098
            col = int(parts[3]) * 1098

            # Load the transformation matrix from the label file
            with rasterio.open(label_file_path) as dataset:
                transform = dataset.transform
                crs = dataset.crs

            # Calculate new upper-left coordinates
            new_upper_left_x, new_upper_left_y = transform * (col, row)

            # Extract the scale and rotation components from the original transformation
            pixel_size_x = transform.a
            pixel_size_y = transform.e
            rotation_x = transform.b
            rotation_y = transform.d

            # Create a new Affine transformation matrix with the new upper-left coordinates
            new_transform = Affine.translation(new_upper_left_x, new_upper_left_y) * Affine(
                pixel_size_x, rotation_x, 0,
                rotation_y, pixel_size_y, 0
            )

            # Calculate the total number of bands (timesteps * original bands)
            total_bands = data.shape[0] * data.shape[1]  # (24 * 10)

            # Define the output GeoTIFF file path, using the original npy file name for uniqueness
            geotiff_file_name = f"{os.path.splitext(npy_file)[0]}.tif"
            geotiff_file_path = os.path.join(output_dir, geotiff_file_name)

            # Write the data to a single GeoTIFF file
            with rasterio.open(
                geotiff_file_path,
                'w',
                driver='GTiff',
                height=data.shape[2],  # 1098
                width=data.shape[3],  # 1098
                count=total_bands,  # Total number of bands (24 timesteps * 10 original bands = 240 channels)
                dtype=data.dtype,
                crs=crs,
                transform=new_transform
            ) as dst:
                band_index = 1
                for timestep in range(data.shape[0]):
                    for band in range(data.shape[1]):
                        dst.write(data[timestep, band], band_index)
                        band_index += 1

            print(f"GeoTIFF file created at {geotiff_file_path}")           

# Define the directories
# TODO Change the directories to the correct paths
label_dir = '/data/hpate061/CalCROP21/ACCEPTABLE_GRIDS/LABEL_MAPS'
npy_dir = '/data/hpate061/CalCROP21/ACCEPTABLE_GRIDS/IMAGE_GRIDS'
output_dir = '/data/hpate061/CalCROP21/ACCEPTABLE_GRIDS/GeoTIFF'

# Process the files
process_files(label_dir, npy_dir, output_dir)


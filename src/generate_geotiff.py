'''
Script to iterate through each '.npy' file and its corresponding label GeoTIFF file, pre-process the data,
and write the output to a new GeoTIFF file. 
'''
import os
import rasterio
import numpy as np

def process_files(label_dir, npy_dir, output_dir):
    
    os.makedirs(output_dir, exist_ok=True)
    
    #sub dir for npy files
    npy_output_dir = os.path.join(output_dir, 'npy_with_labels')
    os.makedirs(npy_output_dir, exist_ok=True)

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
            data = np.load(npy_file_path)  # Shape: (24, 10, h, w)

            # Extract row and column from the npy file name
            parts = npy_file.split('_')
            row = int(parts[2]) * data.shape[2]  # Height of npy data
            col = int(parts[3]) * data.shape[3]  # Width of npy data

            # Load the transformation matrix and label data from the label file
            with rasterio.open(label_file_path) as dataset:
                transform = dataset.transform
                crs = dataset.crs
                label_data = dataset.read(1)  # Read the label data

            # Extract the corresponding label data section
            label_data_cut = label_data[row:row+data.shape[2], col:col+data.shape[3]]

            # Expand label_data_cut to match the time steps in the npy data
            label_data_expanded = np.repeat(label_data_cut[np.newaxis, np.newaxis, :, :], repeats=24, axis=0)  # Shape: (24, 1, h, w)

            # Combine the original npy data with the label data as the 11th band
            combined_data = np.concatenate((data, label_data_expanded), axis=1)  # Shape: (24, 11, h, w)

            # Save the new npy file
            npy_output_file_path = os.path.join(npy_output_dir, npy_file)
            np.save(npy_output_file_path, combined_data)
            print(f"New npy file saved at {npy_output_file_path}")

            # Write the data to a GeoTIFF file as before (updated with the combined data)
            total_bands = combined_data.shape[0] * combined_data.shape[1]  # (24 * 11)

            # geotiff file name
            geotiff_file_name = f"{os.path.splitext(npy_file)[0]}.tif"
            geotiff_file_path = os.path.join(output_dir, geotiff_file_name)

            # Write the data to a single GeoTIFF file
            with rasterio.open(
                geotiff_file_path,
                'w',
                driver='GTiff',
                height=combined_data.shape[2],  # Height of npy data
                width=combined_data.shape[3],  # Width of npy data
                count=total_bands,  # Total number of bands (24 timesteps * 11 bands)
                dtype=combined_data.dtype,
                crs=crs,
                transform=transform
            ) as dst:
                band_index = 1
                for timestep in range(combined_data.shape[0]):
                    for band in range(combined_data.shape[1]):
                        dst.write(combined_data[timestep, band], band_index)
                        band_index += 1

            print(f"GeoTIFF file created at {geotiff_file_path}")

# Define the directories
label_dir = '/data/hpate061/CalCROP21/ACCEPTABLE_GRIDS/LABEL_MAPS'
npy_dir = '/data/hpate061/CalCROP21/ACCEPTABLE_GRIDS/IMAGE_GRIDS'
output_dir = '/data/hpate061/CalCROP21/ACCEPTABLE_GRIDS/GeoTIFF'

# Process the files
process_files(label_dir, npy_dir, output_dir)



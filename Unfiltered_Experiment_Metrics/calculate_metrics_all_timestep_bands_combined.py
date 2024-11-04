import os
import numpy as np
import pandas as pd 
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, jaccard_score
import rasterio
from tqdm import tqdm

# paths
sam_output_folder = '/data2/hpate061/CalCROP21/ACCEPTABLE_GRIDS/SAM_OUTPUTS_NUMPY'
geotiff_folder = '/data/hpate061/CalCROP21/ACCEPTABLE_GRIDS/GeoTIFF'

# list to store metrics for all time steps and bands
all_metrics = []

# Function to get shapefile band from geoTiff file
def load_geotiff(geotiff_path):
    with rasterio.open(geotiff_path) as src:
        data = src.read()
        ground_truth = data[-1]  # Assuming ground truth is the last band
    return ground_truth

# Get all timestep folders
timestep_folders = [f for f in os.listdir(sam_output_folder) if f.startswith('Timestep')]
# Assume 10 bands (Band_0 to Band_9)
band_folders = [f'Band_{i}' for i in range(10)]

# Find unique tile IDs across all timesteps and bands
all_tile_ids = set()

for timestep_folder in timestep_folders:
    for band_folder in band_folders:
        band_path = os.path.join(sam_output_folder, timestep_folder, band_folder)
        tile_ids = set(f.split('_IMAGE.npy')[0] for f in os.listdir(band_path) if f.endswith('.npy'))
        if not all_tile_ids:
            all_tile_ids = tile_ids  # Initialize on first pass
        else:
            all_tile_ids.intersection_update(tile_ids)  # Ensure the tile exists across all timesteps and bands

# Process each tile across all timesteps and bands
for tile_id in tqdm(all_tile_ids, desc="Processing Tiles", unit="tile"):
    
    combined_prediction = None  # To store the ORed predictions across all timesteps and bands

    for timestep_folder in timestep_folders:
        for band_folder in band_folders:
            # Path to the current tile's numpy file
            numpy_file = os.path.join(sam_output_folder, timestep_folder, band_folder, f"{tile_id}_IMAGE.npy")
            
            # Load numpy file (SAM output for the current band at the current timestep)
            band_sam_output = np.load(numpy_file)
            
            # OR along the last axis (reduce over the last axis for this tile and band)
            band_sam_output_or = np.logical_or.reduce(band_sam_output, axis=-1)
            
            # Combine across all timesteps and bands
            if combined_prediction is None:
                combined_prediction = band_sam_output_or
            else:
                combined_prediction = np.logical_or(combined_prediction, band_sam_output_or)
    
    # Load the corresponding GeoTIFF (ground truth)
    geotiff_file = os.path.join(geotiff_folder, f"{tile_id}_IMAGE_combined.tif")
    ground_truth = load_geotiff(geotiff_file)

    # Flatten the arrays for metrics calculation
    combined_prediction_flat = combined_prediction.flatten()
    ground_truth_flat = ground_truth.flatten()
    
    # Calculate metrics
    metrics = {
        'tile_id': tile_id,
        'accuracy': accuracy_score(ground_truth_flat, combined_prediction_flat),
        'precision': precision_score(ground_truth_flat, combined_prediction_flat),
        'recall': recall_score(ground_truth_flat, combined_prediction_flat),
        'f1': f1_score(ground_truth_flat, combined_prediction_flat),
        'jaccard': jaccard_score(ground_truth_flat, combined_prediction_flat)
    }
    
    # Append the metrics to the list
    all_metrics.append(metrics)

# Convert the metrics list to a DataFrame
df = pd.DataFrame(all_metrics)
df.to_csv('metrics_all_timesteps_bands.csv', index=False)

# Calculate and save mean and std metrics
std_metrics = df.std()
mean_metrics = df.mean()

mean_metrics.to_csv('mean_metrics_all_timesteps_bands.csv')
std_metrics.to_csv('std_metrics_all_timesteps_bands.csv')

print('Metrics saved to csv files')

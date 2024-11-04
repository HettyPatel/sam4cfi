'''
script to calculate metrics for SAM output vs Shapefiles for all timesteps rgb bands
'''

import os
import numpy as np
import pandas as pd 
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, jaccard_score
import rasterio
from tqdm import tqdm

# paths
sam_output_folder = '/data2/hpate061/CalCROP21/ACCEPTABLE_GRIDS/SAM_OUTPUTS_NUMPY'
geotiff_folder = '/data/hpate061/CalCROP21/ACCEPTABLE_GRIDS/GeoTIFF'

#list to store metrics for all time steps and rgb channels

all_metrics = []

# Function to get shapefile band from geoTiff file

def load_geotiff(geotiff_path):
    with rasterio.open(geotiff_path) as src:
        data = src.read()
        ground_truth = data[-1]
    return ground_truth

# get all timesteps
timestep_folders = [f for f in os.listdir(sam_output_folder) if f.startswith('Timestep')]

for timestep_folder in tqdm(timestep_folders, desc="Processing Timesteps", unit='timestep'):
    
    timestep_path = os.path.join(sam_output_folder, timestep_folder, 'RGB')
    
    # Get numpy and geotiff files from the timestep's RGB folder
    numpy_files = [f for f in os.listdir(timestep_path) if f.endswith('.npy')]
    geotiff_files = [f for f in os.listdir(geotiff_folder) if f.endswith('.tif')]
    
    # Get IDs for matching
    numpy_ids = {f.split('_IMAGE.npy')[0]: f for f in numpy_files}
    geotiff_ids = {f.split('_IMAGE_combined.tif')[0]: f for f in geotiff_files}
    
    for tile_id in tqdm(numpy_ids.keys() & geotiff_ids.keys(), desc=f"Processing Tiles in {timestep_folder}", unit="tile"):
        
        # Load numpy files (sam output files for rgb channels at a timestep)
        numpy_file = os.path.join(timestep_path, numpy_ids[tile_id])
        rgb_sam_output = np.load(numpy_file)
        
        # Calculate OR of all segments
        combined_prediction = np.logical_or.reduce(rgb_sam_output, axis=-1)
        
        # Load geotiff files (ground truth)
        geotiff_file = os.path.join(geotiff_folder, geotiff_ids[tile_id])
        ground_truth = load_geotiff(geotiff_file)
        
        # Flatten the arrays for metrics calculation
        combined_prediction_flat = combined_prediction.flatten()
        ground_truth_flat = ground_truth.flatten()
        
        # Calculate metrics
        metrics = {
            'timestep': timestep_folder,
            'tile_id': tile_id,
            'accuracy': accuracy_score(ground_truth_flat, combined_prediction_flat),
            'precision': precision_score(ground_truth_flat, combined_prediction_flat),
            'recall': recall_score(ground_truth_flat, combined_prediction_flat),
            'f1': f1_score(ground_truth_flat, combined_prediction_flat),
            'jaccard': jaccard_score(ground_truth_flat, combined_prediction_flat)
        }
        
        # Append the metrics to the list
        all_metrics.append(metrics)


df = pd.DataFrame(all_metrics)
df.to_csv('metrics_all_timestep_rgb.csv', index=False)

std_metrics = df.groupby('timestep').std()
mean_metrics = df.groupby('timestep').mean()

print('Mean Metrics: \n', mean_metrics)
print('Standard Deviation Metrics: \n', std_metrics)

# save the mean and std metrics to csv
mean_metrics.to_csv('mean_metrics_all_timestep_rgb.csv')
std_metrics.to_csv('std_metrics_all_timestep_rgb.csv')

print('Metrics saved to csv files')


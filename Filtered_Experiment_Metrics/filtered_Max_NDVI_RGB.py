import os
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, jaccard_score
import rasterio
from tqdm import tqdm

max_ndvi_sam_output_folder = '/data2/hpate061/CalCROP21/ACCEPTABLE_GRIDS/SAM_OUTPUTS_NUMPY/Max_NDVI_RGB'
geotiff_folder = '/data/hpate061/CalCROP21/ACCEPTABLE_GRIDS/GeoTIFF'
cdl_grids_folder = '/data/hpate061/CalCROP21/ACCEPTABLE_GRIDS/CDL_GRIDS'

# Label values of the crops in the CDL (CROPS ONLY)
cdl_values_list = [1, 2, 3, 6, 21, 24, 33, 42, 49, 54, 66, 69, 72, 75, 76, 204, 208, 211, 217, 36, 37]

all_metrics = []

def load_geotiff(geotiff_path):
    with rasterio.open(geotiff_path) as src:
        data = src.read()
        ground_truth = data[-1]
    return ground_truth

# Get numpy and geotiff files from the folders
numpy_files = [f for f in os.listdir(max_ndvi_sam_output_folder) if f.endswith('.npy')]
geotiff_files = [f for f in os.listdir(geotiff_folder) if f.endswith('.tif')]
cdl_files = [f for f in os.listdir(cdl_grids_folder) if f.endswith('.npy')]

# Get ids for matching
numpy_ids = {f.split('_IMAGE.npy')[0]: f for f in numpy_files}
geotiff_ids = {f.split('_IMAGE_combined.tif')[0]: f for f in geotiff_files}
cdl_ids = {f.split('_CDL_LABEL.npy')[0]: f for f in cdl_files}

# for each tile, calculate metrics
for tile_id in tqdm(numpy_ids.keys() & geotiff_ids.keys() & cdl_ids.keys(), desc="Processing tiles"):
    
    # Load numpy files (SAM output)
    numpy_file = os.path.join(max_ndvi_sam_output_folder, numpy_ids[tile_id])
    stacked_sam_output = np.load(numpy_file)
    
    # Calculate OR of all segments
    combined_prediction = np.logical_or.reduce(stacked_sam_output, axis=-1)
    
    # Load geotiff files (ground truth)
    geotiff_file = os.path.join(geotiff_folder, geotiff_ids[tile_id])
    ground_truth = load_geotiff(geotiff_file)
    
    # Load CDL file
    cdl_file = os.path.join(cdl_grids_folder, cdl_ids[tile_id])
    cdl_data = np.load(cdl_file)
    
    # Filter crop areas
    crop_mask = np.isin(cdl_data, cdl_values_list)
    
    # Apply mask to ground truth and prediction
    combined_prediction_crops = combined_prediction[crop_mask]
    ground_truth_crops = ground_truth[crop_mask]
    
    # Flatten the arrays for metrics calculation
    combined_prediction_flat = combined_prediction_crops.flatten()
    ground_truth_flat = ground_truth_crops.flatten()
    
    # Calculate metrics
    metrics = {
        'tile_id': tile_id,
        'accuracy': accuracy_score(ground_truth_flat, combined_prediction_flat),
        'precision': precision_score(ground_truth_flat, combined_prediction_flat),
        'recall': recall_score(ground_truth_flat, combined_prediction_flat),
        'f1': f1_score(ground_truth_flat, combined_prediction_flat),
        'jaccard': jaccard_score(ground_truth_flat, combined_prediction_flat)
    }
    
    all_metrics.append(metrics)
    
# Convert results to a DataFrame
df = pd.DataFrame(all_metrics)
df.to_csv('Max_NDVI_RGB_metrics_filtered.csv', index=False)

# Calculate mean and standard deviation of metrics
std_metrics = df.std()
mean_metrics = df.mean()

print('Mean Metrics (Crops Only): \n', mean_metrics)
print('Standard Deviation Metrics (Crops Only): \n', std_metrics)

# Save mean and standard deviation to a file
mean_metrics.to_csv('Max_NDVI_RGB_mean_metrics_filtered.csv')
std_metrics.to_csv('Max_NDVI_RGB_std_metrics_filtered.csv')

print('Metrics saved to files')

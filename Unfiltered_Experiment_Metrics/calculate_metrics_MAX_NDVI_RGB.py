'''
script to calculate metrics for SAM output vs Shapefiles
'''

import os
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, jaccard_score
import rasterio

max_ndvi_sam_output_folder = '/data2/hpate061/CalCROP21/ACCEPTABLE_GRIDS/SAM_OUTPUTS_NUMPY/Max_NDVI_RGB'
geotiff_folder = '/data/hpate061/CalCROP21/ACCEPTABLE_GRIDS/GeoTIFF'


# list to store metrics for all samples
all_metrics = [] 

# func to get shapefile band from geoTIFF
def load_geotiff(geotiff_path):
    with rasterio.open(geotiff_path) as src:
        data = src.read()
        ground_truth = data[-1]
    return ground_truth

# get numpy and geotiff files from the folders
numpy_files = [f for f in os.listdir(max_ndvi_sam_output_folder) if f.endswith('.npy')]
geotiff_files = [f for f in os.listdir(geotiff_folder) if f.endswith('.tif')]

# getting ids for matching
numpy_ids = {f.split('_IMAGE.npy')[0]: f for f in numpy_files}
geotiff_ids = {f.split('_IMAGE_combined.tif')[0]: f for f in geotiff_files}

for tile_id in numpy_ids.keys() & geotiff_ids.keys():
    
    # load numpy files (sam output)
    numpy_file = os.path.join(max_ndvi_sam_output_folder, numpy_ids[tile_id])
    stacked_sam_output = np.load(numpy_file)
    
    # calculate OR of all segments
    combined_prediction = np.logical_or.reduce(stacked_sam_output, axis=-1)
    
    # load geotiff files (ground truth)
    geotiff_file = os.path.join(geotiff_folder, geotiff_ids[tile_id])
    ground_truth = load_geotiff(geotiff_file)
    
    #flatten the arrays for metrics calculation
    combined_prediction_flat = combined_prediction.flatten()
    ground_truth_flat = ground_truth.flatten()
    
    # calculate metrics
    metrics = {
        'tile_id': tile_id,
        'accuracy': accuracy_score(ground_truth_flat, combined_prediction_flat),
        'precision': precision_score(ground_truth_flat, combined_prediction_flat),
        'recall': recall_score(ground_truth_flat, combined_prediction_flat),
        'f1': f1_score(ground_truth_flat, combined_prediction_flat),
        'jaccard': jaccard_score(ground_truth_flat, combined_prediction_flat)
    }
    
    all_metrics.append(metrics)
    
   
df = pd.DataFrame(all_metrics)

df.to_csv('Max_NDVI_RGB_metrics.csv', index=False)

std_metrics = df.std()
mean_metrics = df.mean()

print('Mean Metrics: \n', mean_metrics)
print('Standard Deviation Metrics: \n', std_metrics)


    
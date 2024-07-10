import os
import torch
import numpy as np
import cv2
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import argparse
import matplotlib.pyplot as plt

'''
generate_sam_output.py
Description: 
    This script generates SAM outputs for all ACCEPTABLE_GRIDS using given parameters for the SAM AMG
Usage:
    python generate_sam_output.py 
'''

parser = argparse.ArgumentParser()

# Add arguments
parser.add_argument('--points_per_side', type=int, default=32, help='Number of points per side for AMG')
parser.add_argument('--iou_threshold', type=float, default=0.9, help='IOU threshold for SAM')
parser.add_argument('--stability_threshold', type=float, default=0.9, help='Stability score threshold for SAM AMG')
parser.add_argument('--mmra', type=int, default=100, help='minimum mask region area for SAM AMG') # for post processing holes in mask etc.
# TODO: Add argument for BANDS to be used for SAM output, 
parser.add_argument('--bands', type=list, default=[-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9], help='Bands to be used for SAM output. Use -1 for RGB bands') #-1 for max ndvi and each timestep rgb

args = parser.parse_args()

pps = args.points_per_side
iou_threshold = args.iou_threshold
stability_threshold = args.stability_threshold
mmra = args.mmra
bands = args.bands

""" 
Index    Acceptable Grid Channel Information:
   0     B2	    10 m	490 nm	    Blue
   1     B3	    10 m	560 nm	    Green
   2     B4	    10 m	665 nm	    Red
   3     B5	    20 m	705 nm	    Visible and Near Infrared (VNIR)
   4     B6	    20 m	740 nm	    Visible and Near Infrared (VNIR)
   5     B7	    20 m	783 nm	    Visible and Near Infrared (VNIR)
   6     B8	    10 m	842 nm	    Visible and Near Infrared (VNIR)
   7     B8a	20 m	865 nm	    Visible and Near Infrared (VNIR)
   8     B11	20 m	1610 nm	    Short Wave Infrared (SWIR)
   9    B12	20 m	2190 nm	    Short Wave Infrared (SWIR)
"""

ACCEPTABLE_GRIDS_PATH = '/data/hpate061/CalCROP21/ACCEPTABLE_GRIDS/IMAGE_GRIDS'
OUTPUT_FOLDER_PATH = '/data/hpate061/CalCROP21/ACCEPTABLE_GRIDS/SAM_OUTPUTS' # add SAM_OUTPUT folder path here

# If output folder does not exist, create it
if not os.path.exists(OUTPUT_FOLDER_PATH):
    os.makedirs(OUTPUT_FOLDER_PATH)
    
# Get all acceptable grids
acceptable_grids = os.listdir(ACCEPTABLE_GRIDS_PATH)
acceptable_grid_paths = [os.path.join(ACCEPTABLE_GRIDS_PATH, grid) for grid in acceptable_grids]

# SAM setup (moved outside the loop)
sam_checkpoint = "/home/hpate061/SAM_Weights/sam_vit_h_4b8939.pth"
model_type = "vit_h"
device = "cuda"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device)

mask_generator = SamAutomaticMaskGenerator(
    model=sam,
    points_per_side=pps,
    pred_iou_thresh=iou_threshold,
    stability_score_thresh=stability_threshold,
    min_mask_region_area=mmra,
)

for image_path in acceptable_grid_paths:
    # Load image
    image = np.load(image_path) # shape: (24, 10, 1098, 1098)
    
    # ======================== SAM OUTPUT FOR (RGB) MAX NDVI TIMESTEP ========================
    
    B4, B8 = image[:, 2, :, :], image[:, 6, :, :]
    image_ndvi = np.divide((B8 - B4), (B8 + B4), out=np.zeros_like(B4), where=(B8 + B4) != 0)
    image_ndvi = np.nan_to_num(image_ndvi)
    
    # Max average NDVI timestep
    average_ndvi = image_ndvi.mean(axis=(1, 2))
    max_ndvi_timestep = np.argmax(average_ndvi)
    
    max_ndvi_image = image[max_ndvi_timestep]
    max_ndvi_rgb_image = max_ndvi_image[0:3, :] # shape: (3, 1098, 1098), blue, green, red
    max_ndvi_rgb_image = max_ndvi_rgb_image.transpose(1, 2, 0) # shape: (1098, 1098, 3)
    
    # BGR to RGB, normalize, and convert to uint8
    max_ndvi_rgb_image = max_ndvi_rgb_image[:, :, [2, 1, 0]] # Convert BGR to RGB
    max_ndvi_rgb_image = (max_ndvi_rgb_image - np.min(max_ndvi_rgb_image)) / (np.max(max_ndvi_rgb_image) - np.min(max_ndvi_rgb_image))
    max_ndvi_rgb_image = (max_ndvi_rgb_image * 255).astype(np.uint8)
    
    # Generate SAM output for max NDVI timestep
    masks = mask_generator.generate(max_ndvi_rgb_image)
    sorted_masks = sorted(masks, key=lambda x: x["predicted_iou"])
    
    # Merge masks into single mask
    pred_mask = np.zeros((1098, 1098))
    for j, mask in enumerate(sorted_masks):
        pred_mask += np.where(mask['segmentation'] == True, j + 1, 0)
    pred_mask = pred_mask.astype('uint8')
    
    # Save Max_NDVI_RGB folder and save the image
    max_ndvi_rgb_folder = 'Max_NDVI_RGB'
    max_ndvi_rgb_folder_path = os.path.join(OUTPUT_FOLDER_PATH, max_ndvi_rgb_folder)
    os.makedirs(max_ndvi_rgb_folder_path, exist_ok=True)
    
    image_name = os.path.basename(image_path).replace('.npy', '')
    plt.imsave(os.path.join(max_ndvi_rgb_folder_path, f'{image_name}.png'), pred_mask, cmap='rainbow')

    # ======================== SAM OUTPUT RGB AND SINGLE BAND FOR ALL TIMESTEPS ========================
    for timestep in range(24):
        # Create timestep folder
        timestep_folder = os.path.join(OUTPUT_FOLDER_PATH, f'Timestep_{timestep}')
        os.makedirs(timestep_folder, exist_ok=True)
        
        # Save RGB image for each timestep
        image_i = image[timestep]
        rgb_image = image_i[0:3, :] # shape: (3, 1098, 1098), blue, green, red
        rgb_image = rgb_image.transpose(1, 2, 0) # shape: (1098, 1098, 3)
        
        # Convert BGR to RGB, normalize, and convert to uint8
        rgb_image = rgb_image[:, :, [2, 1, 0]]
        rgb_image = (rgb_image - np.min(rgb_image)) / (np.max(rgb_image) - np.min(rgb_image))
        rgb_image = (rgb_image * 255).astype(np.uint8)
        
        rgb_folder = os.path.join(timestep_folder, 'RGB')
        os.makedirs(rgb_folder, exist_ok=True)
        
        masks = mask_generator.generate(rgb_image)
        sorted_masks = sorted(masks, key=lambda x: x["predicted_iou"])
        
        # Merge masks into single mask
        pred_mask_rgb = np.zeros((1098, 1098))
        for k, mask in enumerate(sorted_masks):
            pred_mask_rgb += np.where(mask['segmentation'] == True, k + 1, 0)
        pred_mask_rgb = pred_mask_rgb.astype('uint8')
        
        plt.imsave(os.path.join(rgb_folder, f'{image_name}.png'), pred_mask_rgb, cmap='rainbow')

        # Save single band images for each timestep
        for band in bands:
            if band != -1: # Skip if band is -1 (RGB bands already handled)
                single_band_image = image_i[band, :]
                single_band_image = np.expand_dims(single_band_image, axis=0) # shape: (1, 1098, 1098)
                single_band_image = single_band_image.transpose(1, 2, 0)
                single_band_image = cv2.cvtColor(single_band_image, cv2.COLOR_GRAY2RGB)
                
                single_band_image_normalized = (single_band_image - np.min(single_band_image)) / (np.max(single_band_image) - np.min(single_band_image))
                single_band_image_normalized = (single_band_image_normalized * 255).astype(np.uint8)
                
                masks = mask_generator.generate(single_band_image_normalized)
                sorted_masks = sorted(masks, key=lambda x: x["predicted_iou"])
                
                # Merge masks into single mask
                pred_mask_band = np.zeros((1098, 1098))
                for l, mask in enumerate(sorted_masks):
                    pred_mask_band += np.where(mask['segmentation'] == True, l + 1, 0)
                pred_mask_band = pred_mask_band.astype('uint8')
                
                single_band_folder = os.path.join(timestep_folder, f'Band_{band}')
                os.makedirs(single_band_folder, exist_ok=True)
                
                plt.imsave(os.path.join(single_band_folder, f'{image_name}.png'), pred_mask_band, cmap='rainbow')
                
print('SAM outputs generated successfully!')

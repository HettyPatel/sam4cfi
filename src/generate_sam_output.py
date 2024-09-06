import os
import torch
import numpy as np
import cv2
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import argparse
from multiprocessing import Pool, current_process
from tqdm import tqdm

parser = argparse.ArgumentParser()

# Add arguments
parser.add_argument('--points_per_side', type=int, default=32, help='Number of points per side for AMG')
parser.add_argument('--iou_threshold', type=float, default=0.95, help='IOU threshold for SAM')
parser.add_argument('--stability_threshold', type=float, default=0.9, help='Stability score threshold for SAM AMG')
parser.add_argument('--mmra', type=int, default=100, help='minimum mask region area for SAM AMG')
parser.add_argument('--bands', type=list, default=[-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9], help='Bands to be used for SAM output. Use -1 for RGB bands')
parser.add_argument('--gpus', type=list, default=[0, 2, 3, 4, 5, 6], help='List of specific GPUs to use')
parser.add_argument('--procs_per_gpu', type=int, default=2, help='Number of processes per GPU')

args = parser.parse_args()

# Extract arguments
pps = args.points_per_side
iou_threshold = args.iou_threshold
stability_threshold = args.stability_threshold
mmra = args.mmra
bands = args.bands
gpus = args.gpus  # List of GPUs to use
procs_per_gpu = args.procs_per_gpu  # Processes per GPU

# Output paths
ACCEPTABLE_GRIDS_PATH = '/data/hpate061/CalCROP21/ACCEPTABLE_GRIDS/IMAGE_GRIDS'
OUTPUT_FOLDER_PATH = '/data2/hpate061/CalCROP21/ACCEPTABLE_GRIDS/SAM_OUTPUTS_NUMPY'

# Create output folder if it doesn't exist
if not os.path.exists(OUTPUT_FOLDER_PATH):
    os.makedirs(OUTPUT_FOLDER_PATH)

# Get all grid paths
acceptable_grids = os.listdir(ACCEPTABLE_GRIDS_PATH)
acceptable_grid_paths = [os.path.join(ACCEPTABLE_GRIDS_PATH, grid) for grid in acceptable_grids]

sam_checkpoint = "/home/hpate061/SAM_Weights/sam_vit_h_4b8939.pth"
model_type = "vit_h"

def process_image(image_path):
    # Get the current process ID and GPU assignment
    process_id = current_process()._identity[0] - 1  # 0-indexed process ID
    gpu_id = gpus[process_id % len(gpus)]  # Cycle through the specific GPUs based on process ID
    device = f'cuda:{gpu_id}'

    # Load the SAM model
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam = torch.nn.DataParallel(sam)
    sam.to(device)

    # Create mask generator
    mask_generator = SamAutomaticMaskGenerator(
        model=sam.module,  # Access the original model through module
        points_per_side=pps,
        pred_iou_thresh=iou_threshold,
        stability_score_thresh=stability_threshold,
        min_mask_region_area=mmra,
    )

    # Load the image data and process it
    image = np.load(image_path)

    # Calculate NDVI and get the timestep with the maximum average NDVI
    B4, B8 = image[:, 2, :, :], image[:, 6, :, :]
    image_ndvi = np.divide((B8 - B4), (B8 + B4), out=np.zeros_like(B4), where=(B8 + B4) != 0)
    image_ndvi = np.nan_to_num(image_ndvi)
    max_ndvi_timestep = np.argmax(image_ndvi.mean(axis=(1, 2)))

    # Get the RGB image for the timestep with the maximum average NDVI
    max_ndvi_rgb_image = image[max_ndvi_timestep][0:3, :, :].transpose(1, 2, 0)
    max_ndvi_rgb_image = (max_ndvi_rgb_image - np.min(max_ndvi_rgb_image)) / (np.max(max_ndvi_rgb_image) - np.min(max_ndvi_rgb_image))
    max_ndvi_rgb_image = (max_ndvi_rgb_image * 255).astype(np.uint8)

    # Generate masks for Max NDVI RGB
    masks = mask_generator.generate(max_ndvi_rgb_image)
    mask_cube = np.zeros((1098, 1098, len(masks)))
    for k, mask in enumerate(masks):
        mask_cube[:, :, k] = mask['segmentation']
    mask_cube = mask_cube.astype('uint8')

    # Save the Max NDVI RGB image as numpy
    image_name = os.path.basename(image_path).replace('.npy', '')
    max_ndvi_rgb_folder = os.path.join(OUTPUT_FOLDER_PATH, 'Max_NDVI_RGB')
    os.makedirs(max_ndvi_rgb_folder, exist_ok=True)
    np.save(os.path.join(max_ndvi_rgb_folder, f'{image_name}.npy'), mask_cube)
    print(f'Saved Max_NDVI_RGB image for {image_name}')

    # Process all timesteps and bands
    for timestep in range(24):
        timestep_folder = os.path.join(OUTPUT_FOLDER_PATH, f'Timestep_{timestep}')
        os.makedirs(timestep_folder, exist_ok=True)

        image_i = image[timestep]
        rgb_image = image_i[0:3, :, :].transpose(1, 2, 0)
        rgb_image = (rgb_image - np.min(rgb_image)) / (np.max(rgb_image) - np.min(rgb_image))
        rgb_image = (rgb_image * 255).astype(np.uint8)

        # Save RGB for each timestep
        rgb_folder = os.path.join(timestep_folder, 'RGB')
        os.makedirs(rgb_folder, exist_ok=True)
        masks = mask_generator.generate(rgb_image)
        mask_cube = np.zeros((1098, 1098, len(masks)))
        for l, mask in enumerate(masks):
            mask_cube[:, :, l] = mask['segmentation']
        mask_cube = mask_cube.astype('uint8')
        np.save(os.path.join(rgb_folder, f'{image_name}.npy'), mask_cube)
        print(f'Saved RGB image for timestep {timestep} for {image_name}')

        # Process and save each band for the current timestep
        for band in bands:
            if band != -1:
                single_band_image = image_i[band, :, :]
                single_band_image = np.expand_dims(single_band_image, axis=-1)  # Add a new axis
                single_band_image = cv2.cvtColor(single_band_image, cv2.COLOR_GRAY2RGB)

                single_band_image = (single_band_image - np.min(single_band_image)) / (np.max(single_band_image) - np.min(single_band_image))
                single_band_image = (single_band_image * 255).astype(np.uint8)

                masks = mask_generator.generate(single_band_image)
                mask_cube = np.zeros((1098, 1098, len(masks)))
                for m, mask in enumerate(masks):
                    mask_cube[:, :, m] = mask['segmentation']
                mask_cube = mask_cube.astype('uint8')

                band_folder = os.path.join(timestep_folder, f'Band_{band}')
                os.makedirs(band_folder, exist_ok=True)
                np.save(os.path.join(band_folder, f'{image_name}.npy'), mask_cube)
                print(f'Saved Band_{band} image for timestep {timestep} for {image_name}')

if __name__ == "__main__":
    total_processes = procs_per_gpu * len(gpus)
    
    # Initialize tqdm progress bar
    with Pool(processes=total_processes) as pool:
        for _ in tqdm(pool.imap(process_image, acceptable_grid_paths), total=len(acceptable_grid_paths)):
            pass
    
    print('SAM outputs generated successfully!')

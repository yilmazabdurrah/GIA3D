"""
Main Script by
Abdurrahman Yilmaz (ayilmaz@lincoln.ac.uk) v01 for getting Matterport Dataset RGBD Data 
18 Apr 2025
"""

import os
import argparse
import zipfile
from PIL import Image
from util import *

def extract_rgbd_per_region(scene_name, rgb_zip, depth_zip, save_path):
    """
    Extract RGB and depth images for each region from Matterport zip files.
    """
    rgb_files = [f for f in rgb_zip.namelist() if f.endswith(('.jpg', '.jpeg', '.png'))]
    depth_files = [f for f in depth_zip.namelist() if f.endswith(('.png', '.tiff'))]

    print(f"Found {len(rgb_files)} RGB files and {len(depth_files)} depth files")

    for file_list, zip_file, subfolder in [
        (rgb_files, rgb_zip, "region_rgb_images"),
        (depth_files, depth_zip, "region_depth_images"),
    ]:
        for file_path in file_list:
            region_name = file_path.split('/')[0]  # e.g., 'region0'
            filename = os.path.basename(file_path)
            target_dir = os.path.join(save_path, scene_name, subfolder, region_name)
            os.makedirs(target_dir, exist_ok=True)

            target_file_path = os.path.join(target_dir, filename)
            if os.path.exists(target_file_path):
                print(f"Skipping existing file: {target_file_path}")
                continue

            with zip_file.open(file_path) as source_file:
                data = source_file.read()
                with open(target_file_path, 'wb') as out_file:
                    out_file.write(data)

            print(f"Saved {file_path} -> {target_file_path}")

def get_args():
    '''Command line arguments.'''
    parser = argparse.ArgumentParser(description='Segment Anything on Matterport dataset.')
    parser.add_argument('--scans_file_path', type=str, required=True, help='Path to scans.txt file containing scene names')
    parser.add_argument('--scans_root', type=str, required=True, help='Root path to the Matterport dataset scans folder')
    parser.add_argument('--save_path', type=str, required=True, help='Directory where to save the PCD results')
    parser.add_argument('--save_2dmask_path', type=str, default='', help='Directory where to save 2D segmentation results')
    parser.add_argument('--sam_checkpoint_path', type=str, required=True, help='Path to the SAM checkpoint')
    parser.add_argument('--voxel_size', type=float, default=0.05, help='Voxel size for downsampling')
    parser.add_argument('--th', type=int, default=50, help='Threshold for ignoring small groups to reduce noise')
    parser.add_argument('--category_mapping_path', type=str, default='', help='Path to Matterport dataset semantic class id mapping')
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()

    os.makedirs(args.save_path, exist_ok=True)

    # Load scene names from scans.txt
    with open(args.scans_file_path, 'r') as scans_file:
        scene_names = scans_file.read().splitlines()

    for scene_name in scene_names:
        scene_path = os.path.join(args.scans_root, scene_name)
        
        if not os.path.isdir(scene_path):
            print(f"Warning: Scene folder not found for {scene_name}")
            continue
        
        # Extract relevant data from zipped files within each scene
        with zipfile.ZipFile(os.path.join(scene_path, 'undistorted_color_images.zip'), 'r') as rgb_zip, \
             zipfile.ZipFile(os.path.join(scene_path, 'undistorted_depth_images.zip'), 'r') as depth_zip, \
             zipfile.ZipFile(os.path.join(scene_path, 'region_segmentations.zip'), 'r') as gt_region_zip, \
             zipfile.ZipFile(os.path.join(scene_path, 'house_segmentations.zip'), 'r') as gt_scene_zip   :
            
            print(f"Processing scene: {scene_name}")
            
            # RGBD data extraction
            extract_rgbd_per_region(scene_name, rgb_zip, depth_zip, args.save_path)
import numpy as np
import glob
import torch
import os
import json
from util import *

def generate_unique_color(used_colors):
    while True:
        # Generate a random color
        color = tuple(np.random.rand(3))
        # Ensure it's not black and not already used
        if color != (0, 0, 0) and color not in used_colors:
            used_colors.add(color)
            return color

# Define base directory paths
base_dir = '/home/ayilmaz/ws_segment_3d/SegmentAnything3D/'
gt_dir = '/media/ayilmaz/Crucial X9/ScanNetv2_Dataset/gt/processed/'
output_dir = os.path.join(base_dir, 'ScanNetv2_Downloads/gt_PCs/')

# Processed data paths
processed_files = os.listdir(gt_dir)

nyu40_color_to_class_id = {v["id"]: k for k, v in nyu40_colors_to_class.items()}
ScanNet20_color_to_class_id = {v["index"]: k for k, v in ScanNet20_colors_to_class.items()}
ScanNet200_color_to_class_id = {v["index"]: k for k, v in ScanNet200_colors_to_class.items()}
#print(set(ScanNet200_color_to_class_id))

nyu40_color_to_class_id_list = list(nyu40_color_to_class_id)
ScanNet20_color_to_class_id_list = list(ScanNet20_color_to_class_id)
ScanNet200_color_to_class_id_list = list(ScanNet200_color_to_class_id)

for filename in processed_files:
    if filename.endswith('_gt.pth'):
        # Load point cloud data
        print("filename: ", filename)
        pcd_files = glob.glob(os.path.join(gt_dir, filename))
        print("pcd_files: ", pcd_files)
        pcd_data = torch.load(pcd_files[0])

        print(pcd_data.keys())

        # Get the coordinates and colors
        coord = pcd_data['coord'].astype('float64')

        # Save point cloud with semantic_gt20 labels
        labels_gt20 = pcd_data.get("labels_gt20", None)
        colors_gt20 = pcd_data.get("colors_gt20", None)

        if labels_gt20 is not None:
            colors_gt20 =  np.array(colors_gt20)/255.0
            pcd_seg_file_savepath = os.path.join(output_dir, f'{filename[:-4]}_semantic_gt20.ply')
            visualize_pcd(coord, colors_gt20, colors_gt20, pcd_seg_file_savepath, True)
            print(f"Saved: {pcd_seg_file_savepath}")

        # Save point cloud with semantic_gt200 labels
        labels_gt200 = pcd_data.get("labels_gt200", None)
        colors_gt200 = pcd_data.get("colors_gt200", None)
        if labels_gt200 is not None:
            colors_gt200 = np.array(colors_gt200)/255.0
            pcd_seg_file_savepath = os.path.join(output_dir, f'{filename[:-4]}_semantic_gt200.ply')
            visualize_pcd(coord, colors_gt200, colors_gt200, pcd_seg_file_savepath, True)
            print(f"Saved: {pcd_seg_file_savepath}")

        # Save point cloud with instance_gt labels
        labels_gt_instance = pcd_data.get("labels_gt_instance", None)
        colors_gt_instance = pcd_data.get("colors_gt_instance", None)
        if labels_gt_instance is not None:
            colors_gt_instance = np.array(colors_gt_instance)/255.0
            pcd_seg_file_savepath = os.path.join(output_dir, f'{filename[:-4]}_instance_gt.ply')
            visualize_pcd(coord, colors_gt_instance, colors_gt_instance, pcd_seg_file_savepath, True)
            print(f"Saved: {pcd_seg_file_savepath}")
        
        # Save point cloud with nyu labels
        labels_gt_nyu = pcd_data.get("labels_nyu", None)
        colors_gt_nyu = pcd_data.get("colors_nyu", None)
        if labels_gt_nyu is not None:
            colors_gt_nyu = np.array(colors_gt_nyu)/255.0
            pcd_seg_file_savepath = os.path.join(output_dir, f'{filename[:-4]}_nyu_gt.ply')
            visualize_pcd(coord, colors_gt_nyu, colors_gt_nyu, pcd_seg_file_savepath, True)
            print(f"Saved: {pcd_seg_file_savepath}")

        

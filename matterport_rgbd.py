"""
Main Script by
Abdurrahman Yilmaz (ayilmaz@lincoln.ac.uk) v01 for getting Matterport Dataset RGBD Data 
18 Apr 2025
"""

import os
import argparse
import zipfile
import tempfile
import re
from PIL import Image
from util import *
import numpy as np
import cv2

from segment_anything import build_sam, SamAutomaticMaskGenerator

def parse_conf_file(conf_data):
    """Parse the .conf file to extract intrinsics matrices and scan entries."""
    intrinsics_list = []
    scan_entries = []
    current_intrinsics = None
    
    for line in conf_data.splitlines():
        line = line.strip()
        if line.startswith("intrinsics_matrix"):
            # Parse intrinsics matrix (3x3)
            values = list(map(float, line.split()[1:]))
            intrinsics = np.array(values).reshape(3, 3)
            current_intrinsics = intrinsics
            intrinsics_list.append(intrinsics)
        elif line.startswith("scan"):
            # Parse scan entry
            parts = line.split()
            depth_file, color_file = parts[1], parts[2]
            # Extract pose (4x4 matrix) from remaining values
            pose_values = list(map(float, parts[3:]))
            pose = np.array(pose_values).reshape(4, 4)
            scan_entries.append({
                "depth_file": depth_file,
                "color_file": color_file,
                "pose": pose,
                "intrinsics": current_intrinsics
            })
    
    return intrinsics_list, scan_entries

def get_sam(image, mask_generator):
    masks = mask_generator.generate(image)
    #print(masks)
    group_ids = np.full((image.shape[0], image.shape[1]), -1, dtype=int)
    stability_scores = np.full((image.shape[0], image.shape[1]), 0.0, dtype=float)
    predicted_ious = np.full((image.shape[0], image.shape[1]), 0.0, dtype=float)
    feature_shape = len(masks[0]["features"])
    features = np.zeros((image.shape[0], image.shape[1], feature_shape), dtype=float)
    num_masks = len(masks)
    group_counter = 0
    for i in reversed(range(num_masks)):
        group_ids[masks[i]["segmentation"]] = group_counter
        stability_scores[masks[i]["segmentation"]] = masks[i]["stability_score"]
        predicted_ious[masks[i]["segmentation"]] = masks[i]["predicted_iou"]
        features[masks[i]["segmentation"]] = masks[i]["features"]
        group_counter += 1
    
    # with np.printoptions(threshold=np.inf): # print all values

    return group_ids, stability_scores, predicted_ious, features

def get_matterport_pcd(scene_path, color_img_path, depth_img_path, intrinsics_path, pose_path, mask_generator=None):
    
    depth_img = cv2.imread(depth_img_path, -1).astype(np.float32) / 1000.0  # mm to m
    color_img = cv2.imread(color_img_path)
    h, w = depth_img.shape

    # Load intrinsics
    depth_intrinsic = np.loadtxt(intrinsics_path)[:3, :3]  # (3,3) matrix
    fx, fy = depth_intrinsic[0,0], depth_intrinsic[1,1]
    cx, cy = depth_intrinsic[0,2], depth_intrinsic[1,2]

    # Load pose
    pose = np.loadtxt(pose_path)  # (4,4) matrix

    # Get mask (non-zero depth)
    mask = (depth_img > 0)

    # Create 2D pixel grid
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    x, y = x[mask], y[mask]
    depth = depth_img[mask]

    # Unproject to camera coordinates
    X = (x - cx) * depth / fx
    Y = (y - cy) * depth / fy
    Z = depth

    points = np.stack((X, Y, Z, np.ones_like(Z)), axis=1)

    # Transform to world coordinates
    points_world = (points @ pose.T)[:, :3]

    # Get RGB values
    color = color_img[mask]
    colors = np.zeros_like(color)
    colors[:, 0] = color[:, 2]
    colors[:, 1] = color[:, 1]
    colors[:, 2] = color[:, 0]

    if mask_generator is not None:
        seg_input = cv2.resize(color_img, (1280, 1024))
        group_ids, stability_scores, predicted_ious, features = get_sam(seg_input, mask_generator)
        group_ids = group_ids[mask]
        stability_scores = stability_scores[mask]
        predicted_ious = predicted_ious[mask]
        features = features[mask]
    else:
        n = points_world.shape[0]
        group_ids = np.zeros(n)
        stability_scores = np.zeros(n)
        predicted_ious = np.zeros(n)
        features = np.zeros((n, 256))  # assuming 256-dim features

    group_ids = num_to_natural(group_ids)

    save_dict = dict(
        coord=points_world,
        color=colors,
        group=group_ids,
        stability_score=stability_scores,
        predicted_iou=predicted_ious,
        feature=features
    )

    return save_dict

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

    mask_generator = SamAutomaticMaskGenerator(build_sam(checkpoint=args.sam_checkpoint_path).to(device="cuda"))

    # Load scene names from scans.txt
    with open(args.scans_file_path, 'r') as scans_file:
        scene_names = scans_file.read().splitlines()

    for scene_name in scene_names:
        scene_path = os.path.join(args.scans_root, scene_name)
        
        if not os.path.isdir(scene_path):
            print(f"Warning: Scene folder not found for {scene_name}")
            continue
        
        print(f"Processing scene: {scene_name}")
        
        # Extract relevant data from zipped files within each scene
        with zipfile.ZipFile(os.path.join(scene_path, 'undistorted_color_images.zip'), 'r') as rgb_zip, \
             zipfile.ZipFile(os.path.join(scene_path, 'undistorted_depth_images.zip'), 'r') as depth_zip, \
             zipfile.ZipFile(os.path.join(scene_path, 'undistorted_camera_parameters.zip'), 'r') as camera_param_zip, \
             zipfile.ZipFile(os.path.join(scene_path, 'matterport_camera_poses.zip'), 'r') as camera_poses_zip   :
            
            # RGBD data extraction
            #extract_rgbd_per_region(scene_name, rgb_zip, depth_zip, args.save_path)

            # Read the .conf file from camera parameters zip
            conf_file_name = f"{scene_name}/undistorted_camera_parameters/{scene_name}.conf"
            with camera_param_zip.open(conf_file_name) as conf_file:
                conf_data = conf_file.read().decode('utf-8')
                _, scan_entries = parse_conf_file(conf_data)

            # Process each scan entry
            for idx, entry in enumerate(scan_entries):
                depth_file = f"{scene_name}//undistorted_depth_images/{entry['depth_file']}"
                color_file = f"{scene_name}//undistorted_color_images/{entry['color_file']}"
                pose_file = f"{scene_name}//matterport_camera_poses/{entry['depth_file'].replace('_d', '_pose')}.txt"

                try:
                    # Create temporary files to store extracted data
                    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_depth, \
                         tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_color, \
                         tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as tmp_pose, \
                         tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as tmp_intrinsics:

                        # Extract depth image
                        with depth_zip.open(depth_file) as depth_data:
                            tmp_depth.write(depth_data.read())
                        
                        # Extract color image
                        with rgb_zip.open(color_file) as color_data:
                            tmp_color.write(color_data.read())
                        
                        # Save pose to temporary file
                        np.savetxt(tmp_pose.name, entry['pose'])
                        
                        # Save intrinsics to temporary file
                        np.savetxt(tmp_intrinsics.name, entry['intrinsics'])

                        # Call get_matterport_pcd
                        save_dict = get_matterport_pcd(
                            scene_path=scene_path,
                            color_img_path=tmp_color.name,
                            depth_img_path=tmp_depth.name,
                            intrinsics_path=tmp_intrinsics.name,
                            pose_path=tmp_pose.name,
                            mask_generator=mask_generator
                        )

                        # Save the result
                        #save_filename = os.path.join(args.save_path, f"{scene_name}_view_{idx}.npy")
                        #np.save(save_filename, save_dict)
                        print(f"Point cloud processed for view {idx} in {save_dict}")

                        # Clean up temporary files
                        os.unlink(tmp_depth.name)
                        os.unlink(tmp_color.name)
                        os.unlink(tmp_pose.name)
                        os.unlink(tmp_intrinsics.name)

                except Exception as e:
                    print(f"Error processing view {idx} for scene {scene_name}: {e}")
                    continue
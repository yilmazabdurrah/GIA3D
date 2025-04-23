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
    """
    Generate a 3D point cloud from Matterport RGB-D data.
    
    Args:
        scene_path (str): Path to the scene directory.
        color_img_path (str): Path to the undistorted color image (JPG).
        depth_img_path (str): Path to the undistorted depth image (16-bit PNG, 0.25mm units).
        intrinsics_path (str): Path to the intrinsics file (3x3 matrix).
        pose_path (str): Path to the camera-to-world pose file (4x4 matrix).
        mask_generator: Optional segmentation model for generating masks.
    
    Returns:
        dict: Dictionary containing point cloud data (coord, color, group, stability_score, predicted_iou, feature).
    """
    # Load images
    depth_img = cv2.imread(depth_img_path, -1).astype(np.float64)
    color_img = cv2.imread(color_img_path)
    h, w = depth_img.shape

    # Depth scaling (Matterport: 0.25mm per unit, divide by 4000 to get meters)
    depth_scale = 4000.0
    depth_img = depth_img / depth_scale

    # Load intrinsics
    depth_intrinsic = np.loadtxt(intrinsics_path)
    if depth_intrinsic.shape != (3, 3):
        depth_intrinsic = depth_intrinsic[:3, :3]
    fx, fy = depth_intrinsic[0, 0], depth_intrinsic[1, 1]
    cx, cy = depth_intrinsic[0, 2], depth_intrinsic[1, 2]

    # Load pose (camera-to-world, no inversion needed)
    pose = np.loadtxt(pose_path)  # 4x4 matrix

    # Get mask (non-zero depth)
    mask = (depth_img > 0)

    # Create 2D pixel grid (bottom-left origin)
    x, y = np.meshgrid(np.arange(w), np.arange(h), indexing='xy')
    x, y = x[mask], y[mask]
    depth = depth_img[mask]

    # Unproject to camera coordinates
    Z = -depth
    X = (x - cx) * depth / fx
    y = h - 1 - y
    Y = (y - cy) * depth / fy

    points = np.stack((X, Y, Z, np.ones_like(Z)), axis=1, dtype=np.float64)

    # Transform to world coordinates (Z-up)
    points_world = (points @ pose.T)[:, :3]

    # Get RGB values
    color = color_img[mask]
    colors = np.zeros_like(color)
    colors[:, 0] = color[:, 2]  # BGR to RGB
    colors[:, 1] = color[:, 1]
    colors[:, 2] = color[:, 0]

    # Segmentation (if mask_generator is provided)
    if mask_generator is not None:
        seg_input = cv2.resize(color_img, (1280, 1024))
        group_ids, stability_scores, predicted_ious, features = get_sam(seg_input, mask_generator)
        # Resize segmentation mask to original resolution, accounting for bottom-left origin
        group_ids = cv2.resize(group_ids, (w, h), interpolation=cv2.INTER_NEAREST)[mask]
        stability_scores = stability_scores[mask]
        predicted_ious = predicted_ious[mask]
        features = features[mask]
    else:
        n = points_world.shape[0]
        group_ids = np.zeros(n)
        stability_scores = np.zeros(n)
        predicted_ious = np.zeros(n)
        features = np.zeros((n, 256))  # Assuming 256-dim features

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
             zipfile.ZipFile(os.path.join(scene_path, 'undistorted_camera_parameters.zip'), 'r') as camera_param_zip:
            
            # Read the .conf file from camera parameters zip
            conf_file_name = f"{scene_name}/undistorted_camera_parameters/{scene_name}.conf"
            with camera_param_zip.open(conf_file_name) as conf_file:
                conf_data = conf_file.read().decode('utf-8')
                _, scan_entries = parse_conf_file(conf_data)

            # Process each scan entry
            for idx, entry in enumerate(scan_entries):
                depth_file = f"{scene_name}//undistorted_depth_images/{entry['depth_file']}"
                color_file = f"{scene_name}//undistorted_color_images/{entry['color_file']}"

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

                        save_dict = get_matterport_pcd(
                            scene_path=scene_path,
                            color_img_path=tmp_color.name,
                            depth_img_path=tmp_depth.name,
                            intrinsics_path=tmp_intrinsics.name,
                            pose_path=tmp_pose.name,
                            mask_generator=mask_generator
                        )

                        pcd = o3d.geometry.PointCloud()
                        pcd.points = o3d.utility.Vector3dVector(save_dict["coord"])
                        pcd.colors = o3d.utility.Vector3dVector(save_dict["color"] / 255.0)  # normalize RGB to [0,1]

                        filename_base = os.path.splitext(entry['color_file'])[0]

                        # Save PLY file
                        ply_filename = os.path.join(args.save_path, f"{scene_name}_{filename_base}.ply")
                        o3d.io.write_point_cloud(ply_filename, pcd)
                        print(f"Saved point cloud as {ply_filename}")

                        color_img = cv2.imread(tmp_color.name)
                        color_save_path = os.path.join(args.save_path, f"{scene_name}_{filename_base}.png")
                        cv2.imwrite(color_save_path, color_img)
                        print(f"Saved color image as {color_save_path}")

                        # Save the result
                        #save_filename = os.path.join(args.save_path, f"{scene_name}_view_{idx}.npy")
                        #np.save(save_filename, save_dict)
                        print(f"Point cloud processed for view {idx} in a dictionary")

                        # Clean up temporary files
                        os.unlink(tmp_depth.name)
                        os.unlink(tmp_color.name)
                        os.unlink(tmp_pose.name)
                        os.unlink(tmp_intrinsics.name)

                except Exception as e:
                    print(f"Error processing view {idx} for scene {scene_name}: {e}")
                    continue
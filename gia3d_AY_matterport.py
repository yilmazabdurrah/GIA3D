"""
Main Script

Author: Yunhan Yang (yhyang.myron@gmail.com)

Updated by
Abdurrahman Yilmaz (ayilmaz@lincoln.ac.uk) v08 for Matterport Dataset
12 Jan 2025
"""

import os
import cv2
import numpy as np
import open3d as o3d
import torch
import copy
import multiprocessing as mp
import pointops
import argparse
import zipfile
import tempfile
import csv

import itertools

import pickle

from segment_anything import build_sam, SamAutomaticMaskGenerator
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
from PIL import Image
from os.path import join
from util import *

import networkx as nx
import matplotlib.pyplot as plt

from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

import gc
import math

verbose = False # To print out intermediate data
verbose_graph = False # To plot correspondence graphs before and after optimization
verbose_comparisons = False # To plot comparison output

maxVal = 0
maxVal_stab = 0
maxVal_pred = 0
maxVal_iou = 0

MATTERPORT_DEPTH_SCALE = 4000.0

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

def get_camera_parameters(camera_param_file, panorama_uuid, camera_index, yaw_index):
    """
    Parses camera parameters from the undistorted_camera_parameters file.

    Args:
        camera_param_file (str): Path to the camera parameter file.
        panorama_uuid (str): Panorama UUID.
        camera_index (int): Camera index.
        yaw_index (int): Yaw index.

    Returns:
        dict: Intrinsic and extrinsic parameters.
    """
    intrinsics = {}
    extrinsics = []
    with open(camera_param_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith(f"scan {panorama_uuid}_d{camera_index}_{yaw_index}"):
                data = line.strip().split()
                # Extract intrinsics
                fx, fy, cx, cy = map(float, data[3:7])
                intrinsics.update({'fx': fx, 'fy': fy, 'cx': cx, 'cy': cy, 'width': int(data[-6]), 'height': int(data[-5])})
                # Extract extrinsics (16 values)
                extrinsics = np.array(data[7:23], dtype=float).reshape((4, 4))
                break

    if not extrinsics:
        raise ValueError(f"Camera parameters for {panorama_uuid}_d{camera_index}_{yaw_index} not found.")

    return intrinsics, extrinsics

def get_pcd_matterport(house_id, panorama_uuid, camera_index, yaw_index, base_path):
    """
    Reads the depth and RGB image for the given Matterport dataset identifiers,
    and returns a point cloud in global coordinates.

    Args:
        house_id (str): House identifier.
        panorama_uuid (str): Unique panorama identifier.
        camera_index (int): Camera index (0-2).
        yaw_index (int): Yaw index (0-5).
        base_path (str): Root path to the Matterport dataset.

    Returns:
        open3d.geometry.PointCloud: The generated point cloud.
    """
    depth_filename = os.path.join(
        base_path,
        house_id,
        "undistorted_depth_images",
        f"{panorama_uuid}_d{camera_index}_{yaw_index}.png"
    )
    color_filename = os.path.join(
        base_path,
        house_id,
        "undistorted_color_images",
        f"{panorama_uuid}_i{camera_index}_{yaw_index}.jpg"
    )
    camera_param_file = os.path.join(
        base_path, house_id, "undistorted_camera_parameters.txt"
    )

    if not os.path.exists(depth_filename) or not os.path.exists(color_filename):
        raise FileNotFoundError(f"Missing depth or color image for {house_id}, {panorama_uuid}")

    depth_image = o3d.io.read_image(depth_filename)
    color_image = o3d.io.read_image(color_filename)

    # Load camera parameters for the specific image
    intrinsics, extrinsics = get_camera_parameters(camera_param_file, panorama_uuid, camera_index, yaw_index)

    # Convert images to point cloud
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_image, depth_image, depth_scale=MATTERPORT_DEPTH_SCALE, convert_rgb_to_intensity=False
    )
    camera_intrinsics = o3d.camera.PinholeCameraIntrinsic(
        intrinsics['width'],
        intrinsics['height'],
        intrinsics['fx'],
        intrinsics['fy'],
        intrinsics['cx'],
        intrinsics['cy'],
    )
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image, camera_intrinsics
    )
    pcd.transform(extrinsics)  # Apply transformation to global coordinates
    return pcd

def get_pcd(scene_name, color_name, rgb_path, mask_generator, save_2dmask_path):
    # Define possible paths for the intrinsic file
    intrinsic_path1 = os.path.join(rgb_path, scene_name, 'intrinsics_depth.txt')
    intrinsic_path2 = os.path.join(rgb_path, scene_name, 'intrinsics', 'intrinsic_depth.txt')

    # Check which path exists
    if os.path.exists(intrinsic_path1):
        intrinsic_path = intrinsic_path1
    elif os.path.exists(intrinsic_path2):
        intrinsic_path = intrinsic_path2
    else:
        print(f"Intrinsic file not found in either {intrinsic_path1} or {intrinsic_path2}")

    # Load the intrinsic file
    depth_intrinsic = np.loadtxt(intrinsic_path)

    pose = join(rgb_path, scene_name, 'pose', color_name[0:-4] + '.txt')
    depth = join(rgb_path, scene_name, 'depth', color_name[0:-4] + '.png')
    color = join(rgb_path, scene_name, 'color', color_name)

    depth_img = cv2.imread(depth, -1) # read 16bit grayscale image
    mask = (depth_img != 0)
    color_image = cv2.imread(color)
    color_image = cv2.resize(color_image, (640, 480))

    save_2dmask_path = join(save_2dmask_path, scene_name)
    if mask_generator is not None:
        group_ids, stability_scores, predicted_ious, features = get_sam(color_image, mask_generator)
        if not os.path.exists(save_2dmask_path):
            os.makedirs(save_2dmask_path)
        img = Image.fromarray(num_to_natural(group_ids).astype(np.int16), mode='I;16')
        img.save(join(save_2dmask_path, color_name[0:-4] + '.png'))
    else:
        group_path = join(save_2dmask_path, color_name[0:-4] + '.png')
        img = Image.open(group_path)
        group_ids = np.array(img, dtype=np.int16)
    
    color_image = np.reshape(color_image[mask], [-1,3])
    group_ids = group_ids[mask]
    stability_scores = stability_scores[mask]
    predicted_ious = predicted_ious[mask]
    features = features[mask]
    colors = np.zeros_like(color_image)
    colors[:,0] = color_image[:,2]
    colors[:,1] = color_image[:,1]
    colors[:,2] = color_image[:,0]

    pose = np.loadtxt(pose)
    
    depth_shift = 1000.0
    x,y = np.meshgrid(np.linspace(0,depth_img.shape[1]-1,depth_img.shape[1]), np.linspace(0,depth_img.shape[0]-1,depth_img.shape[0]))
    uv_depth = np.zeros((depth_img.shape[0], depth_img.shape[1], 3))
    uv_depth[:,:,0] = x
    uv_depth[:,:,1] = y
    uv_depth[:,:,2] = depth_img/depth_shift
    uv_depth = np.reshape(uv_depth, [-1,3])
    uv_depth = uv_depth[np.where(uv_depth[:,2]!=0),:].squeeze()
    
    intrinsic_inv = np.linalg.inv(depth_intrinsic)
    fx = depth_intrinsic[0,0]
    fy = depth_intrinsic[1,1]
    cx = depth_intrinsic[0,2]
    cy = depth_intrinsic[1,2]
    bx = depth_intrinsic[0,3]
    by = depth_intrinsic[1,3]
    n = uv_depth.shape[0]
    points = np.ones((n,4))
    X = (uv_depth[:,0]-cx)*uv_depth[:,2]/fx + bx
    Y = (uv_depth[:,1]-cy)*uv_depth[:,2]/fy + by
    points[:,0] = X
    points[:,1] = Y
    points[:,2] = uv_depth[:,2]
    points_world = np.dot(points, np.transpose(pose))
    group_ids = num_to_natural(group_ids)
    save_dict = dict(coord=points_world[:,:3], color=colors, group=group_ids, stability_score=stability_scores, predicted_iou=predicted_ious,feature=features)
    return save_dict

def make_open3d_point_cloud(input_dict, voxelize, th):
    input_dict["group"] = remove_small_group(input_dict["group"], th)
    # input_dict = voxelize(input_dict)

    xyz = input_dict["coord"]
    if np.isnan(xyz).any():
        return None
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    return pcd

# Focus on this function for global mask_ID solution
def seg_pcd(scene_name, rgb_path, data_path, save_path, mask_generator, voxel_size, voxelize, th, train_scenes, val_scenes, save_2dmask_path, gt_data_path):
    
    print(scene_name, flush=True)

    if scene_name in train_scenes:
        scene_path = join(data_path, "train", scene_name + ".pth")
    elif scene_name in val_scenes:
        scene_path = join(data_path, "val", scene_name + ".pth")
    else: 
        scene_path = join(data_path, "test", scene_name + ".pth")
    #data_dict = torch.load(scene_path)

    #print("Available keys:", data_dict.keys())

    if os.path.exists(join(save_path, scene_name + "_comparisons_output_part1.pth")):
        return

    # Step 1 in pipeline: SAM Generate Masks

    step1_output_path = os.path.join(save_2dmask_path, scene_name + "_step1.pth")

    # Returns the names of the multi-images in the scene
    color_names = sorted(os.listdir(os.path.join(rgb_path, scene_name, 'color')), key=lambda a: int(os.path.basename(a).split('.')[0]))

    voxelize_new = Voxelize(voxel_size=args.voxel_size, mode="train", keys=("coord", "color", "group", "feature", "predicted_iou", "stability_score"))

    # If the output of Step 1 and 2 already exists, load it
    if os.path.exists(step1_output_path):
        data = torch.load(step1_output_path)
        pcd_list, pcd_list_ = data['pcd_list'], data['pcd_list_']
        print("Loaded Step 1 and 2 output from file.")
    else:
        pcd_list = []
        pcd_list_ = []
        for color_id, color_name in enumerate(color_names):
            print(color_name, flush=True) # print name of the image used for mask generation
            pcd_dict = get_pcd(scene_name, color_name, rgb_path, mask_generator, save_2dmask_path)
            if len(pcd_dict["coord"]) == 0:
                continue

            viewpoint_ids = [color_id + 1] * len(pcd_dict["coord"])
            viewpoint_names = [color_name] * len(pcd_dict["coord"])

            pcd_dict.update(viewpoint_id=viewpoint_ids, viewpoint_name=viewpoint_names)

            if verbose:
                # Extract data from the dictionary
                coords = pcd_dict_['coord']
                colors = pcd_dict_['color']
                group_ids = pcd_dict_['group']
                viewpoint_ids = pcd_dict_['viewpoint_id']
                stability_scores = pcd_dict_['stability_score']
                predicted_ious = pcd_dict_['predicted_iou']
                features = pcd_dict_['feature']

                # Print the header
                print(f"{'Viewpoint ID':<10} {'Coord':<65} {'Color':<20} {'Group ID':<10} {'Stability Score':<20} {'Prediction IOU':<20} {'Feature':<40}")

                for viewpoint_id, coord, color, group_id, stability_score, predicted_iou, feature in zip(viewpoint_ids, coords, colors, group_ids, stability_scores, predicted_ious, features):
                    coord_str = ', '.join(map(str, coord))
                    color_str = ', '.join(map(str, color))
                    feature_str = ', '.join(map(str, feature))
                    print(f"{viewpoint_id:<10} {coord_str:<65} {color_str:<20} {group_id:<10} {stability_score:<20} {predicted_iou:<20} {feature_str:<40}")

            pcd_dict_ = voxelize_new(pcd_dict)
            pcd_dict = voxelize(pcd_dict)

            if verbose:
                # Extract data from the dictionary
                coords = pcd_dict_['coord']
                colors = pcd_dict_['color']
                group_ids = pcd_dict_['group']
                viewpoint_ids = pcd_dict_['viewpoint_id']
                stability_scores = pcd_dict_['stability_score']
                predicted_ious = pcd_dict_['predicted_iou']
                features = pcd_dict_['feature']

                # Print the header
                print(f"{'Viewpoint ID':<10} {'Coord':<65} {'Color':<20} {'Group ID':<10} {'Stability Score':<20} {'Prediction IOU':<20} {'Feature':<40}")

                for viewpoint_id, coord, color, group_id, stability_score, predicted_iou, feature in zip(viewpoint_ids, coords, colors, group_ids, stability_scores, predicted_ious, features):
                    coord_str = ', '.join(map(str, coord))
                    color_str = ', '.join(map(str, color))
                    feature_str = ', '.join(map(str, feature))
                    print(f"{viewpoint_id:<10} {coord_str:<65} {color_str:<20} {group_id:<10} {stability_score:<20} {predicted_iou:<20} {feature_str:<40}")

            pcd_list.append(pcd_dict)
            pcd_list_.append(pcd_dict_)

def seg_pcd_matterport(scene_name, rgb_zip, depth_zip, save_path, mask_generator, voxel_size, voxelize, th, save_2dmask_path):
    
    print(f"Starting segmentation for scene: {scene_name}", flush=True)

    # Check if RGB data is available
    if not rgb_zip.namelist():
        raise FileNotFoundError(f"RGB data not found for scene {scene_name} at {rgb_zip}")
    print(f"RGB data found for scene: {scene_name}", flush=True)

    # Check if depth data is available
    if not depth_zip.namelist():
        raise FileNotFoundError(f"Depth data not found for scene {scene_name} at {depth_zip}")
    print(f"Depth data found for scene: {scene_name}", flush=True)

    step1_output_path = os.path.join(save_2dmask_path, scene_name + "_step1.pth")
    print(f"Step 1 output path: {step1_output_path}", flush=True)

    # Get and sort the ground truth files
    gt_files = sorted([name for name in gt_zip.namelist() if name.endswith('.png') or name.endswith('.jpg')])
    print(f"Found {len(gt_files)} ground truth files: {gt_files}", flush=True)

    # Extract color image names from the RGB zip
    color_names = sorted(
        [name for name in rgb_zip.namelist() if name.endswith('.jpg')], 
        key=lambda a: int(os.path.splitext(os.path.basename(a))[0])
    )
    print(f"Found {len(color_names)} RGB images: {color_names}", flush=True)

    voxelize_new = Voxelize(voxel_size=voxel_size, mode="train", keys=("coord", "color", "group", "feature", "predicted_iou", "stability_score"))
    print(f"Voxelizer initialized with voxel size: {voxel_size}", flush=True)

    if os.path.exists(step1_output_path):
        data = torch.load(step1_output_path)
        pcd_list = data['pcd_list']
        print("Loaded Step 1 and 2 output from file.")
    else:
        pcd_list = []
        print("Step 1 and 2 data not found, processing color images...", flush=True)
        for color_id, color_name in enumerate(color_names):
            print(f"Processing color image {color_id + 1}/{len(color_names)}: {color_name}", flush=True)
            
            # Extract color image file from zip
            with rgb_zip.open(color_name) as color_file:
                print(f"Extracting point cloud data from {color_name}", flush=True)
                pcd_dict = get_pcd(scene_name, color_name, color_file, mask_generator, save_2dmask_path)
            
            if len(pcd_dict["coord"]) == 0:
                print(f"No coordinates found for {color_name}, skipping...", flush=True)
                continue

            viewpoint_ids = [color_id + 1] * len(pcd_dict["coord"])
            viewpoint_names = [color_name] * len(pcd_dict["coord"])
            pcd_dict.update(viewpoint_id=viewpoint_ids, viewpoint_name=viewpoint_names)

            # Append the point cloud data
            pcd_list.append(pcd_dict)
            print(f"Processed {len(pcd_dict['coord'])} points for {color_name}", flush=True)
        
        print(f"Processed a total of {len(pcd_list)} point clouds.", flush=True)

    # Now that we have the point clouds, perform further processing or saving if needed.
    print(f"Segmentation for scene {scene_name} completed.", flush=True)

def load_point_cloud_with_faces(ply_path):
    mesh = o3d.io.read_triangle_mesh(ply_path)  # Read as a mesh to access faces
    if mesh.is_empty():
        raise ValueError("Empty point cloud or mesh.")
    vertices = np.asarray(mesh.vertices)  # Use vertices directly
    faces = np.asarray(mesh.triangles)  # Face indices for vertex references
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(vertices)
    if mesh.has_vertex_colors():
        point_cloud.colors = o3d.utility.Vector3dVector(np.asarray(mesh.vertex_colors))
    return point_cloud, faces

def assign_labels_from_faces(point_cloud, semseg_data, fseg_data, faces, category_mapping):
    """
    Assign semantic and instance IDs to points based on face indices and segmentations.

    Args:
        point_cloud: Open3D PointCloud object.
        semseg_data: Data containing segmentation groups and their corresponding labels.
        fseg_data: Data containing the face indices of each segment.
        faces: List of faces where each face is represented by indices of points.
        category_mapping: Dictionary that maps raw categories to semantic IDs.

    Returns:
        point_cloud: The PointCloud object with assigned semantic and instance IDs.
    """
    # Initialize arrays to store the IDs
    num_points = len(point_cloud.points)
    semantic_ids = np.full(num_points, -1, dtype=int)  # Default to -1
    instance_ids = np.full(num_points, -1, dtype=int)  # Default to -1
    assigned_points = set()  # To avoid re-assigning points

    # Retrieve segIndices (face indices) from fseg data
    seg_indices = fseg_data.get("segIndices", [])

    print(f"length of seg_indices: {len(seg_indices)}")
    print(f"length of faces_region: {len(faces)}")

    semantic_colors = np.zeros((num_points, 3), dtype=np.float64)
    instance_colors = np.zeros((num_points, 3), dtype=np.float64)
    instance_color_map = {}

    # For each segment in semseg_data, assign the corresponding instance and semantic IDs
    for seg_group in semseg_data.get("segGroups", []):
        instance_id = seg_group["id"]
        raw_category = seg_group["label"]  # The label is the raw category
        print(f"raw_category: {raw_category}")
        
        # Map the raw category to its corresponding semantic ID using category_mapping
        category_info = category_mapping.get(raw_category, {})
        semantic_id = category_info.get("semantic_id", -1)  # Default to -1 if not found
        semantic_rgb = category_info.get("rgb", (0.0, 0.0, 0.0))

        print(f"semantic_id in category_info: {semantic_id}")
        
        for segment_idx in seg_group["segments"]:  # Segment indices
            # Find all occurrences of segment_idx in seg_indices
            face_indices = [i for i, x in enumerate(seg_indices) if x == segment_idx]
            
            #print(f"length of face_indices: {len(face_indices)}")
            for face_idx in face_indices:  # Iterate over all matching face indices
                if face_idx < len(faces):  # Check if the face exists
                    # Get the point indices corresponding to the face
                    point_indices = faces[face_idx]
                    
                    # Assign semantic and instance IDs to the points belonging to the face
                    for point_idx in point_indices:
                        if point_idx not in assigned_points:  # Avoid redundant assignments
                            semantic_ids[point_idx] = semantic_id
                            instance_ids[point_idx] = instance_id
                            assigned_points.add(point_idx)

                            semantic_colors[point_idx] = semantic_rgb
                            # Check if color for this instance_id is already generated
                            if instance_id not in instance_color_map:
                                instance_color_map[instance_id] = generate_random_color()
                            # Assign the color to the point
                            instance_colors[point_idx] = instance_color_map[instance_id]
    # Add scalar fields for semantic and instance IDs
    point_cloud_gt_semantic = o3d.geometry.PointCloud()
    point_cloud_gt_semantic.points = o3d.utility.Vector3dVector(np.asarray(point_cloud.points))
    point_cloud_gt_semantic.colors = o3d.utility.Vector3dVector(semantic_colors)
    point_cloud_gt_instance = o3d.geometry.PointCloud()
    point_cloud_gt_instance.points = o3d.utility.Vector3dVector(np.asarray(point_cloud.points))
    point_cloud_gt_instance.colors = o3d.utility.Vector3dVector(instance_colors)

    return point_cloud_gt_semantic, point_cloud_gt_instance

def load_category_mapping(category_mapping_file):
    category_mapping = {}
    
    # Open and read the category_mapping file
    with open(category_mapping_file, mode='r') as file:
        reader = csv.DictReader(file, delimiter='\t')
        
        # Read each row and extract the mapping information along with RGB values
        for row in reader:
            raw_category = row['raw_category']
            semantic_id = int(row['index'])
            # Extract the RGB string and convert it to a tuple (R, G, B)
            rgb = tuple(map(float, row['rgb'].split(',')))
            
            # Store in the dictionary: raw_category -> (semantic_id, rgb)
            category_mapping[raw_category] = {
                'semantic_id': semantic_id,
                'rgb': rgb
            }
    
    return category_mapping

def write_category_colors_mapping(category_mapping_file):
    category_mapping = {}

    # Open and read the category_mapping file
    with open(category_mapping_file, mode='r') as file:
        reader = list(csv.DictReader(file, delimiter='\t')) 

        # Generate RGB colors based on number of rows (categories)
        semantic_colors = generate_colors(len(reader))

        # Prepare the updated rows with colors
        updated_rows = []

        for i, row in enumerate(reader):
            raw_category = row['raw_category']
            semantic_id = int(row['index'])
            color = semantic_colors[i]
            category_mapping[raw_category] = {
                'semantic_id': semantic_id,
                'rgb': color
            }
            
            # Add the color as columns to the row
            updated_row = row.copy()  # Copy the original row
            updated_row['rgb'] = f"{color[0]:.6f},{color[1]:.6f},{color[2]:.6f}"  # Add the RGB as a string
            updated_rows.append(updated_row)

    # Write the updated CSV file with RGB colors
    with open(category_mapping_file, mode='w', newline='') as file:
        fieldnames = list(reader[0].keys()) + ['rgb']  # Add 'rgb' field to header
        writer = csv.DictWriter(file, fieldnames=fieldnames, delimiter='\t')

        writer.writeheader()  # Write the header
        writer.writerows(updated_rows)  # Write the updated rows

    return category_mapping

def acquire_matterport_gt(scene_name,gt_scene_zip,gt_region_zip,save_path,category_mapping_file):

    category_mapping = load_category_mapping(category_mapping_file)

    if not gt_scene_zip.namelist():
        print(f"Contents of scene segmentations zip {gt_scene_zip}:")
        for file_name in gt_scene_zip.namelist():
            print(f"  - {file_name}")
        raise FileNotFoundError(f"Ground truth data not found for scene {scene_name} at {gt_scene_zip}")

    if not gt_region_zip.namelist():
        print(f"Contents of region segmentations zip {gt_region_zip}:")
        for file_name in gt_region_zip.namelist():
           print(f"  - {file_name}")
        raise FileNotFoundError(f"Ground truth data not found for regions of {scene_name} at {gt_region_zip}")

    region_pc_files = [f for f in gt_region_zip.namelist() if "region" in f and f.endswith(".ply")]
    region_semseg_files = [f for f in gt_region_zip.namelist() if "semseg.json" in f]
    region_fseg_files = [f for f in gt_region_zip.namelist() if "fsegs.json" in f]

    print(f"region_pc_files: {region_pc_files}, count: {len(region_pc_files)}")
    print(f"region_semseg_files: {region_semseg_files}, count: {len(region_semseg_files)}")
    print(f"region_fseg_files: {region_fseg_files}, count: {len(region_fseg_files)}")


    for region_pc_file, region_semseg_file, region_fseg_file in zip(region_pc_files, region_semseg_files, region_fseg_files):
        region_segmentations_path = os.path.join(save_path, scene_name, "region_segmentations")
        os.makedirs(region_segmentations_path, exist_ok=True)
        base_name, _ = os.path.splitext(region_pc_file)
        save_semantic_ply_path = os.path.join(save_path, f"{base_name}_semantic_gt.ply")
        save_instance_ply_path = os.path.join(save_path, f"{base_name}_instance_gt.ply")
        if os.path.exists(save_semantic_ply_path) and os.path.exists(save_instance_ply_path):
            print(f"Semantic and instance GT files already exist for the region {region_pc_file} at {save_semantic_ply_path} and {save_instance_ply_path}")
        else:
            with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as temp_ply_file:
                temp_ply_file.write(gt_region_zip.read(region_pc_file))
                temp_ply_file.close()
                point_cloud_region, faces_region = load_point_cloud_with_faces(temp_ply_file.name)
                os.remove(temp_ply_file.name)
            with gt_region_zip.open(region_semseg_file) as semseg_file_region:
                semseg_data_region = json.load(semseg_file_region)
            with gt_region_zip.open(region_fseg_file) as fseg_file_region:
                fseg_data_region = json.load(fseg_file_region)

            pc_region_semantic_gt, pc_region_instance_gt = assign_labels_from_faces(point_cloud_region, semseg_data_region, fseg_data_region, faces_region, category_mapping)

            if o3d.io.write_point_cloud(save_semantic_ply_path, pc_region_semantic_gt):
                print(f"Saved region point cloud with instance and semantic IDs to {save_semantic_ply_path}")
            else:
                print(f"Failed to save point cloud to {save_semantic_ply_path}")

            if o3d.io.write_point_cloud(save_instance_ply_path, pc_region_instance_gt):
                print(f"Saved region point cloud with instance and semantic IDs to {save_instance_ply_path}")
            else:
                print(f"Failed to save point cloud to {save_instance_ply_path}")

    '''scene_pc_file = [f for f in gt_scene_zip.namelist() if scene_name in f and f.endswith(".ply")]
    scene_semseg_file = [f for f in gt_scene_zip.namelist() if "semseg.json" in f]

    if scene_pc_file:
        with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as temp_ply_file:
            temp_ply_file.write(gt_scene_zip.read(scene_pc_file[0]))
            temp_ply_file.close()
            point_cloud_scene, faces_scene = load_point_cloud_with_faces(temp_ply_file.name)
            os.remove(temp_ply_file.name)
    else:
        raise FileNotFoundError(f"No .ply file found in {gt_scene_zip.filename} for {scene_name}")

    if scene_semseg_file:
        with gt_scene_zip.open(scene_semseg_file[0]) as semseg_file:
            semseg_data_scene= json.load(semseg_file)
            print(f"Top-level keys in semseg_data: {semseg_data_scene.keys()}")

            # Print the number of segment groups
            seg_groups = semseg_data_scene.get("segGroups", [])
            print(f"Number of segment groups: {len(seg_groups)}")

            # Inspect the first segment group
            if seg_groups:
                first_group = seg_groups[0]
                print("\nFirst Segment Group Info:")
                for key, value in first_group.items():
                    print(f"  {key}: {value}")
            else:
                print("No segment groups found.")
    else:
        raise FileNotFoundError(f"No semseg.json file found in {gt_scene_zip.filename} for {scene_name}")

    point_cloud_gt = assign_labels_from_faces(point_cloud_scene, semseg_data_scene, faces_scene)

    scene_gt_ply_path = os.path.join(save_path, scene_name)
    o3d.io.write_point_cloud(scene_gt_ply_path, point_cloud_gt)
    print(f"Saved ground truth point cloud with semantic and instance labels to {scene_gt_ply_path}")

    region_files = [f for f in gt_region_zip.namelist() if "region" in f and f.endswith(".ply")]
    region_semseg_files = [f for f in gt_region_zip.namelist() if "semseg.json" in f]

    for region_ply_file, semseg_file in zip(region_files, region_semseg_files):
        # Extracting region point cloud
        gt_region_zip.extract(region_ply_file, path=save_path)
        region_ply_path = os.path.join(save_path, region_ply_file)
        point_cloud = o3d.io.read_point_cloud(region_ply_path)
        points = np.asarray(point_cloud.points)
        
        # Load semantic data
        semseg_path = os.path.join(save_path, semseg_file)
        semseg_data = load_json(semseg_path)
        segment_to_label = {seg['segments'][0]: seg['label'] for seg in semseg_data['segGroups']}
        
        # Add custom fields to point cloud (Placeholder logic)
        # Assuming a fixed mapping from `segment_id` to each point
        semantic_classes = np.zeros(len(points), dtype=np.int32)  # Dummy initialization
        instance_ids = np.zeros(len(points), dtype=np.int32)  # Dummy initialization
        
        # Add semantic and instance fields to the point cloud
        point_cloud.colors = o3d.utility.Vector3dVector(points[:, :3])  # Use original RGB
        point_cloud.points = o3d.utility.Vector3dVector(points[:, :3])  # XYZ coordinates
        
        # Save augmented point cloud with new attributes
        output_path = region_ply_path.replace('.ply', '_gt.ply')
        o3d.io.write_point_cloud(output_path, point_cloud)
        print(f"Saved ground truth point cloud with semantic and instance labels to {output_path}")'''

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
    
    mask_generator = SamAutomaticMaskGenerator(build_sam(checkpoint=args.sam_checkpoint_path).to(device="cuda"))
    voxelize = Voxelize(voxel_size=args.voxel_size, mode="train", keys=("coord", "color", "group"))
    os.makedirs(args.save_path, exist_ok=True)
    
    #write_category_colors_mapping(args.category_mapping_path) # Run once to save RGB colormap for semantic classes

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
            
            # GT data extraction
            acquire_matterport_gt(scene_name, gt_scene_zip, gt_region_zip, args.save_path, args.category_mapping_path)
            
            #seg_pcd_matterport(
            #    scene_name, rgb_zip, depth_zip, args.save_path, 
            #    mask_generator, args.voxel_size, voxelize, args.th, 
            #    args.save_2dmask_path)


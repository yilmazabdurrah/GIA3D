"""
Main Script (Just step 1 and 2 for single run to save data)

Author: Yunhan Yang (yhyang.myron@gmail.com)

Updated by
Abdurrahman Yilmaz (ayilmaz@lincoln.ac.uk) v06
22 Jul 2024
"""

import os
import cv2
import numpy as np
import open3d as o3d
import torch
import copy
import multiprocessing as mp
import pointops
import random
import argparse

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

import time, subprocess

verbose = False # To print out intermediate data
verbose_graph = False # To plot correspondence graphs before and after optimization
verbose_comparisons = False # To plot comparison output

def pcd_ensemble(org_path, new_path, data_path, vis_path):
    new_pcd = torch.load(new_path)
    new_pcd = num_to_natural(remove_small_group(new_pcd, 20))
    with open(org_path) as f:
        segments = json.load(f)
        org_pcd = np.array(segments['segIndices'])
    match_inds = [(i, i) for i in range(len(new_pcd))]
    new_group = cal_group(dict(group=new_pcd), dict(group=org_pcd), match_inds)
    print(new_group.shape)
    data = torch.load(data_path)
    visualize_partition(data["coord"], new_group, vis_path)

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

# Mask ID correspondence between two frames
def cal_group(input_dict, new_input_dict, match_inds, ratio=0.5):
    group_0 = input_dict["group"]
    group_1 = new_input_dict["group"]
    group_1[group_1 != -1] += group_0.max() + 1
    
    unique_groups, group_0_counts = np.unique(group_0, return_counts=True)
    group_0_counts = dict(zip(unique_groups, group_0_counts))
    unique_groups, group_1_counts = np.unique(group_1, return_counts=True)
    group_1_counts = dict(zip(unique_groups, group_1_counts))

    # Calculate the group number correspondence of overlapping points
    group_overlap = {}
    for i, j in match_inds:
        group_i = group_1[i]
        group_j = group_0[j]
        if group_i == -1:
            group_1[i] = group_0[j]
            continue
        if group_j == -1:
            continue
        if group_i not in group_overlap:
            group_overlap[group_i] = {}
        if group_j not in group_overlap[group_i]:
            group_overlap[group_i][group_j] = 0
        group_overlap[group_i][group_j] += 1

    # Update group information for point cloud 1
    for group_i, overlap_count in group_overlap.items():
        # for group_j, count in overlap_count.items():
        max_index = np.argmax(np.array(list(overlap_count.values())))
        group_j = list(overlap_count.keys())[max_index]
        count = list(overlap_count.values())[max_index]
        total_count = min(group_0_counts[group_j], group_1_counts[group_i]).astype(np.float32)
        # print(count / total_count)
        if count / total_count >= ratio:
            group_1[group_1 == group_i] = group_j
    return group_1

def normalized_feature_difference(f_i, f_j):
    # Calculate the Euclidean norm of the difference
    difference_norm = np.linalg.norm(f_i - f_j)
    # Normalize the difference norm
    normalized_difference = difference_norm / len(f_i)
    return normalized_difference

# Mask ID correspondence between all frames
def cal_graph(input_dict, new_input_dict, match_inds, coefficient_combinations=[[0.25, 0.25, 0.25, 0.25]]):

    group_0 = input_dict["group"]
    group_1 = new_input_dict["group"]
    coord_0 = input_dict["coord"]
    coord_1 = new_input_dict["coord"]
    view_ids_0 = input_dict["viewpoint_id"]
    view_ids_1 = new_input_dict["viewpoint_id"]
    view_names_0 = input_dict["viewpoint_name"]
    view_names_1 = new_input_dict["viewpoint_name"]

    features_0 = input_dict["feature"]
    features_1 = new_input_dict["feature"]
    stability_scores_0 = input_dict["stability_score"]
    stability_scores_1 = new_input_dict["stability_score"]
    predicted_ious_0 = input_dict["predicted_iou"]
    predicted_ious_1 = new_input_dict["predicted_iou"]

    unique_nodes_0 = list(set(zip(view_names_0, group_0)))
    unique_nodes_1 = list(set(zip(view_names_1, group_1)))

    # Initialize the graph
    correspondence_graph = nx.DiGraph()   

    for node in unique_nodes_0:
        correspondence_graph.add_node(node)
    for node in unique_nodes_1:
        correspondence_graph.add_node(node)

    # Calculate the group number correspondence of overlapping points
    point_cnt_group_0 = {}
    point_cnt_group_1 = {}
    
    unique_values_group_0 = set(group_0)

    for unique_value in unique_values_group_0:
        point_cnt_group_0[unique_value] = sum(1 for element in group_0 if element == unique_value)

    unique_values_group_1 = set(group_1)

    for unique_value in unique_values_group_1:
        point_cnt_group_1[unique_value] = sum(1 for element in group_1 if element == unique_value)

    #print("Counts for group 0 ", point_cnt_group_0)
    #print("Counts for group 1 ", point_cnt_group_1)

    cost = {}
    group_overlap = {}

    for i, j in match_inds:
        group_i = group_1[i]
        group_j = group_0[j]
        view_id_i = view_ids_1[i]
        view_id_j = view_ids_0[j]
        
        view_name_i = view_names_1[i]
        view_name_j = view_names_0[j]
        coord_i = coord_1[i]

        feature_i = features_1[i]
        feature_j = features_0[j]

        stability_score_i = stability_scores_1[i]
        stability_score_j = stability_scores_0[j]

        predicted_iou_i = predicted_ious_1[i]
        predicted_iou_j = predicted_ious_0[j]

        '''if group_i == -1:
            group_1[i] = group_0[j]
            continue
        if group_j == -1:
            continue'''
        
        if group_i == -1 or group_j == -1:
            continue
        '''elif group_i == -1:
            group_1[i] = group_0[j]
            group_i = group_j
            new_input_dict["group"][i] = group_0[j]'''

        overlap_key = (group_i, view_id_i, view_name_i, point_cnt_group_1[group_i], group_j, view_id_j, view_name_j, point_cnt_group_0[group_j])
        #print("overlap_key: ", overlap_key, " and features: ", feature_i, feature_j)
        if overlap_key not in group_overlap:
            group_overlap[overlap_key] = 0
            cost[overlap_key] = {tuple(coefficients): 0 for coefficients in coefficient_combinations}
        group_overlap[overlap_key] += 1
        '''if group_i == -1 and group_j == -1:
            cost[overlap_key] += 3
        elif group_i == -1 or group_j == -1:
            continue
        else:'''

        for coefficients in coefficient_combinations:
            L1, L2, L3, L4 = coefficients

            cost[overlap_key][tuple(coefficients)] += L1*normalized_feature_difference(feature_i,feature_j) + L2*abs(stability_score_i - stability_score_j) + L3*abs(predicted_iou_i - predicted_iou_j)

    #print("cost: ", cost)

    # Add edges with costs for all coefficient combinations
    #print(f"length of group overlap: {len(group_overlap.items())}")
    for key, count in group_overlap.items():
        group_i, view_id_i, view_name_i, point_cnt_group_i, group_j, view_id_j, view_name_j, point_cnt_group_j = key
        edge_data = {
            'count_common': count,
            'count_total': [point_cnt_group_i, point_cnt_group_j],
            'viewpoint_id_0': view_id_j,
            'viewpoint_id_1': view_id_i,
            'cost': {}
        }

        for coefficients in coefficient_combinations:
            L1, L2, L3, L4 = coefficients
            cost_value = cost[key][tuple(coefficients)] / count + L4 * max(0, (1 - count / min(point_cnt_group_i, point_cnt_group_j)))
            edge_data['cost'][tuple(coefficients)] = cost_value

        correspondence_graph.add_edge(
            (view_name_i, group_i), 
            (view_name_j, group_j), 
            **edge_data
        )
        #print(f"key: {key} and count: {count}")

    return correspondence_graph, input_dict, new_input_dict

def cal_scenes(pcd_list, index, voxel_size, voxelize, th=50, coefficient_combinations=[[0.25, 0.25, 0.25, 0.25]]):
    #print(index, flush=True)
    input_dict_0 = pcd_list[index]
    input_dict_1 = {}
    pcd0 = make_open3d_point_cloud(input_dict_0, voxelize, th)
    merged_graph = nx.DiGraph()
    for i, pcd_dict in enumerate(pcd_list):
        if i > index: # i != index:
            input_dict_1.update(pcd_dict)
            pcd1 = make_open3d_point_cloud(input_dict_1, voxelize, th)
            if pcd0 == None:
                if pcd1 == None:                    
                    return merged_graph, pcd_list
                else:
                    pcd_list[i].update(input_dict_1)
                    return merged_graph, pcd_list
            elif pcd1 == None:
                pcd_list[index].update(input_dict_0)
                return merged_graph, pcd_list

            # Cal Dul-overlap
            match_inds = get_matching_indices(pcd1, pcd0, 1.5 * voxel_size, 1)
            if match_inds:
                correspondence_graph, input_dict_0, input_dict_1 = cal_graph(input_dict_0, input_dict_1, match_inds, coefficient_combinations)
                pcd_list[i].update(input_dict_1)
                pcd_list[index].update(input_dict_0)
                if len(correspondence_graph.nodes) > 0 and len(correspondence_graph.edges) > 0:
                    merged_graph = nx.compose(merged_graph, correspondence_graph)

    return merged_graph, pcd_list

def cal_2_scenes(pcd_list, index, voxel_size, voxelize, th=50):
    if len(index) == 1:
        return(pcd_list[index[0]])
    # print(index, flush=True)
    input_dict_0 = pcd_list[index[0]]
    input_dict_1 = pcd_list[index[1]]
    pcd0 = make_open3d_point_cloud(input_dict_0, voxelize, th)
    pcd1 = make_open3d_point_cloud(input_dict_1, voxelize, th)
    if pcd0 == None:
        if pcd1 == None:
            return None
        else:
            return input_dict_1
    elif pcd1 == None:
        return input_dict_0

    # Cal Dul-overlap
    match_inds = get_matching_indices(pcd1, pcd0, 1.5 * voxel_size, 1)
    pcd1_new_group = cal_group(input_dict_0, input_dict_1, match_inds)
    # print(pcd1_new_group)

    match_inds = get_matching_indices(pcd0, pcd1, 1.5 * voxel_size, 1)
    input_dict_1["group"] = pcd1_new_group
    pcd0_new_group = cal_group(input_dict_1, input_dict_0, match_inds)
    # print(pcd0_new_group)

    pcd_new_group = np.concatenate((pcd0_new_group, pcd1_new_group), axis=0)
    pcd_new_group = num_to_natural(pcd_new_group)
    pcd_new_coord = np.concatenate((input_dict_0["coord"], input_dict_1["coord"]), axis=0)
    pcd_new_color = np.concatenate((input_dict_0["color"], input_dict_1["color"]), axis=0)
    pcd_dict = dict(coord=pcd_new_coord, color=pcd_new_color, group=pcd_new_group)

    pcd_dict = voxelize(pcd_dict)
    return pcd_dict

# Focus on this function for global mask_ID solution
def seg_pcd(scene_name, rgb_path, data_path, save_path, mask_generator, voxel_size, voxelize, th, train_scenes, val_scenes, save_2dmask_path, gt_data_path):
    print(scene_name, flush=True)
    
    if scene_name in train_scenes:
        scene_path = join(data_path, "train", scene_name + ".pth")
    elif scene_name in val_scenes:
        scene_path = join(data_path, "val", scene_name + ".pth")
    else: 
        scene_path = join(data_path, "test", scene_name + ".pth")

    # Step 1 in pipeline: SAM Generate Masks

    step1_output_path = os.path.join(save_path, scene_name + "_step1.pth")

    # Returns the names of the multi-images in the scene
    color_names = sorted(os.listdir(os.path.join(rgb_path, scene_name, 'color')), key=lambda a: int(os.path.basename(a).split('.')[0]))

    voxelize_new = Voxelize(voxel_size=args.voxel_size, mode="train", keys=("coord", "color", "group", "feature", "predicted_iou", "stability_score"))

    # If the output of Step 1 and 2 already exists, load it
    if os.path.exists(step1_output_path):
        #data = torch.load(step1_output_path)
        #pcd_list, pcd_list_ = data['pcd_list'], data['pcd_list_']
        print(f"Step 1 and 2 output data available for {scene_name}.")
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

        # Step 2 in pipeline: Merge Two Adjacent Pointclouds until get single point cloud
        while len(pcd_list) != 1:
            print(len(pcd_list), flush=True)
            new_pcd_list = []
            for indice in pairwise_indices(len(pcd_list)):
                pcd_frame = cal_2_scenes(pcd_list, indice, voxel_size=voxel_size, voxelize=voxelize)
                if pcd_frame is not None:
                    new_pcd_list.append(pcd_frame)
            pcd_list = new_pcd_list

        # Save the output of Step 1 and 2 to a file for future use
        torch.save({'pcd_list': pcd_list, 'pcd_list_': pcd_list_}, step1_output_path)
        print("Saved Step 1 and 2 output to a file.")

def get_args():
    '''Command line arguments.'''

    parser = argparse.ArgumentParser(
        description='Segment Anything on ScanNet.')
    parser.add_argument('--rgb_path', type=str, help='the path of rgb data')
    parser.add_argument('--data_path', type=str, default='', help='the path of pointcload data')
    parser.add_argument('--save_path', type=str, help='Where to save the pcd results')
    parser.add_argument('--save_2dmask_path', type=str, default='', help='Where to save 2D segmentation result from SAM')
    parser.add_argument('--sam_checkpoint_path', type=str, default='', help='the path of checkpoint for SAM')
    parser.add_argument('--scannetv2_train_path', type=str, default='scannet-preprocess/meta_data/scannetv2_train.txt', help='the path of scannetv2_train.txt')
    parser.add_argument('--scannetv2_val_path', type=str, default='scannet-preprocess/meta_data/scannetv2_val.txt', help='the path of scannetv2_val.txt')
    parser.add_argument('--img_size', default=[640,480])
    parser.add_argument('--voxel_size', default=0.05)
    parser.add_argument('--th', default=50, help='threshold of ignoring small groups to avoid noise pixel')
    parser.add_argument('--gt_data_path', type=str, help='the path of ground truth data')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()
    print("Arguments:")
    print(args)
    with open(args.scannetv2_train_path) as train_file:
        train_scenes = train_file.read().splitlines()
    with open(args.scannetv2_val_path) as val_file:
        val_scenes = val_file.read().splitlines()
    mask_generator = SamAutomaticMaskGenerator(build_sam(checkpoint=args.sam_checkpoint_path).to(device="cuda"))
    voxelize = Voxelize(voxel_size=args.voxel_size, mode="train", keys=("coord", "color", "group"))
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    scene_names = sorted(os.listdir(args.rgb_path))
    for scene_name in scene_names:
        seg_pcd(scene_name, args.rgb_path, args.data_path, args.save_path, mask_generator, args.voxel_size, 
            voxelize, args.th, train_scenes, val_scenes, args.save_2dmask_path, args.gt_data_path)

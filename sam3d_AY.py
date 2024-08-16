"""
Main Script

Author: Yunhan Yang (yhyang.myron@gmail.com)

Updated by
Abdurrahman Yilmaz (ayilmaz@lincoln.ac.uk) v07
10 Aug 2024
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

import gc
import math

verbose = False # To print out intermediate data
verbose_graph = False # To plot correspondence graphs before and after optimization
verbose_comparisons = False # To plot comparison output

maxVal = 0
maxVal_stab = 0
maxVal_pred = 0
maxVal_iou = 0

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
    print(f"unique_groups in: {unique_groups}")
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
    print(f"unique_groups out: {np.unique(group_1)}")
    return group_1

def normalized_feature_difference(f_i, f_j):
    # Calculate the Euclidean norm of the difference
    difference_norm = np.linalg.norm(f_i - f_j)
    # Normalize the difference norm
    normalized_difference = difference_norm / len(f_i)
    return normalized_difference

# Mask ID correspondence between all frames
def cal_graph(input_dict, new_input_dict, match_inds, coefficient_combinations=[[0.25, 0.25, 0.25, 0.25]]):

    global maxVal
    global maxVal_stab
    global maxVal_pred
    global maxVal_iou

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

        if group_i == -1 or group_j == -1:
            continue
        '''if group_i == -1 and group_j == -1:
            continue
        elif group_i == -1:
            group_1[i] = group_j
            new_input_dict["group"][i] = group_j
            point_cnt_group_1[group_j] = point_cnt_group_1.get(group_j, 0) + 1
            group_i = group_j
            correspondence_graph.add_node((view_name_i, group_j))
        # If group_j is -1, skip this match
        elif group_j == -1:
            continue
        elif group_i == -1:
            new_input_dict["group"][i] = group_0[j]
            group_1[i] = group_0[j]
            print("point_cnt_group_1: ", point_cnt_group_1)
            point_cnt_group_1[group_j] = point_cnt_group_1[group_i]
            print("point_cnt_group_1: ", point_cnt_group_1)
            #del point_cnt_group_1[group_i]
            group_i = group_j

        if group_i == -1:
            group_1[i] = group_0[j]
            continue
        if group_j == -1:
            continue'''
        
        
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
            L1, L2, L3, _ = coefficients
            fd = normalized_feature_difference(feature_i,feature_j)
            ss = abs(stability_score_i - stability_score_j)
            if ss > 0.5:
                ss = 0.0
            piou = abs(predicted_iou_i - predicted_iou_j)
            if piou > 0.5:
                piou = 0.0
            
            cost[overlap_key][tuple(coefficients)] += L1*fd + L2*ss + L3*piou

    # Add edges with costs for all coefficient combinations
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
            _, _, _, L4 = coefficients
            
            cost_value = cost[key][tuple(coefficients)] / count + L4 * max(0, (1 - count / min(point_cnt_group_i, point_cnt_group_j)))
            edge_data['cost'][tuple(coefficients)] = cost_value

        correspondence_graph.add_edge(
            (view_name_i, group_i), 
            (view_name_j, group_j), 
            **edge_data
        )
        #print(f"key: {key} and count: {count} and cost: {edge_data['cost']}")

    return correspondence_graph, input_dict, new_input_dict

def cal_scenes(pcd_list, index, voxel_size, voxelize, th=50, coefficient_combinations=[[0.25, 0.25, 0.25, 0.25]]):
    #print(index, flush=True)
    input_dict_0 = pcd_list[index]
    input_dict_1 = {}
    pcd0 = make_open3d_point_cloud(input_dict_0, voxelize, th)
    merged_graph = nx.DiGraph()
    for i, pcd_dict in enumerate(pcd_list):
        if i != index: # i > index
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

#def update_groups_and_merge_dictionaries(pcd_list_, merged_graph_min, merged_nodes):
def update_groups_and_merge_dictionaries(pcd_list_, merged_graph_min):
    # Create a mapping for final groups to ensure all connected nodes have the same group
    final_group_map = {}
    
    grp_cnt = 0
    for u, v in merged_graph_min.edges():
        group_u = final_group_map.get(u, None)
        group_v = final_group_map.get(v, None)
        
        if group_u is None and group_v is None:
            # If neither node has a group assigned, create a new group and assign it to both
            new_group = grp_cnt
            grp_cnt += 1
            final_group_map[u] = new_group
            final_group_map[v] = new_group
        elif group_u is not None and group_v is None:
            # If u has a group but v does not, assign v to u's group
            final_group_map[v] = group_u
        elif group_u is None and group_v is not None:
            # If v has a group but u does not, assign u to v's group
            final_group_map[u] = group_v
        elif group_u != group_v:
            # If both nodes have different groups, unify the groups
            for node in final_group_map:
                if final_group_map[node] == group_v:
                    final_group_map[node] = group_u

    # Assign unique groups to nodes that are not connected
    for node in merged_graph_min.nodes():
        if node not in final_group_map:
            _, mask = node
            if mask != -1:
                final_group_map[node] = grp_cnt
                grp_cnt += 1
            else:
                final_group_map[node] = -1

    #print(f"The number of nodes {len(merged_graph_min.nodes())}")
    #print(f"The number of groups in the map {len(final_group_map)}")

    # Initialize empty lists for concatenated data
    all_coords = []
    all_colors = []
    all_groups = []

    # Iterate over the pcd_list_ and update group information
    for pcd in pcd_list_:
        viewpoint_name = pcd["viewpoint_name"][0]
        mask_ids = pcd["group"]
        
        # Create a mapping from old mask ids to new group ids
        new_groups = []
        cnt = 0
        for mask_id in mask_ids:
            if (viewpoint_name, mask_id) in final_group_map:
                #print(f"mask_id: {mask_id}, mapping: {final_group_map[(viewpoint_name, mask_id)]}")
                new_groups.append(final_group_map[(viewpoint_name, mask_id)])
            else:
                new_groups.append(-1)
                cnt+=1
                print(f"Not in dictionary {cnt}")
        
        all_coords.append(pcd["coord"])
        all_colors.append(pcd["color"])
        all_groups.append(new_groups)
    
    #print("All colors: ", len(np.concatenate(all_coords, axis=0)))
    #print("All colors: ", len(np.concatenate(all_colors, axis=0)))
    #print("All groups: ", len(np.concatenate(all_groups, axis=0)))

    #unique_groups = {tuple(group) for group in all_groups}
    #print(f"Unique groups {set(np.concatenate(all_groups, axis=0))} length: {len(set(np.concatenate(all_groups, axis=0)))}")
    
    # Combine all data into a single dictionary
    pcd_dict = {
        "coord": np.concatenate(all_coords, axis=0),
        "color": np.concatenate(all_colors, axis=0),
        "group": np.concatenate(all_groups, axis=0)
    }
    #print(f"number of groups: {grp_cnt}")
    return voxelize(pcd_dict)

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

    # New Step 2 in pipeline: Merge All Pointclouds in one shot globally to get single point cloud
    if len(pcd_list_) != 1:
        print("New Step 2")
        print(len(pcd_list_), flush=True)
        
        for index in range(1, len(pcd_list_)):
            # Get the 'group' value of the elements
            group_index_last = pcd_list_[index - 1]["group"]
            group_index = pcd_list_[index]["group"]
            
            # Update the current 'group' array
            group_index[group_index != -1] += group_index_last.max() + 1
            pcd_list_[index]["group"] = group_index

        threshold_to_coefficients = {
            0.25: [0.092, 0.082, 0.081, 0.745],
            0.3: [0.07, 0.057, 0.063, 0.811],
            0.35: [0.038, 0.029, 0.029, 0.904],
            0.4: [0.027, 0.027, 0.027, 0.919],
            0.45: [0.042, 0.042, 0.042, 0.875],
            0.5: [0.038, 0.038, 0.038, 0.888],
            0.55: [0.012, 0.012, 0.012, 0.962]
        }

        combinations_th = [0.3]

        print(f"Number of valid combinations for thresholds: {len(combinations_th)}")

        coefficients = [threshold_to_coefficients[th] for th in combinations_th]

        print(f"Number of valid combinations for coefficients: {len(coefficients)}")

        merged_graph = nx.DiGraph()
        print(f"Number of scenes: {len(pcd_list_)}")
        num_scene = len(pcd_list_) - 1
        for indice in range(len(pcd_list_)):
            print(f"Current scene: {indice}/{num_scene}")
            corr_graph, pcd_list_ = cal_scenes(pcd_list_, indice, voxel_size=voxel_size, voxelize=voxelize_new, coefficient_combinations=coefficients) 
            if len(corr_graph.nodes) > 0 and len(corr_graph.edges) > 0:
                merged_graph = nx.compose(merged_graph, corr_graph)
                #print(f"Num nodes: {len(merged_graph.nodes)}, nodes are : {merged_graph.nodes}")
            del corr_graph
            gc.collect()

        # Initialize the cost matrix with infinity
        large_value = 1e9  # Define a large value to replace infinity

        # Prepare to store merged graphs for each coefficient combination
        merged_graphs = {}
        merged_dicts = {}
        results = {}

        # Iterate over each coefficient combination
        for coefficients in coefficients:

            coefficients_str = "_".join(map(str, coefficients)).replace(" ", "").replace(",", "_")
            #print(f"coefficients_str: {coefficients_str}")

            results[tuple(coefficients)] = {}

            for threshold in combinations_th:

                threshold_str = str(threshold).replace(" ", "").replace(",", "_")
                #print(f"threshold_str: {threshold_str}")

                # Merge clusters and update the graph
                merged_graph_min = nx.DiGraph()
                
                for node in merged_graph.nodes():
                    if not merged_graph_min.has_node(node):
                        merged_graph_min.add_node(node)

                # Add edges to the new graph if they meet the threshold
                for u, v, data in merged_graph.edges(data=True):

                    # Ensure that the edge meets the threshold condition and connects different representatives
                    if u != v and data['cost'].get(tuple(coefficients), large_value) <= threshold:
                        if not merged_graph_min.has_edge(u, v) and not merged_graph_min.has_edge(v, u):
                            edge_data = {
                                'count_common': data['count_common'],
                                'count_total': data['count_total'],
                                'cost': data['cost'].get(tuple(coefficients), large_value),
                                'viewpoint_id_0': data['viewpoint_id_0'],
                                'viewpoint_id_1': data['viewpoint_id_1']
                            }
                            merged_graph_min.add_edge(u, v, **edge_data)
                        elif not merged_graph_min.has_edge(u, v) and merged_graph_min.has_edge(v, u):
                            reverse_data = merged_graph_min.get_edge_data(v, u)
                            reverse_cost = reverse_data['cost']
                            edge_cost = data['cost'].get(tuple(coefficients), large_value)
                            if edge_cost < reverse_cost:
                                merged_graph_min.remove_edge(v, u)
                                edge_data = {
                                    'count_common': data['count_common'],
                                    'count_total': data['count_total'],
                                    'cost': edge_cost,
                                    'viewpoint_id_0': data['viewpoint_id_0'],
                                    'viewpoint_id_1': data['viewpoint_id_1']
                                }
                                merged_graph_min.add_edge(u, v, **edge_data)
                        else:
                            print("Duplicate edges!!!")

                #print(f"Num nodes: {len(merged_graph_min.nodes)}, nodes are : {merged_graph_min.nodes} for minimized graph")
                # Store the result for this coefficient combination
                merged_graphs[tuple(coefficients)] = merged_graph_min

                #pcd_dict_merged_ = update_groups_and_merge_dictionaries(pcd_list_, merged_graph_min, merged_nodes)
                pcd_dict_merged_ = update_groups_and_merge_dictionaries(pcd_list_, merged_graph_min)
                merged_dicts[tuple(coefficients)] = pcd_dict_merged_
                results[tuple(coefficients)][threshold] = {
                    'pcd_dict_merged_': pcd_dict_merged_
                }

                if verbose_graph:
                    # Create a figure for subplots
                    plt.figure(figsize=(15, 12))

                    # Draw the first graph
                    plt.subplot(1, 2, 1)  # 1 row, 2 columns, 1st subplot
                    draw_graph(merged_graph, "Correspondences for All Viewpoints and Mask IDs before optimization", 121)

                    # Draw the second graph
                    plt.subplot(1, 2, 2)  # 1 row, 2 columns, 2nd subplot
                    draw_graph(merged_graph_min, "Correspondences for All Viewpoints and Mask IDs after optimization", 122)

                    # Display the plot
                    plt.show()

    # Step 3 in pipeline: Region Merging Method
    # Stage 1: Run i = 0 case once
    seg_dict = pcd_list[0]
    print(f'Unique groups baseline: {set(seg_dict["group"])}')
    seg_dict["group"] = num_to_natural(remove_small_group(seg_dict["group"], th))
    print(f'Unique groups baseline: {set(seg_dict["group"])}')

    if scene_name in train_scenes:
        scene_path = join(data_path, "train", scene_name + ".pth")
    elif scene_name in val_scenes:
        scene_path = join(data_path, "val", scene_name + ".pth")
    else: 
        scene_path = join(data_path, "test", scene_name + ".pth")

    data_dict = torch.load(scene_path)
    scene_coord = torch.tensor(data_dict["coord"]).cuda().contiguous()
    new_offset = torch.tensor(scene_coord.shape[0]).cuda()
    gen_coord = torch.tensor(seg_dict["coord"]).cuda().contiguous().float()
    offset = torch.tensor(gen_coord.shape[0]).cuda()
    gen_group = seg_dict["group"]
    indices, dis = pointops.knn_query(1, gen_coord, offset, scene_coord, new_offset)
    indices = indices.cpu().numpy()
    group = gen_group[indices.reshape(-1)].astype(np.int16)
    mask_dis = dis.reshape(-1).cpu().numpy() > 0.6
    group[mask_dis] = -1
    group = group.astype(np.int16)
    labels = np.array(group)
    print(f"Unique groups baseline: {set(labels)}")
    coord_ = data_dict.get("coord", None)

    torch.save(num_to_natural(group), join(save_path, scene_name + "_baseline.pth"))

    # Stage 2: Iterate over the merged_dicts for the i = 1 case
    labels_new_list = []

    for coefficients, merged_dicts in results.items():
        for threshold, result in merged_dicts.items():
            print(f"\nProcessing for coefficients {coefficients} and threshold {threshold}:")

            seg_dict = result['pcd_dict_merged_']
            print(f'Unique groups global: {set(seg_dict["group"])}')
            seg_dict["group"] = num_to_natural(remove_small_group(seg_dict["group"], th))
            print(f'Unique groups global: {set(seg_dict["group"])}')

            gen_coord = torch.tensor(seg_dict["coord"]).cuda().contiguous().float()
            gen_group = seg_dict["group"]
            indices, dis = pointops.knn_query(1, gen_coord, offset, scene_coord, new_offset)
            indices = indices.cpu().numpy()
            group = gen_group[indices.reshape(-1)].astype(np.int16)
            mask_dis = dis.reshape(-1).cpu().numpy() > 0.6
            group[mask_dis] = -1
            group = group.astype(np.int16)
            labels_new = np.array(group)
            print(f"Unique groups global: {set(labels_new)}")

            # Convert coefficients tuple to a string, replace spaces and commas with underscores
            coefficients_str = "_".join(map(str, coefficients)).replace(" ", "").replace(",", "_")

            threshold_str = str(threshold).replace(" ", "").replace(",", "_")

            # Use the coefficients string in the filename
            filename = f"{scene_name}_{threshold_str}_{coefficients_str}_global.pth"
            torch.save(num_to_natural(group), join(save_path, filename))

            # Store labels_new in the list
            labels_new_list.append((coefficients, threshold, labels_new))

    # Get GT data for the scene from saved files
    gt_path = join(gt_data_path, scene_name + "_gt.pth")
    if os.path.exists(gt_path):
        gt_dict = torch.load(gt_path)
        coord_gt = gt_dict.get("coord", None)
        labels_gt20 = gt_dict.get("labels_gt20", None)
        colors_gt20 = gt_dict.get("colors_gt20", None)
        labels_gt200 = gt_dict.get("labels_gt200", None)
        colors_gt200 = gt_dict.get("colors_gt200", None)
        labels_gt_instance = gt_dict.get("labels_gt_instance", None)
        colors_gt_instance = gt_dict.get("colors_gt_instance", None)
        labels_gt_nyu = gt_dict.get("labels_nyu", None)
        colors_gt_nyu = gt_dict.get("colors_nyu", None)
        print(f"GT data is loaded from {gt_path}")
        del gt_dict
        gc.collect()

    # Comparisons

    # Initialize an empty dictionary for storing the comparison results
    comparison_results_dict = {}

    #gt_methods = {"nyu40", "ScanNet20", "ScanNet200", "Instance"}
    #gt_methods = {"ScanNet200"}
    gt_methods = {"Instance"}

    gt_method_semantic = "ScanNet200"
    gt_method_instance = "Instance"

    if gt_method_semantic == "nyu40":
        labels_gt_sem = labels_gt_nyu
    elif gt_method_semantic == "ScanNet20":
        labels_gt_sem = labels_gt20
    elif gt_method_semantic == "ScanNet200":
        labels_gt_sem = labels_gt200
    else:
        print("GT semantic method chosen is not available!!!")

    if gt_method_instance == "Instance":
        labels_gt_ins = labels_gt_instance
    else:
        print("GT instance method chosen is not available!!!")
    
    baseline_metrics = compare_segmentation_output(labels, labels_gt_ins, labels_gt_sem, method="baseline", gt=gt_method_instance)
    comparison_results_dict[f"{gt_method_instance}_baseline"] = baseline_metrics

    global_metrics_list = compare_segmentation_output(labels_new_list, labels_gt_ins, labels_gt_sem, method="global", gt=gt_method_instance)
    for idx, global_metrics in enumerate(global_metrics_list):
        print(global_metrics['threshold'])
        th = global_metrics['threshold']
        comparison_results_dict[f"{gt_method_instance}_global_coeff_{idx}_threshold_{th}"] = global_metrics
    
    if verbose_comparisons:
        metrics_to_plot = ["iou", "pq", "precision", "recall", "f1"]
        for metric in metrics_to_plot:
            plot_comparison_metrics(baseline_metrics, global_metrics_list, metric)

    num_parts = 1
    total_items = len(comparison_results_dict)
    part_size = math.ceil(total_items / num_parts)

    # Split the dictionary and save each part
    keys_list = list(comparison_results_dict.keys())

    for part_num in range(num_parts):
        # Get the range of keys for this part
        start_idx = part_num * part_size
        end_idx = start_idx + part_size

        # Slice the keys list to get the keys for this part
        part_keys = keys_list[start_idx:end_idx]
        
        # Create a dictionary for this part
        part_dict = {key: comparison_results_dict[key] for key in part_keys}
        
        # Save this part dictionary
        part_path = join(save_path, f"{scene_name}_comparisons_output_part{part_num + 1}.pth")

        print(f"A part of comparison results is ready to save {part_path}")
        torch.save(part_dict, part_path)
        print(f"Comparison results part {part_num + 1} saved to {part_path}")


def plot_comparison_metrics(baseline_metrics, global_metrics_list, metric_name):
    # Extract overall and groupwise metrics
    baseline_overall = baseline_metrics[f"{metric_name}_metrics"]["overall"]
    baseline_groupwise = baseline_metrics[f"{metric_name}_metrics"]["groupwise"]

    for _, global_metrics in enumerate(global_metrics_list):
        threshold = global_metrics["threshold"]
        coefficients = global_metrics["coefficients"]

    global_overall_values = [metrics["metrics"][f"{metric_name}_metrics"]["overall"] for metrics in global_metrics_list]
    global_groupwise_values = [metrics["metrics"][f"{metric_name}_metrics"]["groupwise"] for metrics in global_metrics_list]

    # Get all unique groups/classes from the baseline and global groupwise metrics
    groups = sorted(set(baseline_groupwise.keys()).union(*[set(gw.keys()) for gw in global_groupwise_values]))
    groups.append("Overall")  # Add an 'Overall' label at the end

    # Prepare data for plotting
    baseline_values = [baseline_groupwise.get(group, np.nan) for group in groups[:-1]] + [baseline_overall]
    global_values = [[gw.get(group, np.nan) for group in groups[:-1]] + [overall] for gw, overall in zip(global_groupwise_values, global_overall_values)]

    # Plotting
    bar_width = 0.35
    index = np.arange(len(groups))

    fig, ax = plt.subplots(figsize=(12, 7))

    # Plotting baseline bars
    ax.bar(index, baseline_values, bar_width, label="SAM3D")

    # Plotting global bars (each bar set corresponds to a different threshold/coefficients combination)
    for i, global_val in enumerate(global_values):
        ax.bar(index + (i + 1) * bar_width, global_val, bar_width, label=f"SAM3D-G")

    ax.set_xlabel('Instances/Classes')
    ax.set_ylabel(metric_name.upper())
    rounded_coef = [round(c, 3) for c in coefficients]
    ax.set_title(f'Comparison of Baseline and Global Methods for {metric_name.upper()}, th: {round(threshold,3)}, {rounded_coef}')
    ax.set_xticks(index + bar_width * (len(global_values) / 2))
    ax.set_xticklabels(groups)
    ax.legend()

    plt.tight_layout()
    plt.show()

def compare_segmentation_output(labels_list, labels_gt_instance, labels_gt_semantic, method="baseline", gt="Instance"):
    def calculate_and_store_metrics(labels):
        accuracy_metrics = calculate_segmentation_accuracy_iou(labels, labels_gt_instance, labels_gt_semantic)
        iou_dict, mean_iou = compute_iou(accuracy_metrics["remapped_predicted_groups"], accuracy_metrics["ground_truth_groups"])
        pq_dict, overall_pq = compute_pq(accuracy_metrics["remapped_predicted_groups"], accuracy_metrics["ground_truth_groups"], accuracy_metrics["ground_truth_classes_semantic"])
        precision_dict, recall_dict, f1_dict, mean_precision, mean_recall, mean_f1 = compute_metrics(accuracy_metrics["remapped_predicted_groups"], accuracy_metrics["ground_truth_groups"])

        iou_metrics = {
            "overall": mean_iou,
            "groupwise": iou_dict
        }

        pq_metrics = {
            "overall": overall_pq,
            "groupwise": pq_dict
        }

        f1_metrics = {
            "overall": mean_f1,
            "groupwise": f1_dict
        }

        recall_metrics = {
            "overall": mean_recall,
            "groupwise": recall_dict
        }

        precision_metrics = {
            "overall": mean_precision,
            "groupwise": precision_dict
        }

        metrics = {
            "accuracy_metrics": {
                "overall": accuracy_metrics["overall"],
                "groupwise": accuracy_metrics["groupwise"]
            },
            "iou_metrics": iou_metrics,
            "pq_metrics": pq_metrics,
            "f1_metrics": f1_metrics,
            "recall_metrics": recall_metrics,
            "precision_metrics": precision_metrics,
            "ground_truth_groups": accuracy_metrics["ground_truth_groups"],
            "ground_truth_classes_semantic": accuracy_metrics["ground_truth_classes_semantic"],
            "remapped_predicted_groups": accuracy_metrics["remapped_predicted_groups"]
        }
        
        return metrics
    
    if method == "baseline":
        labels = labels_list
        metrics = calculate_and_store_metrics(labels)
        return metrics
    
    elif method == "global":
        metrics_list = []
        for coefficients, threshold, labels in labels_list:
            metrics = calculate_and_store_metrics(labels)
            metrics_list.append({
                "coefficients": coefficients,
                "threshold": threshold,
                "metrics": metrics
            })
        return metrics_list
    
    else:
        labels = labels_list
        metrics = calculate_and_store_metrics(labels)
        
        return metrics

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

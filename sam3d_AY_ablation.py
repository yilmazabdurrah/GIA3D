"""
Main Script

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

verbose = False # To print out intermediate data
verbose_graph = False # To plot correspondence graphs before and after optimization
verbose_comparisons = True # To plot comparison output

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
    #intrinsic_path = join(rgb_path, scene_name,'intrinsics_depth.txt') 
    intrinsic_path = join(rgb_path, scene_name, 'intrinsics', 'intrinsic_depth.txt')
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
def cal_graph(input_dict, new_input_dict, match_inds, coefficients=[0.25, 0.25, 0.25, 0.25]):

    L1 = coefficients[0]
    L2 = coefficients[0]
    L3 = coefficients[0]
    L4 = coefficients[0]

    group_0 = input_dict["group"]
    group_1 = new_input_dict["group"]
    coord_0 = input_dict["coord"]
    coord_1 = new_input_dict["coord"]
    view_ids_0 = input_dict["viewpoint_id"]
    view_ids_1 = new_input_dict["viewpoint_id"]
    view_names_0 = input_dict["viewpoint_name"]
    view_names_1 = new_input_dict["viewpoint_name"]

    unique_nodes_0 = list(set(zip(view_names_0, group_0)))
    unique_nodes_1 = list(set(zip(view_names_1, group_1)))

    features_0 = input_dict["feature"]
    features_1 = new_input_dict["feature"]
    stability_scores_0 = input_dict["stability_score"]
    stability_scores_1 = new_input_dict["stability_score"]
    predicted_ious_0 = input_dict["predicted_iou"]
    predicted_ious_1 = new_input_dict["predicted_iou"]

    # Calculate the group number correspondence of overlapping points
    group_overlap = {}
    point_cnt_group_0 = {}
    point_cnt_group_1 = {}
    cost = {}

    unique_values_group_0 = set(group_0)

    for unique_value in unique_values_group_0:
        point_cnt_group_0[unique_value] = sum(1 for element in group_0 if element == unique_value)

    unique_values_group_1 = set(group_1)

    for unique_value in unique_values_group_1:
        point_cnt_group_1[unique_value] = sum(1 for element in group_1 if element == unique_value)

    #print("Counts for group 0 ", point_cnt_group_0)
    #print("Counts for group 1 ", point_cnt_group_1)

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
            cost[overlap_key] = 0
        group_overlap[overlap_key] += 1
        '''if group_i == -1 and group_j == -1:
            cost[overlap_key] += 3
        elif group_i == -1 or group_j == -1:
            continue
        else:'''
        cost[overlap_key] += L1*normalized_feature_difference(feature_i,feature_j) + L2*abs(stability_score_i - stability_score_j) + L3*abs(predicted_iou_i - predicted_iou_j)

    #print("cost: ", cost)

    # Initialize the graph
    correspondence_graph = nx.DiGraph()   

    for node in unique_nodes_0:
        correspondence_graph.add_node(node)
    for node in unique_nodes_1:
        correspondence_graph.add_node(node)

    # Add edges based on group_overlap
    for (group_i, view_id_i, view_name_i, point_cnt_group_i, group_j, view_id_j, view_name_j, point_cnt_group_j), count in group_overlap.items():
        cost[(group_i, view_id_i, view_name_i, point_cnt_group_i, group_j, view_id_j, view_name_j, point_cnt_group_j)] /= count
        cost[(group_i, view_id_i, view_name_i, point_cnt_group_i, group_j, view_id_j, view_name_j, point_cnt_group_j)] += L4*max(0,(1 - count/min(point_cnt_group_i,point_cnt_group_j)))
        #cost[(group_i, view_id_i, view_name_i, point_cnt_group_i, group_j, view_id_j, view_name_j, point_cnt_group_j)] /= 4 # divide the number of components in cost function
        correspondence_graph.add_edge(
                (view_name_i, group_i), 
                (view_name_j, group_j), 
                count_common=count,
                count_total=[point_cnt_group_i,point_cnt_group_j],
                cost = cost[(group_i, view_id_i, view_name_i, point_cnt_group_i, group_j, view_id_j, view_name_j, point_cnt_group_j)],
                viewpoint_id_0=view_id_j, 
                viewpoint_id_1=view_id_i
            )

    return correspondence_graph, input_dict, new_input_dict

def cal_scenes(pcd_list, index, voxel_size, voxelize, th=50, coefficients=[0.25, 0.25, 0.25, 0.25]):
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
                correspondence_graph, input_dict_0, input_dict_1 = cal_graph(input_dict_0, input_dict_1, match_inds, coefficients)
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
        for mask_id in mask_ids:
            if (viewpoint_name, mask_id) in final_group_map:
                new_groups.append(final_group_map[(viewpoint_name, mask_id)])
            else:
                new_groups.append(mask_id)
        
        all_coords.append(pcd["coord"])
        all_colors.append(pcd["color"])
        all_groups.append(new_groups)
    
    # Combine all data into a single dictionary
    pcd_dict = {
        "coord": np.concatenate(all_coords, axis=0),
        "color": np.concatenate(all_colors, axis=0),
        "group": np.concatenate(all_groups, axis=0)
    }

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

    if os.path.exists(join(save_path, scene_name + ".pth")):
        return

    # Step 1 in pipeline: SAM Generate Masks
    
    step1_output_path = os.path.join(save_path, scene_name + "_step1.pkl")

    # Returns the names of the multi-images in the scene
    color_names = sorted(os.listdir(join(rgb_path, scene_name, 'color')), key=lambda a: int(os.path.basename(a).split('.')[0]))
        
    voxelize_new = Voxelize(voxel_size=args.voxel_size, mode="train", keys=("coord", "color", "group", "feature", "predicted_iou", "stability_score"))

    # If the output of Step 1 and 2 already exists, load it
    if os.path.exists(step1_output_path):
        with open(step1_output_path, 'rb') as f:
            pcd_list, pcd_list_ = pickle.load(f)
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

            '''if verbose:
                # Extract data from tvoxelizedhe dictionary
                coords = pcd_dict['coord']
                colors = pcd_dict['color']
                group_ids = pcd_dict['group']
                #viewpoint_name = pcd_dict['viewpoint_name']
                viewpoint_ids = pcd_dict['viewpoint_id']
                stability_scores = pcd_dict['stability_score']
                predicted_ious = pcd_dict['predicted_iou']
                features = pcd_dict['feature']

                # Print the header
                print(f"{'Viewpoint ID':<10} {'Coord':<65} {'Color':<20} {'Group ID':<10} {'Stability Score':<20} {'Prediction IOU':<20} {'Feature':<40}")

                for viewpoint_id, coord, color, group_id, stability_score, predicted_iou, feature in zip(viewpoint_ids, coords, colors, group_ids, stability_scores, predicted_ious, features):
                    coord_str = ', '.join(map(str, coord))
                    color_str = ', '.join(map(str, color))
                    feature_str = ', '.join(map(str, feature))
                    #if group_id == -1:
                    print(f"{viewpoint_id:<10} {coord_str:<65} {color_str:<20} {group_id:<10} {stability_score:<20} {predicted_iou:<20} {feature_str:<40}")'''

            pcd_dict_ = voxelize_new(pcd_dict)
            pcd_dict = voxelize(pcd_dict)

            if verbose:
                # Extract data from the dictionary
                coords = pcd_dict_['coord']
                colors = pcd_dict_['color']
                group_ids = pcd_dict_['group']
                #viewpoint_name = pcd_dict['viewpoint_name']
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
                    #if group_id == -1:
                    print(f"{viewpoint_id:<10} {coord_str:<65} {color_str:<20} {group_id:<10} {stability_score:<20} {predicted_iou:<20} {feature_str:<40}")

            pcd_list.append(pcd_dict)
            pcd_list_.append(pcd_dict_)
    
        # Step 2 in pipeline: Merge Two Adjacent Pointclouds until get single point cloud
        while len(pcd_list) != 1:
            print(len(pcd_list), flush=True)
            new_pcd_list = []
            for indice in pairwise_indices(len(pcd_list)):
                #print(indice)
                pcd_frame = cal_2_scenes(pcd_list, indice, voxel_size=voxel_size, voxelize=voxelize)
                if pcd_frame is not None:
                    new_pcd_list.append(pcd_frame)
            pcd_list = new_pcd_list
        
        # Save the output of Step 1 and 2 to a file for future use
        with open(step1_output_path, 'wb') as f:
            pickle.dump((pcd_list, pcd_list_), f)
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

        step_size = 0.1
        coefficients_range = np.arange(0, 1.1, step_size)
        
        # Generate all possible combinations of lambda values
        for combination in itertools.product(coefficients_range, repeat=4):
            if sum(combination) == 1:
                coefficients = list(combination)
                #print("coefficients: ", coefficients) # 258 combinations we have

        # ADD ITERATION HERE TO SEE THE RESULT FOR EACH COMBINATION

        merged_graph = nx.DiGraph()
        for indice in range(len(pcd_list_)):
            corr_graph, pcd_list_ = cal_scenes(pcd_list_, indice, voxel_size=voxel_size, voxelize=voxelize_new, coefficients=coefficients) 
            if len(corr_graph.nodes) > 0 and len(corr_graph.edges) > 0:
                merged_graph = nx.compose(merged_graph, corr_graph)
                #print(corr_graph.edges(data=True))
        #print(merged_graph.edges(data=True))

        # Create cost matrix
        node_list = list(merged_graph.nodes())
        node_index = {node: idx for idx, node in enumerate(node_list)}

        # Initialize the cost matrix with infinity
        large_value = 1e9  # Define a large value to replace infinity
        cost_matrix = np.full((len(node_list), len(node_list)), large_value)

        for u, v, data in merged_graph.edges(data=True):
            i, j = node_index[u], node_index[v]
            cost_matrix[i, j] = data['cost']
            cost_matrix[j, i] = data['cost'] 

        # Convert the cost matrix to a condensed distance matrix for hierarchical clustering
        np.fill_diagonal(cost_matrix, 0)
        condensed_cost_matrix = squareform(cost_matrix)
    
        # Perform hierarchical clustering
        Z = linkage(condensed_cost_matrix, method='average')

        # Set a threshold to determine clusters (tune this threshold based on the data)
        threshold = 0.2
        clusters = fcluster(Z, t=threshold, criterion='distance')

        node_to_cluster = {node: cluster for node, cluster in zip(node_list, clusters)}

        # Create a mapping of merged nodes
        merged_nodes = {}
        for cluster in set(clusters):
            cluster_nodes = [node for node, c in node_to_cluster.items() if c == cluster]
            representative = cluster_nodes[0]
            for node in cluster_nodes:
                if node != representative:
                    merged_nodes[node] = representative

        # Merge clusters and update the graph
        merged_graph_min = nx.DiGraph()

        # Add representative nodes to the new graph
        for cluster in set(clusters):
            cluster_nodes = [node for node, c in node_to_cluster.items() if c == cluster]
            representative = cluster_nodes[0]
            merged_graph_min.add_node(representative)

        # Add edges to the new graph only between the representative nodes
        for u, v, data in merged_graph.edges(data=True):
            v_pos = merged_nodes.get(u)

            if v_pos is not None and v_pos == v:
                if not merged_graph_min.has_edge(u, v):
                    merged_graph_min.add_edge(u, v, **data)

        # # Print nodes and edges of the new graph
        # print("Nodes in the minimized graph:")
        # for node in merged_graph_min.nodes():
        #     print(node)

        # print("\nEdges in the minimized graph:")
        # for u, v, data in merged_graph_min.edges(data=True):
        #     print(f"From {u} to {v}: {data}")

        pcd_dict_merged_ = update_groups_and_merge_dictionaries(pcd_list_, merged_graph_min)

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
    for i in range(2):
        if i == 0:
            seg_dict = pcd_list[0]
        elif i == 1:
            seg_dict = pcd_dict_merged_
        else: 
            seg_dict = pcd_list[0]
        seg_dict["group"] = num_to_natural(remove_small_group(seg_dict["group"], th))

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
        if i == 0:
            #torch.save(num_to_natural(group), join(save_path, scene_name + ".pth"))
            labels = np.array(group)
            coord_ = data_dict.get("coord", None)
        elif i == 1:
            #torch.save(num_to_natural(group), join(save_path, scene_name + "_new" + ".pth"))
            labels_new = np.array(group)
        else: 
            #torch.save(num_to_natural(group), join(save_path, scene_name + ".pth"))
            labels = np.array(group)

    # Comparisons

    # Get GT data for the scene from dictionary

    # Access the ground truth labels
    semantic_gt20 = data_dict.get("semantic_gt20", None)
    semantic_gt200 = data_dict.get("semantic_gt200", None)
    instance_gt = data_dict.get("instance_gt", None)

    nyu40class_mapping = {value["id"]: value["name"] for key, value in nyu40_colors_to_class.items()}
    nyu40class_list = list(nyu40class_mapping.items())

    ScanNet20class_mapping = {value["id"]: value["name"] for key, value in ScanNet20_colors_to_class.items()}
    ScanNet20class_list = list(ScanNet20class_mapping.items())

    ScanNet200class_mapping = {value["id"]: value["name"] for key, value in ScanNet200_colors_to_class.items()}
    ScanNet200class_list = list(ScanNet200class_mapping.items())
    
    # for class_id, class_name in nyu40class_mapping.items():
        # print(f"ID: {class_id}, Class: {class_name}")
    
    # for class_id, class_name in ScanNet200class_mapping.items():
        # print(f"ID: {class_id}, Class: {class_name}")

    # Check if the labels exist and print their shapes
    if semantic_gt20 is not None:
        #print(f'semantic_gt20 shape: {semantic_gt20.shape}')
        unique_labels20 = set(semantic_gt20)
        #unique_labels20 = {label + 1 for label in set(semantic_gt20)}
        print(f"Unique labels20: {unique_labels20}")
        for label in unique_labels20:
            if label < 0 or label >= len(ScanNet20class_list):
                print(f"Label {label}: {ScanNet20class_mapping.get(label, 'Unknown')}")
            else: 
                label, name = ScanNet20class_list[label]
                print(f"Label {label}: {ScanNet20class_mapping.get(label, 'Unknown')}")
    else:
        print('semantic_gt20 not found in the data_dict')

    if semantic_gt200 is not None:
        #print(f'semantic_gt200 shape: {semantic_gt200.shape}')
        unique_labels200 = set(semantic_gt200)
        #unique_labels200 = {label + 1 for label in set(semantic_gt200)}
        print(f"Unique labels200: {unique_labels200}")
        #print(list(ScanNet200class_mapping)[0])
        for label in unique_labels200:
            if label < 0 or label >= len(ScanNet200class_list):
                print(f"Label {label}: {ScanNet200class_mapping.get(label, 'Unknown')}")
            else:
                label, name = ScanNet200class_list[label]
                print(f"Label {label}: {ScanNet200class_mapping.get(label, 'Unknown')}")
    else:
        print('semantic_gt200 not found in the data_dict')

    if instance_gt is not None:
        #print(f'instance_gt shape: {instance_gt.shape}')
        unique_instances = set(instance_gt)
        print(f"Unique instances: {unique_instances}")
    else:
        print('instance_gt not found in the data_dict')

    # Get GT data for the scene nyu40class

    if os.path.exists(join(gt_data_path, scene_name, scene_name + "_vh_clean_2.labels.ply")):
        print(f"Ground truth data exists for {scene_name}")

        points_gt, colors_gt = load_ply(join(gt_data_path, scene_name, scene_name + "_vh_clean_2.labels.ply"))

        labels_gt = get_labels_from_colors(colors_gt)
        
        data_dict = {
            "coord": points_gt,
            "labels": labels_gt,
            "colors": colors_gt
        }
        torch.save(data_dict, join(save_path, scene_name + "_gt.pth"))

        accuracy_metrics = calculate_segmentation_accuracy_iou(labels, labels_gt)
        
        accuracy_metrics_new = calculate_segmentation_accuracy_iou(labels_new, labels_gt)
        if verbose_comparisons:
            plot_accuracy_metrics(accuracy_metrics, scene_name)
            plot_accuracy_metrics(accuracy_metrics_new, scene_name)
        else:
            print(f"Results for classic one")
            print(f"Overall Accuracy: {accuracy_metrics['overall_accuracy']}")
            for instance, accuracy in accuracy_metrics['instance_accuracy'].items():
                print(f"Instance {instance} Accuracy: {accuracy}")
            
            print(f"Results for updated one")
            print(f"Overall Accuracy: {accuracy_metrics_new['overall_accuracy']}")
            for instance, accuracy in accuracy_metrics_new['instance_accuracy'].items():
                print(f"Instance {instance} Accuracy: {accuracy}")
    else:
        print(f"Ground truth data does not exist for {scene_name} to compare segmentation results")
    

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

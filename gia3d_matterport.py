"""
Main Script by
Abdurrahman Yilmaz (ayilmaz@lincoln.ac.uk) v01 for applying GIA3D on Matterport Dataset
24 Apr 2025 (v01)
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

import gc

from segment_anything import build_sam, SamAutomaticMaskGenerator

save_mask = False
verbose_graph = True

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

def get_matterport_pcd(scene_name, color_img_path, depth_img_path, color_img_name, intrinsics_path, pose_path, save_2dmask_path, mask_generator=None):
    """
    Generate a 3D point cloud from Matterport RGB-D data.
    
    Args:
        scene_name (str): Name of the scene processed.
        color_img_path (str): Path to the undistorted color image (JPG).
        depth_img_path (str): Path to the undistorted depth image (16-bit PNG, 0.25mm units).
        color_img_name (str): Base name of the undistorted color image.
        intrinsics_path (str): Path to the intrinsics file (3x3 matrix).
        pose_path (str): Path to the camera-to-world pose file (4x4 matrix).
        save_2dmask_path (str): Path to the where to store 2d masks.
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
        if save_mask:
            if not os.path.exists(os.path.join(save_2dmask_path, scene_name, 'masks')):
                os.makedirs(os.path.join(save_2dmask_path, scene_name, 'masks'))
            img = Image.fromarray(num_to_natural(group_ids).astype(np.int16), mode='I;16')
            mask_img_full_path = os.path.join(save_2dmask_path, scene_name, 'masks', color_img_name + '_mask.png')
            img.save(mask_img_full_path)
            print(f"Saved mask image as {mask_img_full_path}")
        group_ids = group_ids[mask]
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

    pcd_dict = dict(
        coord=points_world,
        color=colors,
        group=group_ids,
        stability_score=stability_scores,
        predicted_iou=predicted_ious,
        feature=features
    )

    return pcd_dict

def seg_pcd(scene_name, mask_generator, voxelize, voxel_size, th, save_path, save_2dmask_path, gt_data_path):
    ################################################################################################
    ###################### Step 1 in pipeline: SAM Generate Masks ##################################
    ################################################################################################
    print("Step 1", flush=True)
    pcd_list = []
    # Extract relevant data from zipped files within each scene
    with zipfile.ZipFile(os.path.join(scene_path, 'undistorted_color_images.zip'), 'r') as rgb_zip, \
            zipfile.ZipFile(os.path.join(scene_path, 'undistorted_depth_images.zip'), 'r') as depth_zip, \
            zipfile.ZipFile(os.path.join(scene_path, 'undistorted_camera_parameters.zip'), 'r') as camera_param_zip:
        
        # Read the .conf file from camera parameters zip
        conf_file_name = f"{scene_name}/undistorted_camera_parameters/{scene_name}.conf"
        with camera_param_zip.open(conf_file_name) as conf_file:
            conf_data = conf_file.read().decode('utf-8')
            _, scan_entries = parse_conf_file(conf_data)

        # For demo, take only first 10 viewpoints
        #scan_entries = scan_entries[:50]
        max_indice = len(scan_entries) - 1
        print(f"Number of viewpoints in the list: {len(scan_entries)}", flush=True)
        # Process each viewpoint (each scan entry in conf file)
        for idx, entry in enumerate(scan_entries):
            print(f"Processing view {idx}/{max_indice} of {scene_name} using RGB: {entry['color_file']} and Depth: {entry['depth_file']} images", flush=True)
            
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

                    color_img_name_base = os.path.splitext(entry['color_file'])[0]

                    pcd_dict = get_matterport_pcd(
                        scene_name=scene_name,
                        color_img_path=tmp_color.name,
                        depth_img_path=tmp_depth.name,
                        color_img_name=color_img_name_base,
                        intrinsics_path=tmp_intrinsics.name,
                        pose_path=tmp_pose.name,
                        save_2dmask_path=save_2dmask_path,
                        mask_generator=mask_generator
                    )
                    if len(pcd_dict["coord"]) == 0:
                        continue

                    viewpoint_ids = [idx + 1] * len(pcd_dict["coord"])
                    viewpoint_names = [color_img_name_base] * len(pcd_dict["coord"])

                    print(f"before voxelization  len pcd_dict[group]: {len(pcd_dict['coord'])}", flush=True)

                    pcd_dict.update(viewpoint_id=viewpoint_ids, viewpoint_name=viewpoint_names)
                    pcd_dict = voxelize(pcd_dict)

                    print(f"after voxelization len pcd_dict[group]: {len(pcd_dict['coord'])}", flush=True)
                    
                    pcd_list.append(pcd_dict)

                    group_ids = pcd_dict["group"]
                    unique_groups = np.unique(group_ids).astype(int)
                    print(f"Unique groups for view {idx}/{max_indice} of {scene_name}: {unique_groups}")

                    if save_mask:
                        if not os.path.exists(os.path.join(save_2dmask_path, scene_name, 'masks')):
                            os.makedirs(os.path.join(save_2dmask_path, scene_name, 'masks'))

                        # Save PLY color file
                        pcd = o3d.geometry.PointCloud()
                        pcd.points = o3d.utility.Vector3dVector(pcd_dict["coord"])
                        pcd.colors = o3d.utility.Vector3dVector(pcd_dict["color"] / 255.0)  # normalize RGB to [0,1]
                        ply_filename = os.path.join(save_2dmask_path, scene_name, 'masks', f"{color_img_name_base}.ply")
                        o3d.io.write_point_cloud(ply_filename, pcd)
                        print(f"Saved, colored point cloud as {ply_filename}")

                        # Save PLY instance file
                        num_groups = len(unique_groups)
                        group_colors = generate_colors(num_groups)
                        group_color_map = {group: group_colors[i] for i, group in enumerate(unique_groups)}
                        seg_colors = np.array([group_color_map[int(g)] for g in group_ids])
                        seg_pcd = o3d.geometry.PointCloud()
                        seg_pcd.points = o3d.utility.Vector3dVector(pcd_dict["coord"])
                        seg_pcd.colors = o3d.utility.Vector3dVector(seg_colors)
                        seg_ply_filename = os.path.join(save_2dmask_path, scene_name, 'masks', f"{color_img_name_base}_mask.ply")
                        o3d.io.write_point_cloud(seg_ply_filename, seg_pcd)
                        print(f"Saved, instance segmentation point cloud as {seg_ply_filename}")

                        # Save RGB image file
                        color_img = cv2.imread(tmp_color.name)
                        color_save_path = os.path.join(save_2dmask_path, scene_name, 'masks', f"{color_img_name_base}_rgb.png")
                        cv2.imwrite(color_save_path, color_img)
                        print(f"Saved, color image as {color_save_path}")

                    # Clean up temporary files
                    os.unlink(tmp_depth.name)
                    os.unlink(tmp_color.name)
                    os.unlink(tmp_pose.name)
                    os.unlink(tmp_intrinsics.name)
                    
                    print(f"Instance segmentation results for view {idx} of {scene_name} assigned to a dictionary", flush=True)

            except Exception as e:
                print(f"Error processing view {idx} for scene {scene_name}: {e}")
                continue
    ################################################################################################
    ### Step 2 in pipeline: Merge All Pointclouds in one shot globally to get single point cloud ###
    ################################################################################################
    print("Step 2", flush=True)
    if len(pcd_list) != 1:

        for index in range(1, len(pcd_list)):
            # Get the 'group' value of the elements
            group_index_last = pcd_list[index - 1]["group"]
            group_index = pcd_list[index]["group"]

            # Update the current 'group' array
            group_index[group_index != -1] += group_index_last.max() + 1
            pcd_list[index]["group"] = group_index

        for id, pcd in enumerate(pcd_list):
            group_ids = pcd["group"]
            unique_groups = np.unique(group_ids).astype(int)
            print(f"Unique groups for view {id}/{max_indice} of {scene_name}: {unique_groups}")

        coefficients = [[0.07, 0.057, 0.063, 0.811]]
        threshold = 0.3

        merged_graph = nx.DiGraph()
        merged_graph_min = nx.DiGraph()

        for indice in range(len(pcd_list)):
            print(f"Current viewpoint: {indice}/{max_indice}")
            corr_graph, pcd_list = cal_scenes(pcd_list, indice, voxel_size=voxel_size, voxelize=voxelize, th=th, coefficient_combinations=coefficients) 
            if len(corr_graph.nodes) > 0 and len(corr_graph.edges) > 0:
                merged_graph = nx.compose(merged_graph, corr_graph)
            del corr_graph
            gc.collect()

        # Initialize the cost matrix with infinity
        large_value = 1e9  # Define a large value to replace infinity

        # Prepare to store merged graphs for each coefficient combination
        merged_graphs = {}
        merged_dicts = {}
        results = {}
        for coefficients in coefficients:
            results[tuple(coefficients)] = {}

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
        
            merged_graphs[tuple(coefficients)] = merged_graph_min
            pcd_dict_merged = update_groups_and_merge_dictionaries(pcd_list, merged_graph_min)
            merged_dicts[tuple(coefficients)] = pcd_dict_merged
            results[tuple(coefficients)][threshold] = {
                'pcd_dict_merged': pcd_dict_merged
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
    
    ################################################################################################
    ######################## Step 3 in pipeline: Region Merging Method #############################
    ################################################################################################
    print("Step 3", flush=True)
    labels_list = []
    for coefficients, merged_dicts in results.items():
        for threshold, result in merged_dicts.items():
            print(f"\nProcessing for coefficients {coefficients} and threshold {threshold}:")

            seg_dict = result['pcd_dict_merged']
            print(f'Unique groups GIA3D: {set(seg_dict["group"])}')
            seg_dict["group"] = num_to_natural(remove_small_group(seg_dict["group"], th))
            print(f'Unique groups GIA3D: {set(seg_dict["group"])}')

            data_path = os.path.join(gt_data_path,scene_name,'house_segmentations',f"{scene_name}.ply")
            pcd_gt_scene = o3d.io.read_point_cloud(data_path)
            coords_np = np.asarray(pcd_gt_scene.points)
            scene_coord = torch.tensor(coords_np, dtype=torch.float32).cuda().contiguous()
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
            print(f"Unique groups GIA3D: {set(labels)}", flush=True)
            labels_list.append((coefficients, threshold, labels))

            unique_labels = np.unique(labels)
            label_to_color = {}
            color_palette = generate_colors(len(unique_labels))

            for i, label in enumerate(unique_labels):
                label_to_color[label] = color_palette[i]

            # Assign colors to points based on their labels
            colors = np.array([label_to_color[label] for label in labels])

            # Create and save point cloud
            gia3d_pcd = o3d.geometry.PointCloud()
            gia3d_pcd.points = o3d.utility.Vector3dVector(coords_np)
            gia3d_pcd.colors = o3d.utility.Vector3dVector(colors)
            
            coefficients_str = "_".join(map(str, coefficients)).replace(" ", "").replace(",", "_")
            threshold_str = str(threshold).replace(" ", "").replace(",", "_")
            filename = f"GIA3D_{threshold_str}_{coefficients_str}.ply"

            save_segmented_ply_path = os.path.join(save_path, scene_name, filename)
            os.makedirs(os.path.dirname(save_segmented_ply_path), exist_ok=True)
            o3d.io.write_point_cloud(save_segmented_ply_path, gia3d_pcd)
            print(f"Segmented PC with GIA3D for {scene_name} is saved to {save_segmented_ply_path}")

def cal_scenes(pcd_list, index, voxel_size, voxelize, th=50, coefficient_combinations=[[0.25, 0.25, 0.25, 0.25]]):
    input_dict_0 = pcd_list[index]
    input_dict_1 = {}
    pcd0 = make_open3d_point_cloud(input_dict_0, voxelize, th)
    merged_graph = nx.DiGraph()
    for i, pcd_dict in enumerate(pcd_list):
        if i != index: # i > index
            input_dict_1.update(pcd_dict)
            group_ids = input_dict_1["group"]
            unique_groups = np.unique(group_ids).astype(int)
            print(f"A: Unique groups for view j (input): {unique_groups}")
            pcd1 = make_open3d_point_cloud(input_dict_1, voxelize, th)
            group_ids = input_dict_1["group"]
            unique_groups = np.unique(group_ids).astype(int)
            print(f"B: Unique groups for view j (input): {unique_groups}")
            if pcd0 == None:
                if pcd1 == None:                    
                    return merged_graph, pcd_list
                else:
                    pcd_list[i].update(input_dict_1)
                    return merged_graph, pcd_list
            elif pcd1 == None:
                pcd_list[index].update(input_dict_0)
                return merged_graph, pcd_list

            group_ids = input_dict_1["group"]
            unique_groups = np.unique(group_ids).astype(int)
            print(f"C: Unique groups for view j (input): {unique_groups}")

            # Cal Dul-overlap
            match_inds = get_matching_indices(pcd1, pcd0, 1.5 * voxel_size, 1)
            if match_inds:
                correspondence_graph, input_dict_0, input_dict_1 = cal_graph(input_dict_0, input_dict_1, match_inds, coefficient_combinations)
                pcd_list[i].update(input_dict_1)
                pcd_list[index].update(input_dict_0)
                if len(correspondence_graph.nodes) > 0 and len(correspondence_graph.edges) > 0:
                    merged_graph = nx.compose(merged_graph, correspondence_graph)
                    # if verbose_graph:
                    #     # Create a figure for subplots
                    #     plt.figure(figsize=(15, 12))

                    #     # Draw the first graph
                    #     draw_graph(correspondence_graph, "Correspondence Graph between two viewpoints", 121)

                    #     # Display the plot
                    #     plt.show()

    return merged_graph, pcd_list

def cal_graph(input_dict, new_input_dict, match_inds, coefficient_combinations=[[0.25, 0.25, 0.25, 0.25]]):

    global maxVal
    global maxVal_stab
    global maxVal_pred
    global maxVal_iou

    group_0 = input_dict["group"]
    group_1 = new_input_dict["group"]
    print(f"Unique groups for view i: {set(group_0)}")
    print(f"Unique groups for view j: {set(group_1)}")

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

    unique_nodes_0 = list(set(zip(view_ids_0, group_0)))
    unique_nodes_1 = list(set(zip(view_ids_1, group_1)))
    print(f"Unique Nodes for view i: {unique_nodes_0}")
    print(f"Unique Nodes for view j: {unique_nodes_1}")

    # Initialize the graph
    correspondence_graph = nx.DiGraph()   

    for node in unique_nodes_0:
        correspondence_graph.add_node(node)
    for node in unique_nodes_1:
        correspondence_graph.add_node(node)

    print(f"correspondence_graph: {correspondence_graph}")

    # Calculate the group number correspondence of overlapping points
    point_cnt_group_0 = {}
    point_cnt_group_1 = {}
    
    unique_values_group_0 = set(group_0)

    for unique_value in unique_values_group_0:
        point_cnt_group_0[unique_value] = sum(1 for element in group_0 if element == unique_value)

    unique_values_group_1 = set(group_1)

    for unique_value in unique_values_group_1:
        point_cnt_group_1[unique_value] = sum(1 for element in group_1 if element == unique_value)

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
            (view_id_i, group_i), 
            (view_id_j, group_j), 
            **edge_data
        )
        #print(f"key: {key} and count: {count} and cost: {edge_data['cost']}")
    print(f"correspondence_graph: {correspondence_graph}")

    return correspondence_graph, input_dict, new_input_dict

def normalized_feature_difference(f_i, f_j):
    # Calculate the Euclidean norm of the difference
    difference_norm = np.linalg.norm(f_i - f_j)
    # Normalize the difference norm
    normalized_difference = difference_norm / len(f_i)
    return normalized_difference

def update_groups_and_merge_dictionaries(pcd_list, merged_graph_min):
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

    # Iterate over the pcd_list and update group information
    for pcd in pcd_list:
        viewpoint_id = pcd["viewpoint_id"][0]
        mask_ids = pcd["group"]
        
        # Create a mapping from old mask ids to new group ids
        new_groups = []
        cnt = 0
        for mask_id in mask_ids:
            if (viewpoint_id, mask_id) in final_group_map:
                #print(f"mask_id: {mask_id}, mapping: {final_group_map[(viewpoint_name, mask_id)]}")
                new_groups.append(final_group_map[(viewpoint_id, mask_id)])
            else:
                new_groups.append(-1)
                cnt+=1
                print(f"Not in dictionary {cnt}")
        
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

def make_open3d_point_cloud(input_dict, voxelize, th):
    print(f"len input_dict[group]: {len(input_dict['group'])}", flush=True)
    input_dict["group"] = remove_small_group(input_dict["group"], th)
    # input_dict = voxelize(input_dict)

    xyz = input_dict["coord"]
    if np.isnan(xyz).any():
        return None
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    return pcd

def get_args():
    '''Command line arguments.'''
    parser = argparse.ArgumentParser(description='Segment Anything on Matterport dataset.')
    parser.add_argument('--scans_file_path', type=str, required=True, help='Path to scans.txt file containing scene names')
    parser.add_argument('--scans_root', type=str, required=True, help='Root path to the Matterport dataset scans folder')
    parser.add_argument('--save_path', type=str, required=True, help='Directory where to save the PCD results')
    parser.add_argument('--save_2dmask_path', type=str, default='', help='Directory where to save 2D segmentation results')
    parser.add_argument('--sam_checkpoint_path', type=str, required=True, help='Path to the SAM checkpoint')
    parser.add_argument('--voxel_size', type=float, default=0.05, help='Voxel size for downsampling')
    parser.add_argument('--th', type=int, default=10, help='Threshold for ignoring small groups to reduce noise')
    parser.add_argument('--category_mapping_path', type=str, default='', help='Path to Matterport dataset semantic class id mapping')
    parser.add_argument('--gt_data_path', type=str, default='', help='The path of 3D ground truth data (color, panoptic segmented)')
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    print("Arguments:")
    print(args)

    os.makedirs(args.save_path, exist_ok=True)

    mask_generator = SamAutomaticMaskGenerator(build_sam(checkpoint=args.sam_checkpoint_path).to(device="cuda"))
    voxelize = Voxelize(voxel_size=args.voxel_size, mode="train", keys=("coord", "color", "group", "feature", "predicted_iou", "stability_score"))

    # Load scene names from scans.txt
    with open(args.scans_file_path, 'r') as scans_file:
        scene_names = scans_file.read().splitlines()

    for scene_name in scene_names:
        scene_path = os.path.join(args.scans_root, scene_name)
        
        if not os.path.isdir(scene_path):
            print(f"Warning: Scene folder not found for {scene_name}")
            continue
        
        print(f"Processing scene {scene_name}", flush=True)

        seg_pcd(scene_name, mask_generator, voxelize, args.voxel_size, args.th, args.save_path, args.save_2dmask_path, args.gt_data_path)

        print(f"Scene {scene_name} processed", flush=True)
        
        
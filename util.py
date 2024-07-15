import numpy as np
import torch
import open3d as o3d
import os
import copy
from PIL import Image
import json
# import clip
import pointops
from sklearn.metrics import accuracy_score
from scipy.optimize import linear_sum_assignment

import matplotlib.pyplot as plt
import networkx as nx

from plyfile import PlyData

SCANNET_COLOR_MAP_20 = {-1: (0., 0., 0.), 0: (174., 199., 232.), 1: (152., 223., 138.), 2: (31., 119., 180.), 3: (255., 187., 120.), 4: (188., 189., 34.), 5: (140., 86., 75.),
                        6: (255., 152., 150.), 7: (214., 39., 40.), 8: (197., 176., 213.), 9: (148., 103., 189.), 10: (196., 156., 148.), 11: (23., 190., 207.), 12: (247., 182., 210.), 
                        13: (219., 219., 141.), 14: (255., 127., 14.), 15: (158., 218., 229.), 16: (44., 160., 44.), 17: (112., 128., 144.), 18: (227., 119., 194.), 19: (82., 84., 163.)}

# Define the dictionary mapping NYU40 class IDs to their RGB values and label names for GT data
nyu40_colors_to_class = {
    (0, 0, 0): {"id": -1, "name": "Unknown"},
    (174, 199, 232): {"id": 1, "name": "Wall"},
    (152, 223, 138): {"id": 2, "name": "Floor"},
    (31, 119, 180): {"id": 3, "name": "Cabinet"},
    (255, 187, 120): {"id": 4, "name": "Bed"},
    (188, 189, 34): {"id": 5, "name": "Chair"},
    (140, 86, 75): {"id": 6, "name": "Sofa"},
    (255, 152, 150): {"id": 7, "name": "Table"},
    (214, 39, 40): {"id": 8, "name": "Door"},
    (197, 176, 213): {"id": 9, "name": "Window"},
    (148, 103, 189): {"id": 10, "name": "Bookshelf"},
    (196, 156, 148): {"id": 11, "name": "Picture"},
    (23, 190, 207): {"id": 12, "name": "Counter"},
    (178, 76, 76): {"id": 13, "name": "Blinds"},
    (247, 182, 210): {"id": 14, "name": "Desk"},
    (66, 188, 102): {"id": 15, "name": "Shelves"},
    (219, 219, 141): {"id": 16, "name": "Curtain"},
    (140, 57, 197): {"id": 17, "name": "Dresser"},
    (202, 185, 52): {"id": 18, "name": "Pillow"},
    (51, 176, 203): {"id": 19, "name": "Mirror"},
    (200, 54, 131): {"id": 20, "name": "Floor mat"},
    (92, 193, 61): {"id": 21, "name": "Clothes"},
    (78, 71, 183): {"id": 22, "name": "Ceiling"},
    (172, 114, 82): {"id": 23, "name": "Books"},
    (255, 127, 14): {"id": 24, "name": "Refrigerator"},
    (91, 163, 138): {"id": 25, "name": "Television"},
    (153, 98, 156): {"id": 26, "name": "Paper"},
    (140, 153, 101): {"id": 27, "name": "Towel"},
    (158, 218, 229): {"id": 28, "name": "Shower curtain"},
    (100, 125, 154): {"id": 29, "name": "Box"},
    (178, 127, 135): {"id": 30, "name": "Whiteboard"},
    (120, 185, 128): {"id": 31, "name": "Person"},
    (146, 111, 194): {"id": 32, "name": "Nightstand"},
    (44, 160, 44): {"id": 33, "name": "Toilet"},
    (112, 128, 144): {"id": 34, "name": "Sink"},
    (96, 207, 209): {"id": 35, "name": "Lamp"},
    (227, 119, 194): {"id": 36, "name": "Bathtub"},
    (213, 92, 176): {"id": 37, "name": "Bag"},
    (94, 106, 211): {"id": 38, "name": "Other structure"},
    (82, 84, 163): {"id": 39, "name": "Other furniture"},
    (100, 85, 144): {"id": 40, "name": "Other prop"},
}

class Voxelize(object):
    def __init__(self,
                 voxel_size=0.05,
                 hash_type="fnv",
                 mode='train',
                 keys=("coord", "normal", "color", "label"),
                 return_discrete_coord=False,
                 return_min_coord=False):
        self.voxel_size = voxel_size
        self.hash = self.fnv_hash_vec if hash_type == "fnv" else self.ravel_hash_vec
        assert mode in ["train", "test"]
        self.mode = mode
        self.keys = keys
        self.return_discrete_coord = return_discrete_coord
        self.return_min_coord = return_min_coord

    def __call__(self, data_dict):
        assert "coord" in data_dict.keys()
        discrete_coord = np.floor(data_dict["coord"] / np.array(self.voxel_size)).astype(int)
        min_coord = discrete_coord.min(0) * np.array(self.voxel_size)
        discrete_coord -= discrete_coord.min(0)
        key = self.hash(discrete_coord)
        idx_sort = np.argsort(key)
        key_sort = key[idx_sort]
        _, inverse, count = np.unique(key_sort, return_inverse=True, return_counts=True)   
        if self.mode == 'train':  # train mode
            # idx_select = np.cumsum(np.insert(count, 0, 0)[0:-1]) + np.random.randint(0, count.max(), count.size) % count
            idx_select = np.cumsum(np.insert(count, 0, 0)[0:-1])
            idx_unique = idx_sort[idx_select]
            if self.return_discrete_coord:
                data_dict["discrete_coord"] = discrete_coord[idx_unique]
            if self.return_min_coord:
                data_dict["min_coord"] = min_coord.reshape([1, 3])
            for key in self.keys:
                data_dict[key] = data_dict[key][idx_unique]
            return data_dict

        elif self.mode == 'test':  # test mode
            data_part_list = []
            for i in range(count.max()):
                idx_select = np.cumsum(np.insert(count, 0, 0)[0:-1]) + i % count
                idx_part = idx_sort[idx_select]
                data_part = dict(index=idx_part)
                for key in data_dict.keys():
                    if key in self.keys:
                        data_part[key] = data_dict[key][idx_part]
                    else:
                        data_part[key] = data_dict[key]
                if self.return_discrete_coord:
                    data_part["discrete_coord"] = discrete_coord[idx_part]
                if self.return_min_coord:
                    data_part["min_coord"] = min_coord.reshape([1, 3])
                data_part_list.append(data_part)
            return data_part_list
        else:
            raise NotImplementedError

    @staticmethod
    def ravel_hash_vec(arr):
        """
        Ravel the coordinates after subtracting the min coordinates.
        """
        assert arr.ndim == 2
        arr = arr.copy()
        arr -= arr.min(0)
        arr = arr.astype(np.uint64, copy=False)
        arr_max = arr.max(0).astype(np.uint64) + 1

        keys = np.zeros(arr.shape[0], dtype=np.uint64)
        # Fortran style indexing
        for j in range(arr.shape[1] - 1):
            keys += arr[:, j]
            keys *= arr_max[j + 1]
        keys += arr[:, -1]
        return keys

    @staticmethod
    def fnv_hash_vec(arr):
        """
        FNV64-1A
        """
        assert arr.ndim == 2
        # Floor first for negative coordinates
        arr = arr.copy()
        arr = arr.astype(np.uint64, copy=False)
        hashed_arr = np.uint64(14695981039346656037) * np.ones(arr.shape[0], dtype=np.uint64)
        for j in range(arr.shape[1]):
            hashed_arr *= np.uint64(1099511628211)
            hashed_arr = np.bitwise_xor(hashed_arr, arr[:, j])
        return hashed_arr


def overlap_percentage(mask1, mask2):
    intersection = np.logical_and(mask1, mask2)
    area_intersection = np.sum(intersection)

    area_mask1 = np.sum(mask1)
    area_mask2 = np.sum(mask2)

    smaller_area = min(area_mask1, area_mask2)

    return area_intersection / smaller_area


def remove_samll_masks(masks, ratio=0.8):
    filtered_masks = []
    skip_masks = set()

    for i, mask1_dict in enumerate(masks):
        if i in skip_masks:
            continue

        should_keep = True
        for j, mask2_dict in enumerate(masks):
            if i == j or j in skip_masks:
                continue
            mask1 = mask1_dict["segmentation"]
            mask2 = mask2_dict["segmentation"]
            overlap = overlap_percentage(mask1, mask2)
            if overlap > ratio:
                if np.sum(mask1) < np.sum(mask2):
                    should_keep = False
                    break
                else:
                    skip_masks.add(j)

        if should_keep:
            filtered_masks.append(mask1)

    return filtered_masks


def draw_graph(graph, title, subplot_position):
    plt.subplot(subplot_position)
    pos = nx.shell_layout(graph)
    nx.draw(graph, pos, with_labels=True, font_weight='bold', node_size=500)

    # Get all edges in the graph
    edges = graph.edges()

    # Draw edge labels for both count and score
    for edge in edges:
        count_common = graph[edge[0]][edge[1]]['count_common']
        cost = graph[edge[0]][edge[1]]['cost']
        count_cost_label = f"Count: {count_common}, Cost: {cost}"
        nx.draw_networkx_edge_labels(graph, pos, edge_labels={edge: count_cost_label})

    plt.title(title)

def plot_accuracy_metrics(accuracy_metrics, scene_name):
    """
    Plot accuracy metrics.
    
    Parameters:
        accuracy_metrics (dict): Dictionary containing overall accuracy and instance-wise accuracy.
    """
    # Extract overall accuracy and instance-wise accuracy
    overall_accuracy = accuracy_metrics['overall_accuracy']
    instance_accuracy = accuracy_metrics['instance_accuracy']

    # Prepare data for plotting
    group_ids = list(instance_accuracy.keys())
    accuracies = list(instance_accuracy.values())
    
    # Add overall accuracy
    group_ids.append('Overall')
    accuracies.append(overall_accuracy)

    # Convert group IDs to string for labeling
    group_labels = [str(group) for group in group_ids]
    
    # Plotting
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(group_labels)), accuracies, color='skyblue')
    plt.xlabel('Group IDs and Overall')
    plt.ylabel('Accuracy Rate')
    plt.title(f'Accuracy Metrics for Group IDs and Overall - Scene: {scene_name}')
    plt.ylim(-0.1, 1.1)  # Accuracy is between 0 and 1
    plt.xticks(ticks=range(len(group_labels)), labels=group_labels, rotation=45)
    plt.show()

def load_ply(file_path):
    """
    Load a PLY file and return the points and their colors.

    :param file_path: Path to the PLY file.
    :return: List of points and their corresponding colors.
    """
    plydata = PlyData.read(file_path)
    vertex_data = plydata['vertex']
    points = np.vstack([vertex_data['x'], vertex_data['y'], vertex_data['z']]).T
    colors = np.vstack([vertex_data['red'], vertex_data['green'], vertex_data['blue']]).T
    return points, colors

def calculate_segmentation_accuracy(coords_, predicted_labels, ground_truth_labels):
    """
    Calculate the segmentation accuracy metrics.
    
    Parameters:
        coords_ (np.ndarray): Coordinates of the points (not used in accuracy calculation).
        predicted_labels (np.ndarray): Predicted group labels.
        ground_truth_labels (np.ndarray): Ground truth group labels.
    
    Returns:
        dict: Dictionary containing overall accuracy and instance-wise accuracy.
    """
    
    # Flatten the group arrays
    predicted_groups = predicted_labels.flatten()
    ground_truth_groups = ground_truth_labels.flatten()

    # Calculate the overlap matrix
    unique_pred = np.unique(predicted_groups)
    unique_gt = np.unique(ground_truth_groups)
    
    overlap_matrix = np.zeros((len(unique_gt), len(unique_pred)), dtype=int)
    
    for i, gt_label in enumerate(unique_gt):
        for j, pred_label in enumerate(unique_pred):
            overlap_matrix[i, j] = np.sum((ground_truth_groups == gt_label) & (predicted_groups == pred_label))
            #print(f"gt_label: {gt_label}, pred_label: {pred_label}, overlap_matrix[{i}, {j}]: {overlap_matrix[i, j]}")
    
    # Find the best match using the Hungarian algorithm
    row_ind, col_ind = linear_sum_assignment(-overlap_matrix)
    
    # Create a mapping from predicted to ground truth labels
    label_mapping = {unique_pred[col]: unique_gt[row] for row, col in zip(row_ind, col_ind)}

    # Debug statements
    #print("Unique predicted labels:", unique_pred)
    #print("Unique ground truth labels:", unique_gt)
    #print("Initial label mapping:", label_mapping)

    # Handle unmapped predicted labels
    remaining_pred_labels = set(unique_pred) - set(label_mapping.keys())
    
    while remaining_pred_labels:
        # For each remaining predicted label, find the best matching ground truth label
        for pred_label in list(remaining_pred_labels):
            overlaps = np.array([np.sum((predicted_groups == pred_label) & (ground_truth_groups == gt_label)) for gt_label in unique_gt])
            best_gt_label_index = np.argmax(overlaps)
            best_gt_label = unique_gt[best_gt_label_index]
            
            # Update the mapping
            label_mapping[pred_label] = best_gt_label
            remaining_pred_labels.remove(pred_label)
    
    # Print the updated label mapping
    #print("Final label mapping:", label_mapping)
    
    # Remap predicted labels
    remapped_predicted_groups = np.array([label_mapping.get(label, -1) for label in predicted_groups])
        
    # Check if all predicted labels are in the mapping
    missing_labels = [label for label in predicted_groups if label not in label_mapping]
    if missing_labels:
        print("Missing labels in label mapping:", missing_labels)
        
    # Calculate overall accuracy
    overall_accuracy = accuracy_score(ground_truth_groups, remapped_predicted_groups)
    
    # Calculate instance-wise accuracy
    instance_accuracy = {}
    for instance in unique_gt:
        instance_mask = (ground_truth_groups == instance)
        instance_accuracy[instance] = accuracy_score(
            ground_truth_groups[instance_mask],
            remapped_predicted_groups[instance_mask]
        )
    
    return {
        "overall_accuracy": overall_accuracy,
        "instance_accuracy": instance_accuracy
    }

def to_numpy(x):
    if isinstance(x, torch.Tensor):
        x = x.clone().detach().cpu().numpy()
    assert isinstance(x, np.ndarray)
    return x


def save_point_cloud(coord, color=None, file_path="pc.ply", logger=None):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    coord = to_numpy(coord)
    #print("Coord: ", coord)
    #print(coord.size)
    if color is not None:
        color = to_numpy(color)
    #print("Color: ", color)
    #print(color.size)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coord)
    pcd.colors = o3d.utility.Vector3dVector(np.ones_like(coord) if color is None else color)
    o3d.io.write_point_cloud(file_path, pcd)
    if logger is not None:
        logger.info(f"Save Point Cloud to: {file_path}")

def read_point_cloud(file_path="pc.ply", logger=None):
    
    if not os.path.exists(file_path):
        if logger:
            logger.error(f"File {file_path} does not exist.")
        return None, None

    pcd = o3d.io.read_point_cloud(file_path)
    if not pcd:
        if logger:
            logger.error(f"Failed to read point cloud from {file_path}.")
        return None, None

    coord = np.asarray(pcd.points)
    color = np.asarray(pcd.colors) if np.asarray(pcd.colors).size != 0 else None

    if logger:
        logger.info(f"Read Point Cloud from: {file_path}")

    return coord, color


def remove_small_group(group_ids, th):
    unique_elements, counts = np.unique(group_ids, return_counts=True)
    result = group_ids.copy()
    for i, count in enumerate(counts):
        if count < th:
            result[group_ids == unique_elements[i]] = -1
    
    return result


def pairwise_indices(length):
    return [[i, i + 1] if i + 1 < length else [i] for i in range(0, length, 2)]


def num_to_natural(group_ids):
    '''
    Change the group number to natural number arrangement
    '''
    if np.all(group_ids == -1):
        return group_ids
    array = copy.deepcopy(group_ids)
    unique_values = np.unique(array[array != -1])
    mapping = np.full(np.max(unique_values) + 2, -1)
    mapping[unique_values + 1] = np.arange(len(unique_values))
    array = mapping[array + 1]
    return array


def get_matching_indices(pcd0, pcd1, search_voxel_size, K=None):
    match_inds = []
    scene_coord = torch.tensor(np.asarray(pcd0.points)).cuda().contiguous().float()
    new_offset = torch.tensor(scene_coord.shape[0]).cuda().float()
    gen_coord = torch.tensor(np.asarray(pcd1.points)).cuda().contiguous().float()
    offset = torch.tensor(gen_coord.shape[0]).cuda().float()

    indices, dis = pointops.knn_query(1, gen_coord, offset, scene_coord, new_offset)
    indices = indices.cpu().numpy()[:,0]
    dis = dis.cpu().numpy()[:,0]

    for i in range(scene_coord.shape[0]):
        if dis[i] < search_voxel_size:
            match_inds.append([i,indices[i]])
    
    return match_inds


def visualize_3d(data_dict, text_feat_path, save_path):
    text_feat = torch.load(text_feat_path)
    group_logits = np.einsum('nc,mc->nm', data_dict["group_feat"], text_feat)
    group_labels = np.argmax(group_logits, axis=-1)
    labels = group_labels[data_dict["group"]]
    labels[data_dict["group"] == -1] = -1
    visualize_pcd(data_dict["coord"], data_dict["color"], labels, save_path)


def visualize_pcd(coord, pcd_color, labels, save_path, islabel=False):
    # alpha = 0.5
    if islabel:
        #label_color = np.array([SCANNET_COLOR_MAP_20[label] for label in labels])
        # overlay = (pcd_color * (1-alpha) + label_color * alpha).astype(np.uint8) / 255
        save_point_cloud(coord, labels, save_path)
    else:
        pcd_color = pcd_color / 255
        save_point_cloud(coord, pcd_color, save_path) 

def visualize_2d(img_color, labels, img_size, save_path):
    import matplotlib.pyplot as plt
    # from skimage.segmentation import mark_boundaries
    # from skimage.color import label2rgb
    label_names = ["wall", "floor", "cabinet", "bed", "chair",
           "sofa", "table", "door", "window", "bookshelf",
           "picture", "counter", "desk", "curtain", "refridgerator",
           "shower curtain", "toilet", "sink", "bathtub", "other"]
    colors = np.array(list(SCANNET_COLOR_MAP_20.values()))[1:]
    segmentation_color = np.zeros((img_size[0], img_size[1], 3))
    for i, color in enumerate(colors):
        segmentation_color[labels == i] = color
    alpha = 1
    overlay = (img_color * (1-alpha) + segmentation_color * alpha).astype(np.uint8)
    fig, ax = plt.subplots()
    ax.imshow(overlay)
    patches = [plt.plot([], [], 's', color=np.array(color)/255, label=label)[0] for label, color in zip(label_names, colors)]
    plt.legend(handles=patches, bbox_to_anchor=(0.5, -0.1), loc='upper center', ncol=4, fontsize='small')
    plt.savefig(save_path, bbox_inches='tight')
    plt.show()


def visualize_partition(coord, group_id, save_path):
    group_id = group_id.reshape(-1)
    num_groups = group_id.max() + 1
    group_colors = np.random.rand(num_groups, 3)
    group_colors = np.vstack((group_colors, np.array([0,0,0])))
    color = group_colors[group_id]
    save_point_cloud(coord, color, save_path)


def delete_invalid_group(group, group_feat):
    indices = np.unique(group[group != -1])
    group = num_to_natural(group)
    group_feat = group_feat[indices]
    return group, group_feat

def get_labels_from_colors(colors):
    """
    Map RGB colors to class labels.

    :param colors: List of RGB colors.
    :return: List of class labels.
    """
    labels = []
    for color in colors:
        color_tuple = tuple(color)
        if color_tuple in nyu40_colors_to_class:
            labels.append(nyu40_colors_to_class[color_tuple]["id"])
        else:
            labels.append(-1)  # Unknown label
    return np.array(labels, dtype=np.int16)
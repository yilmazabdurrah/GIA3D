"""
Save GT Data

Author: Yunhan Yang (yhyang.myron@gmail.com)

Updated by
Abdurrahman Yilmaz (ayilmaz@lincoln.ac.uk) v06
22 Jul 2024
"""

import os
import numpy as np
import torch
import multiprocessing as mp
import argparse

from itertools import repeat
from PIL import Image
from os.path import join
from util import *

def get_gt_data(scene_name, data_path, save_path, train_scenes, val_scenes, gt_data_path):
    print(scene_name, flush=True)
    
    if scene_name in train_scenes:
        scene_path = join(data_path, "train", scene_name + ".pth")
    elif scene_name in val_scenes:
        scene_path = join(data_path, "val", scene_name + ".pth")
    else: 
        scene_path = join(data_path, "test", scene_name + ".pth")
    data_dict = torch.load(scene_path)

    coord_ = data_dict.get("coord", None)

    # Get GT data for the scene from dictionary

    # Access the ground truth labels
    semantic_gt20 = data_dict.get("semantic_gt20", None)
    semantic_gt200 = data_dict.get("semantic_gt200", None)
    instance_gt = data_dict.get("instance_gt", None)

    nyu40class_mapping = {value["id"]: value["name"] for key, value in nyu40_colors_to_class.items()}
    nyu40color_mapping = {value["id"]: key for key, value in nyu40_colors_to_class.items()}
    nyu40class_list = list(nyu40class_mapping.items())

    ScanNet20class_mapping = {value["id"]: value["name"] for key, value in ScanNet20_colors_to_class.items()}
    ScanNet20color_mapping = {value["index"]: key for key, value in ScanNet20_colors_to_class.items()}
    ScanNet20label_mapping = {value["index"]: value["id"] for key, value in ScanNet20_colors_to_class.items()}
    ScanNet20class_list = list(ScanNet20class_mapping.items())

    ScanNet200class_mapping = {value["id"]: value["name"] for key, value in ScanNet200_colors_to_class.items()}
    ScanNet200color_mapping = {value["index"]: key for key, value in ScanNet200_colors_to_class.items()}
    ScanNet200label_mapping = {value["index"]: value["id"] for key, value in ScanNet200_colors_to_class.items()}
    ScanNet200class_list = list(ScanNet200class_mapping.items())

    # Check if the labels exist and print their shapes
    if semantic_gt20 is not None:
        print(f"Semantic ground truth data exists for {scene_name} by ScanNet20 class")
        #print(f'semantic_gt20 shape: {semantic_gt20.shape}')
        unique_labels20 = set(semantic_gt20)
        for label in unique_labels20:
            if label < 0 or label >= len(ScanNet20class_list):
                print(f"Label {label}: {ScanNet20class_mapping.get(label, 'Unknown')}")
            else: 
                label, name = ScanNet20class_list[label]
        colors_gt20 = np.array([ScanNet20color_mapping[idx] for idx in semantic_gt20])
        colors_gt20 = [tuple(color) for color in colors_gt20]
        labels_gt20 = np.array([ScanNet20label_mapping[idx] for idx in semantic_gt20])

        print(f"Unique colors20: {set(colors_gt20)}")
        print(f"Unique labels20: {set(labels_gt20)}")
    else:
        print(f'semantic_gt20 not found in the data_dict for {scene_name}')

    if semantic_gt200 is not None:
        print(f"Semantic ground truth data exists for {scene_name} by ScanNet200 class")
        #print(f'semantic_gt200 shape: {semantic_gt200.shape}')
        unique_labels200 = set(semantic_gt200)
        #print(list(ScanNet200class_mapping)[0])
        for label in unique_labels200:
            if label < 0 or label >= len(ScanNet200class_list):
                print(f"Label {label}: {ScanNet200class_mapping.get(label, 'Unknown')}")
            else:
                label, name = ScanNet200class_list[label]
                print(f"Label {label}: {ScanNet200class_mapping.get(label, 'Unknown')}")
        colors_gt200 = np.array([ScanNet200color_mapping[idx] for idx in semantic_gt200])
        colors_gt200 = [tuple(color) for color in colors_gt200]
        labels_gt200 = np.array([ScanNet200label_mapping[idx] for idx in semantic_gt200])
    else:
        print(f'semantic_gt200 not found in the data_dict for {scene_name}')

    if instance_gt is not None:
        print(f"Instance ground truth data exists for {scene_name}")
        
        unique_instances = set(instance_gt)
        # Dictionary to store the color for each unique instance
        colorsMap = {}
        
        for instance in unique_instances:
            color = generate_unique_color(colorsMap.values())  # For random unique colors
            colorsMap[instance] = color
        
        labels_gt_instance = instance_gt
        colors_gt_instance = [colorsMap[instance] for instance in instance_gt]
    else:
        print(f'instance_gt not found in the data_dict for {scene_name}')

    # Get GT data for the scene by nyu40class

    if os.path.exists(join(gt_data_path, scene_name, scene_name + "_vh_clean_2.labels.ply")):
        print(f"Semantic ground truth data exists for {scene_name} by nyu40 class")

        points_gt, colors_gt = load_ply(join(gt_data_path, scene_name, scene_name + "_vh_clean_2.labels.ply"))

        labels_gt = get_labels_from_colors(colors_gt, gt_class="nyu40")
    else:
        print(f"Semantic ground truth data does not exist for {scene_name} by nyu40 class to compare segmentation results")  

    
    # Prepare the gt_dict
    gt_dict = {
        "coord": coord_,
    }

    # Add ground truth data if available
    if labels_gt20 is not None:
        gt_dict["labels_gt20"] = labels_gt20
        gt_dict["colors_gt20"] = colors_gt20

    if labels_gt200 is not None:
        gt_dict["labels_gt200"] = labels_gt200
        gt_dict["colors_gt200"] = colors_gt200

    if points_gt is not None:
        gt_dict["coord_nyu"] = points_gt
        gt_dict["labels_nyu"] = labels_gt
        gt_dict["colors_nyu"] = colors_gt

    if instance_gt is not None:
        gt_dict["labels_gt_instance"] = labels_gt_instance
        gt_dict["colors_gt_instance"] = colors_gt_instance

    # Save the gt_dict
    torch.save(gt_dict, join(save_path, scene_name + "_gt.pth"))

def get_args():
    '''Command line arguments.'''

    parser = argparse.ArgumentParser(
        description='Segment Anything on ScanNet.')
    parser.add_argument('--rgb_path', type=str, help='the path of rgb data')
    parser.add_argument('--data_path', type=str, default='', help='the path of pointcload data')
    parser.add_argument('--save_path', type=str, help='Where to save the pcd results')
    parser.add_argument('--scannetv2_train_path', type=str, default='scannet-preprocess/meta_data/scannetv2_train.txt', help='the path of scannetv2_train.txt')
    parser.add_argument('--scannetv2_val_path', type=str, default='scannet-preprocess/meta_data/scannetv2_val.txt', help='the path of scannetv2_val.txt')
    parser.add_argument('--img_size', default=[640,480])
    parser.add_argument('--th', default=50, help='threshold of ignoring small groups to avoid noise pixel')
    parser.add_argument('--gt_data_path', type=str, help='the path of nyu ground truth point clouds')

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
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    scene_names = sorted(os.listdir(args.rgb_path))
    for scene_name in scene_names:
        get_gt_data(scene_name, args.data_path, args.save_path, train_scenes, val_scenes, args.gt_data_path)

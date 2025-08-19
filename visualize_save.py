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
processed_dir = os.path.join(base_dir, 'scannet-preprocess/processed_dataset/point_cloud/*/')
processed_dir2 = os.path.join(base_dir, 'output_global_merger_comparisons/')
processed_dir3 = '/media/ayilmaz/Crucial X9/ScanNetv2_Dataset/comparisons/'
output_dir = os.path.join(base_dir, 'output_global_merger_comparisons/PCs_/')

# Processed data paths
processed_files = os.listdir(processed_dir2)
#processed_files = os.listdir(processed_dir3)

nyu40_color_to_class_id = {v["id"]: k for k, v in nyu40_colors_to_class.items()}
ScanNet20_color_to_class_id = {v["index"]: k for k, v in ScanNet20_colors_to_class.items()}
ScanNet200_color_to_class_id = {v["index"]: k for k, v in ScanNet200_colors_to_class.items()}
#print(set(ScanNet200_color_to_class_id))

nyu40_color_to_class_id_list = list(nyu40_color_to_class_id)
ScanNet20_color_to_class_id_list = list(ScanNet20_color_to_class_id)
ScanNet200_color_to_class_id_list = list(ScanNet200_color_to_class_id)

raw_saved = False
for filename in processed_files:
    if filename.endswith('_global.pth'):
        parts = filename.split('_')
        print(parts)
        if len(parts) >= 6:
            base_filename = '_'.join(parts[:-6])  # Adjust the number according to your file structure
        else:
            base_filename = '_'.join(parts[:-1])
        print("base_filename: ", base_filename)
        pcd_filepath = os.path.join(processed_dir, f'{base_filename}.pth')
        filename_ = base_filename
        print(pcd_filepath)
    elif filename.endswith('_baseline.pth'):
        pcd_filepath = os.path.join(processed_dir, f'{filename[:-13]}.pth')
        print(pcd_filepath)
        filename_ = filename[:-13]
    if filename.endswith('_global.pth') or filename.endswith('_baseline.pth'):
        pcd_seg_filepath = os.path.join(processed_dir2, filename)
    else:
        continue

    # Load point cloud data
    pcd_files = glob.glob(pcd_filepath)
    pcd_data = torch.load(pcd_files[0])
    seg_data = torch.load(pcd_seg_filepath)

    # Get the coordinates and colors
    coord = pcd_data['coord'].astype('float64')
    color = pcd_data['color'].astype('float64')
    labels = np.array(seg_data)

    # Get unique labels and generate color map
    color_map = {}
    used_colors = set()
    unique_labels = np.unique(seg_data)
    for label in unique_labels:
        if label == -1:
            color_map[label] = (0, 0, 0)
        else:
            color_map[label] = generate_unique_color(used_colors)
    label_colors = np.array([color_map[label] for label in labels])

    pcd_seg_file_savepath = os.path.join(output_dir, f'{filename[:-4]}_with_labels.ply')

    pcd_file_savepath = os.path.join(output_dir, f'{filename_}_raw.ply')

    # Save point cloud with labels
    visualize_pcd(coord, color, label_colors, pcd_seg_file_savepath, True)
    print(f"Saved: {pcd_seg_file_savepath}")
    if not raw_saved:
        visualize_pcd(coord, color, label_colors, pcd_file_savepath, False)
        print(f"Saved: {pcd_file_savepath}")


'''for filename in processed_files:
    if filename.endswith('_comparisons_input.pth'):
        # Load point cloud data
        print("filename: ", filename)
        pcd_files = glob.glob(os.path.join(processed_dir3, filename))
        print("pcd_files: ", pcd_files)
        pcd_data = torch.load(pcd_files[0])

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

    elif filename.endswith('.pth'):
        print(filename)
        if filename.endswith('_comparisons_output.pth'):
            comparisons_filepath = os.path.join(comparisons_dir, filename)
            comp_files = glob.glob(comparisons_filepath)
            # Check if any comparison files are found
            if not comp_files:
                print(f"No comparison files found for {filename}")
                continue
            comparisons_output_data = torch.load(comp_files[0])

            # Check the keys in the loaded data
            print(f"Loaded comparison output data for {filename}:")
            for key in comparisons_output_data.keys():
                value = comparisons_output_data[key]
                print(f"  Key: {key}, Type: {type(value)}")
                
                # Print some values based on type
                if isinstance(value, dict):
                    # Print a few items from the dictionary
                    print(f"    Dictionary with {len(value)} items")
                    for sub_key, sub_value in list(value.items())[:3]:  # Print only a few items
                        print(f"      Sub-key: {sub_key}, Type: {type(sub_value)}")
                        if isinstance(sub_value, dict):
                            print(f"        Contains keys: {list(sub_value.keys())}")
                        elif isinstance(sub_value, list) or isinstance(sub_value, np.ndarray):
                            print(f"        Sample values: {sub_value[:5]}")  # Print first 5 values
                        else:
                            print(f"        Sample value: {sub_value}")
                elif isinstance(value, list):
                    print(f"    List with {len(value)} items")
                    if len(value) > 0:
                        for idx, item in enumerate(value[:3]):  # Print only a few items
                            print(f"      List item {idx}:")
                            if isinstance(item, dict):
                                print(f"        Contains keys: {list(item.keys())}")
                            elif isinstance(item, np.ndarray):
                                print(f"        Sample values: {item.flatten()[:5]}")  # Print first 5 values
                            else:
                                print(f"        Sample value: {item}")
                elif isinstance(value, np.ndarray):
                    print(f"    Array with shape: {value.shape}")
                    print(f"      Sample values: {value.flatten()[:5]}")  # Print first 5 values
                else:
                    print(f"    Sample value: {value}")

        if filename.endswith('_comparisons_input.pth'):
            comparisons_filepath = os.path.join(comparisons_dir, filename)
            comp_files = glob.glob(comparisons_filepath)
            # Check if any comparison files are found
            if not comp_files:
                print(f"No comparison files found for {filename}")
                continue
            
            # Load the comparison input data
            comparisons_input_data = torch.load(comp_files[0])
            
            # Check the keys in the loaded data
            print(f"Loaded comparison input data for {filename}:")
            for key in comparisons_input_data.keys():
                value = comparisons_input_data[key]
                print(f"  Key: {key}, Type: {type(value)}")
                
                # Print some values based on type
                if isinstance(value, np.ndarray):
                    print(f"    Array with shape: {value.shape}")
                    print(f"      Sample values: {value.flatten()[:5]}")  # Print first 5 values
                elif isinstance(value, list):
                    print(f"    List with {len(value)} items")
                    if len(value) > 0:
                        for idx, item in enumerate(value[:3]):  # Print only a few items
                            print(f"      List item {idx}:")
                            if isinstance(item, np.ndarray):
                                print(f"        Sample values: {item.flatten()[:5]}")  # Print first 5 values
                            else:
                                print(f"        Sample value: {item}")
                else:
                    print(f"    Sample value: {value}")

        if filename.endswith('_global.pth'):
            pcd_filepath = os.path.join(processed_dir, f'{filename[:-8]}.pth')
        elif filename.endswith('_gt.pth'):
            continue
        else:
            continue
        pcd_seg_filepath = os.path.join(processed_dir2, filename)

        # Load point cloud data
        pcd_files = glob.glob(pcd_filepath)
        pcd_data = torch.load(pcd_files[0])
        seg_data = torch.load(pcd_seg_filepath)

        # Get the coordinates and colors
        coord = pcd_data['coord'].astype('float64')
        color = pcd_data['color'].astype('float64')
        labels = np.array(seg_data)

        # Get unique labels and generate color map
        unique_labels = np.unique(seg_data)
        color_map = {label: np.random.rand(3) for label in unique_labels}
        label_colors = np.array([color_map[label] for label in labels])
        
        pcd_seg_file_savepath = os.path.join(output_dir, f'{filename[:-4]}_with_labels.ply')

        pcd_file_savepath = os.path.join(output_dir, f'{filename[:-4]}.ply')

        # Save point cloud with labels
        visualize_pcd(coord, color, label_colors, pcd_seg_file_savepath, True)
        print(f"Saved: {pcd_seg_file_savepath}")
        visualize_pcd(coord, color, label_colors, pcd_file_savepath, False)
        print(f"Saved: {pcd_file_savepath}")

        # Save point cloud with semantic_gt20 labels
        semantic_gt20 = pcd_data.get("semantic_gt20", None)

        if semantic_gt20 is not None:
            label_colors = np.array([ScanNet20_color_to_class_id.get(label, [0, 0, 0]) for label in semantic_gt20])/255.0
            pcd_seg_file_savepath = os.path.join(output_dir, f'{filename[:-4]}_semantic_gt20_with_labels.ply')
            visualize_pcd(coord, color, label_colors, pcd_seg_file_savepath, True)
            print(f"Saved: {pcd_seg_file_savepath}")

        # Save point cloud with semantic_gt200 labels
        semantic_gt200 = pcd_data.get("semantic_gt200", None)
        if semantic_gt200 is not None:
            label_colors = np.array([ScanNet200_color_to_class_id.get(label, [0, 0, 0]) for label in semantic_gt200])/255.0
            pcd_seg_file_savepath = os.path.join(output_dir, f'{filename[:-4]}_semantic_gt200_with_labels.ply')
            visualize_pcd(coord, color, label_colors, pcd_seg_file_savepath, True)
            print(f"Saved: {pcd_seg_file_savepath}")

        # Save point cloud with instance_gt labels
        instance_gt = pcd_data.get("instance_gt", None)
        if instance_gt is not None:
            unique_labels = np.unique(instance_gt)
            color_map = {label: np.random.randint(0, 255, 3) for label in unique_labels}
            label_colors = np.array([color_map[label] for label in instance_gt])
            pcd_seg_file_savepath = os.path.join(output_dir, f'{filename[:-4]}_instance_gt_with_labels.ply')
            visualize_pcd(coord, color, label_colors/255.0, pcd_seg_file_savepath, True)
            print(f"Saved: {pcd_seg_file_savepath}")'''


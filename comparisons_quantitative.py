import numpy as np
import glob
import torch
import os
import json
from util import *
import matplotlib.pyplot as plt

from collections import defaultdict

def print_nested_dict(d, indent=0):
    """Recursively prints keys and values of a nested dictionary with proper indentation."""
    for key, value in d.items():
        print('    ' * indent + f"Key {key}")
        if isinstance(value, dict):
            print_nested_dict(value, indent + 1)
        else:
            print('    ' * (indent + 1) + f"Value {value}")

# Define base directory paths
base_dir = '/home/ayilmaz/ws_segment_3d/SegmentAnything3D/'
#comparisons_dir = os.path.join(base_dir, 'output_global_merger_ablation/th_0_5_merged_nodes/')
#comparisons_dir = os.path.join(base_dir, 'output_global_merger_ablation/th_0_2/')
comparisons_dir = os.path.join(base_dir, 'output_global_merger_comparisons/')

nyu40_color_to_class_id = {v["id"]: k for k, v in nyu40_colors_to_class.items()}
ScanNet20_color_to_class_id = {v["index"]: k for k, v in ScanNet20_colors_to_class.items()}
ScanNet200_color_to_class_id = {v["index"]: k for k, v in ScanNet200_colors_to_class.items()}

ScanNet200_class_id_to_name = {v["id"]: v["name"] for _, v in ScanNet200_colors_to_class.items()}

nyu40_color_to_class_id_list = list(nyu40_color_to_class_id)
ScanNet20_color_to_class_id_list = list(ScanNet20_color_to_class_id)
ScanNet200_color_to_class_id_list = list(ScanNet200_color_to_class_id)

scene_files = {}

# List and filter the processed files in the current scene folder
processed_files = [f for f in os.listdir(comparisons_dir) if f.endswith('.pth') and not (f.endswith('baseline.pth') or f.endswith('global.pth'))]
print(f"processed_files in: {processed_files}")

for filename in processed_files:
    # Extract the scene name part (e.g., 'scene0013_00')
    scene_name = '_'.join(filename.split('_')[:2])
    if scene_name not in scene_files:
        scene_files[scene_name] = []
    # Append the filename with its full path to the list
    scene_files[scene_name].append(os.path.join(comparisons_dir, filename))

# Print the dictionary containing scene files
print(f"scene_files: {scene_files}")

#combined_results_dict = {}

threshold_to_optimal_coefficients_scenes = {}

#gt = "ScanNet200"
gt = "Instance"
filtered_metrics = {}

for scene_name, files in scene_files.items():
    print(f"Processing scene: {scene_name}")
    
    # Initialize a temporary dictionary to hold data from all parts for the current scene
    temp_combined_dict = {}

    # Load each part and update the temporary dictionary
    for filename in sorted(files):
        comparisons_filepath = os.path.join(comparisons_dir, filename)
        print(f"Loading {comparisons_filepath}")
        comparisons_output_data = torch.load(comparisons_filepath)
        temp_combined_dict.update(comparisons_output_data)

    temp_filtered_dict_baseline = {}
    temp_filtered_dict_global = {}
    for key, value in temp_combined_dict.items():
        if key == gt+'_baseline':
            for k, v in temp_combined_dict[key].items():
                if k == 'pq_metrics':
                    temp_filtered_dict_baseline[k] = v
                elif k.endswith('_metrics'):
                    temp_filtered_dict_baseline[k] = v['overall']
        if key.startswith(gt+'_global'):
            for k, v in temp_combined_dict[key]['metrics'].items():
                if k == 'pq_metrics':
                    temp_filtered_dict_global[k] = v
                elif k.endswith('_metrics'):
                    temp_filtered_dict_global[k] = v['overall']
        #print(temp_filtered_dict_baseline)
        #print(temp_filtered_dict_global)
    filtered_metrics[scene_name] = {'baseline': temp_filtered_dict_baseline,
                                    'global': temp_filtered_dict_global}

    #print(filtered_metrics)

metrics_data = {
    'baseline': {},
    'global': {}
}

# Collect all metrics except 'pq_metrics' from all scenes
for scene_name, data in filtered_metrics.items():
    baseline_metrics = data['baseline']
    global_metrics = data['global']
    
    for metric in baseline_metrics:
        if metric != 'pq_metrics':
            if metric not in metrics_data['baseline']:
                metrics_data['baseline'][metric] = []
            metrics_data['baseline'][metric].append(baseline_metrics[metric])

    for metric in global_metrics:
        if metric != 'pq_metrics':
            if metric not in metrics_data['global']:
                metrics_data['global'][metric] = []
            metrics_data['global'][metric].append(global_metrics[metric])
    if baseline_metrics[metric] <= global_metrics[metric]:
        print(scene_name)

aggregated_stats = {
    'baseline': {},
    'global': {}
}

for metric in metrics_data['baseline']:
    baseline_values = np.array(metrics_data['baseline'][metric])
    global_values = np.array(metrics_data['global'][metric])

    aggregated_stats['baseline'][metric] = {
        'mean': np.mean(baseline_values),
        'std': np.std(baseline_values),
        'min': np.min(baseline_values),
        'max': np.max(baseline_values)
    }
    
    aggregated_stats['global'][metric] = {
        'mean': np.mean(global_values),
        'std': np.std(global_values),
        'min': np.min(global_values),
        'max': np.max(global_values)
    }

metrics_to_plot = list(aggregated_stats['baseline'].keys())
print(f"metrics_to_plot: {metrics_to_plot}")
x = np.arange(len(metrics_to_plot))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots(figsize=(10, 6))

# Plot baseline metrics
baseline_means = [aggregated_stats['baseline'][metric]['mean'] for metric in metrics_to_plot]
baseline_stds = [aggregated_stats['baseline'][metric]['std'] for metric in metrics_to_plot]
print(f"baseline_means: {baseline_means} and baseline_stds: {baseline_stds}")
ax.bar(x - width/2, baseline_means, width, yerr=baseline_stds, label='Baseline', capsize=5)

# Plot global metrics
global_means = [aggregated_stats['global'][metric]['mean'] for metric in metrics_to_plot]
global_stds = [aggregated_stats['global'][metric]['std'] for metric in metrics_to_plot]
print(f"global_means: {global_means} and global_stds: {global_stds}")
ax.bar(x + width/2, global_means, width, yerr=global_stds, label='Global', capsize=5)

# Add some text for labels, title, and custom x-axis tick labels, etc.
ax.set_xlabel('Metrics')
ax.set_ylabel('Overall Values')
ax.set_title('Comparison of Baseline and Global Metrics (excluding pq_metrics)')
ax.set_xticks(x)
ax.set_xticklabels(metrics_to_plot, rotation=45, ha="right")
ax.legend()

# Show the plot
plt.tight_layout()
plt.show()

pq_data = {
    'baseline': defaultdict(list),
    'global': defaultdict(list)
}

for scene_name, data in filtered_metrics.items():
    baseline_pq = data['baseline'].get('pq_metrics', {})
    baseline_pq = baseline_pq['groupwise']
    global_pq = data['global'].get('pq_metrics', {})
    global_pq = global_pq['groupwise']
    
    for class_id, value in baseline_pq.items():
        if class_id != "overall":
            pq_data['baseline'][class_id].append(value)
        else:
            print(class_id)

    for class_id, value in global_pq.items():
        pq_data['global'][class_id].append(value)

aggregated_pq_stats = {
    'baseline': {},
    'global': {}
}

for class_id in pq_data['baseline']:
    baseline_values = np.array(pq_data['baseline'][class_id])
    global_values = np.array(pq_data['global'][class_id])
    
    aggregated_pq_stats['baseline'][class_id] = {
        'mean': np.mean(baseline_values),
        'std': np.std(baseline_values)
    }
    
    aggregated_pq_stats['global'][class_id] = {
        'mean': np.mean(global_values),
        'std': np.std(global_values)
    }

sorted_class_ids = sorted([class_id for class_id in ScanNet200_class_id_to_name.keys() if class_id != -1])
x = np.arange(len(sorted_class_ids))  # the label locations
width = 0.35  # the width of the bars

num_classes_per_subplot = len(sorted_class_ids) // 4
remainder = len(sorted_class_ids) % 4

print(f"num class: {len(sorted_class_ids)}, num class per plot: {num_classes_per_subplot}, num remainder: {remainder}")

fig, axs = plt.subplots(4, 1, figsize=(12, 18))

for i, ax in enumerate(axs):
    # Determine the classes to plot in this subplot
    start_idx = i * num_classes_per_subplot
    end_idx = start_idx + num_classes_per_subplot + (1 if i < remainder else 0)
    class_ids_subset = sorted_class_ids[start_idx:end_idx]
    x_subset = np.arange(len(class_ids_subset))
    
    # Plot baseline pq_metrics
    baseline_means = [
        aggregated_pq_stats['baseline'].get(class_id, {'mean': 0})['mean']
        for class_id in class_ids_subset
    ]
    ax.bar(x_subset - width/2, baseline_means, width, label='SAM3D', edgecolor='black', linestyle='--')

    # Plot global pq_metrics
    global_means = [
        aggregated_pq_stats['global'].get(class_id, {'mean': 0})['mean']
        for class_id in class_ids_subset
    ]
    ax.bar(x_subset + width/2, global_means, width, label='GIA3D')

    class_names_subset = [ScanNet200_class_id_to_name[class_id] for class_id in class_ids_subset]

    # Add some text for labels, title, and custom x-axis tick labels, etc.
    #ax.set_xlabel('Class IDs', fontsize=16)
    #ax.set_xlabel('Classes', fontsize=16)
    ax.set_ylabel('Panoptic Quality', fontsize=16)
    #ax.set_title(f'Comparison of Classwise PQ Metrics (Baseline vs Global) - Subplot {i+1}')
    ax.set_xticks(x_subset)
    #ax.set_xticklabels(class_ids_subset, rotation=45, ha="right")
    ax.set_xticklabels(class_names_subset, rotation=45, ha="right", fontsize=12)
    ax.set_ylim(0.0, 1.02)
    if i == 0:
        ax.legend()

# Adjust layout
plt.tight_layout()
plt.show()


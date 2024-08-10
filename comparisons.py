import numpy as np
import glob
import torch
import os
import json
from util import *
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

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
comparisons_dir = os.path.join(base_dir, 'output_global_merger_ablation/scene0703_01/')
is_sorted = True

is3D = False

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

nyu40_color_to_class_id = {v["id"]: k for k, v in nyu40_colors_to_class.items()}
ScanNet20_color_to_class_id = {v["index"]: k for k, v in ScanNet20_colors_to_class.items()}
ScanNet200_color_to_class_id = {v["index"]: k for k, v in ScanNet200_colors_to_class.items()}

nyu40_color_to_class_id_list = list(nyu40_color_to_class_id)
ScanNet20_color_to_class_id_list = list(ScanNet20_color_to_class_id)
ScanNet200_color_to_class_id_list = list(ScanNet200_color_to_class_id)

# Processed data paths
processed_files = [f for f in os.listdir(comparisons_dir) if f.endswith('.pth') and not (f.endswith('baseline.pth') or f.endswith('global.pth'))]
print(f"processed_files: {processed_files}")

scene_files = {}
for filename in processed_files:
    # Extract the scene name part (e.g., 'scene0013_00')
    scene_name = '_'.join(filename.split('_')[:2])
    if scene_name not in scene_files:
        scene_files[scene_name] = []
    scene_files[scene_name].append(filename)

print(f"scene_files: {scene_files}")

combined_results_dict = {}

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

    # Store the combined dictionary for the current scene in the overall dictionary
    combined_results_dict[scene_name] = temp_combined_dict

gt = "ScanNet200"
gt = "Instance"

for scene_name, __ in combined_results_dict.items():
    comparisons_output_data = combined_results_dict[scene_name]
    threshold_to_coefficients_accuracy = {}
    for key, value in comparisons_output_data.items():
        # Extract baseline accuracy if available
        if key.startswith(gt + "_baseline"):
            accuracy_metrics = value.get("iou_metrics", {})  # "accuracy_metrics"
            baseline_accuracy = accuracy_metrics.get("overall")
        
        # Extract accuracy for coefficient-based comparisons
        elif key.startswith(gt + "_global_coeff"):
            coefficients = value.get("coefficients")
            threshold = value.get("threshold")
            if coefficients and threshold is not None:
                # Retrieve the overall accuracy metric
                metrics = value.get("metrics", {})
                accuracy_metrics = metrics.get("iou_metrics", {})  # "accuracy_metrics"
                overall_accuracy = accuracy_metrics.get("overall")
                
                if overall_accuracy is not None:
                    # Ensure the threshold is a key in the dictionary
                    if threshold not in threshold_to_coefficients_accuracy:
                        threshold_to_coefficients_accuracy[threshold] = {}
                    
                    # Store the coefficient pair as a key and the accuracy as the value
                    threshold_to_coefficients_accuracy[threshold][tuple(coefficients)] = overall_accuracy
    
    x_vals = []
    y_vals = []
    z_vals = []
    coeff_set = set()

    for threshold, coefficients_to_accuracy in threshold_to_coefficients_accuracy.items():
        # Check if the data is correctly extracted
        print(f"\nExtracted {len(coefficients_to_accuracy)} entries of coefficients and accuracy values for threshold {threshold}")

        # Sort the dictionary by overall accuracy in descending order
        sorted_coeff_to_accuracy = sorted(coefficients_to_accuracy.items(), key=lambda x: x[1], reverse=True)

        # Separate the sorted coefficients and accuracy values
        sorted_coefficients_str = [", ".join([f"{elem:.2f}" for elem in k]) for k, _ in sorted_coeff_to_accuracy]
        sorted_overall_accuracy_list = [v for _, v in sorted_coeff_to_accuracy]

        coefficients_str = [", ".join([f"{elem:.2f}" for elem in k]) for k in coefficients_to_accuracy.keys()]
        overall_accuracy_list = list(coefficients_to_accuracy.values())

        tolerance = 0.95
        max_value = max(sorted_overall_accuracy_list)

        vicinity_coefficients = []
        vicinity_overall_accuracy = []

        for coeff, accuracy in sorted_coeff_to_accuracy:
            if max_value - tolerance <= accuracy <= max_value + tolerance:
                vicinity_coefficients.append(", ".join([f"{elem:.2f}" for elem in coeff]))
                vicinity_overall_accuracy.append(accuracy)
        
        if is3D:
            
            for coeff, accuracy in coefficients_to_accuracy.items():
                if isinstance(coeff, tuple):
                    coeff = tuple(float(c) for c in coeff)
                    #print(f"coeff: {coeff}")
                    coeff_set.add(coeff)
                    x_vals.append(coeff)
                    #print(f"x_vals: {x_vals}")
                    y_vals.extend([threshold] * len(coeff))
                    z_vals.extend([accuracy] * len(coeff))
                else:
                    x_vals.append(float(coeff))
                    y_vals.append(threshold)
                    z_vals.append(accuracy)

        else:
            plt.figure(figsize=(10, 5))
            if is_sorted:
                # Plotting the sorted data
                #plt.plot(sorted_coefficients_str, sorted_overall_accuracy_list, marker='o')
                plt.plot(vicinity_coefficients, vicinity_overall_accuracy, marker='o', label='Accuracy near max value')
            else:
                # Plotting the unsorted data
                plt.plot(coefficients_str, overall_accuracy_list, marker='o')

            # Plot the baseline accuracy as a horizontal line if available
            if baseline_accuracy is not None:
                plt.axhline(y=baseline_accuracy, color='r', linestyle='--', linewidth=2, label='Baseline Accuracy')

            plt.xlabel('Coefficients (lambda values)')
            plt.ylabel('Overall Accuracy')
            plt.title(f'Overall Accuracy vs Coefficients (ScanNet200_global)\nThreshold: {threshold}, Scene: {scene_name}')
            plt.xticks(rotation=45, ha="right")
            plt.ylim(-0.1, 1.1)
            plt.grid(True)
            plt.tight_layout()
            plt.show()
    
    if is3D:
        #x_vals = [float(val) for val in x_vals]
        #y_vals = [float(val) for val in y_vals]
        #z_vals = [float(val) for val in z_vals]
        ax.set_xlabel('Coefficients (lambda values)')
        ax.set_ylabel('Threshold')
        ax.set_zlabel('Overall Accuracy')

        
        num_points = len(x_vals)
        print(x_vals[0][0])
        indices = np.arange(1, num_points + 1)
        coeff_strs = [str(coeff) for coeff in coefficients]

        print(len(indices))
        print(num_points)
        print(len(y_vals))
        print(len(z_vals))

        sc = ax.scatter(indices, y_vals, z_vals, c=z_vals, cmap='viridis', marker='o')
        ax.set_xticklabels(coeff_strs, rotation=45)

        # Add colorbar to indicate accuracy values
        cbar = plt.colorbar(sc)
        cbar.set_label('Overall Accuracy')

        # Set labels for each axis
        ax.set_xlabel('Coefficients (lambda values)')
        ax.set_ylabel('Threshold')
        ax.set_zlabel('Overall Accuracy')

        # Set plot title
        ax.set_title(f'3D Plot of Overall Accuracy vs Coefficients and Thresholds (ScanNet200_global)\nScene: {scene_name}')

        # Show plot
        plt.show()






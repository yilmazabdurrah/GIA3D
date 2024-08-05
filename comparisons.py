import numpy as np
import glob
import torch
import os
import json
from util import *
import matplotlib.pyplot as plt

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
#comparisons_dir = os.path.join(base_dir, 'output_global_merger_ablation/th_0_5/')
comparisons_dir = os.path.join(base_dir, 'output_global_merger_ablation/th_0_2/')
is_sorted = True

nyu40_color_to_class_id = {v["id"]: k for k, v in nyu40_colors_to_class.items()}
ScanNet20_color_to_class_id = {v["index"]: k for k, v in ScanNet20_colors_to_class.items()}
ScanNet200_color_to_class_id = {v["index"]: k for k, v in ScanNet200_colors_to_class.items()}

nyu40_color_to_class_id_list = list(nyu40_color_to_class_id)
ScanNet20_color_to_class_id_list = list(ScanNet20_color_to_class_id)
ScanNet200_color_to_class_id_list = list(ScanNet200_color_to_class_id)

# Processed data paths
processed_files = [f for f in os.listdir(comparisons_dir) if f.endswith('.pth')]
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

for scene_name, __ in combined_results_dict.items():
    comparisons_output_data = combined_results_dict[scene_name]
    coefficients_to_accuracy = {}
    for key, value in comparisons_output_data.items():
        # Extract baseline accuracy if available
        if key.startswith("ScanNet200_baseline"):
            accuracy_metrics = value.get("accuracy_metrics", {})
            baseline_accuracy = accuracy_metrics.get("overall")
        
        # Extract accuracy for coefficient-based comparisons
        elif key.startswith("ScanNet200_global_coeff"):
            coefficients = value.get("coefficients")
            if coefficients:
                # Retrieve the overall accuracy metric
                metrics = value.get("metrics", {})
                accuracy_metrics = metrics.get("accuracy_metrics", {})
                overall_accuracy = accuracy_metrics.get("overall")
                
                if overall_accuracy is not None:
                    # Store the coefficient pair as a key and the accuracy as the value
                    coefficients_to_accuracy[tuple(coefficients)] = overall_accuracy

    # Check if the data is correctly extracted
    print(f"Extracted {len(coefficients_to_accuracy)} entries of coefficients and accuracy values")

    # Sort the dictionary by overall accuracy in descending order
    sorted_coeff_to_accuracy = sorted(coefficients_to_accuracy.items(), key=lambda x: x[1], reverse=True)

    # Separate the sorted coefficients and accuracy values
    sorted_coefficients_str = [str(k) for k, _ in sorted_coeff_to_accuracy]
    sorted_overall_accuracy_list = [v for _, v in sorted_coeff_to_accuracy]

    coefficients_str = [str(k) for k in coefficients_to_accuracy.keys()]
    overall_accuracy_list = list(coefficients_to_accuracy.values())

    tolerance = 0.01
    max_value = max(sorted_overall_accuracy_list)

    vicinity_coefficients = []
    vicinity_overall_accuracy = []

    for coeff, accuracy in sorted_coeff_to_accuracy:
        if max_value - tolerance <= accuracy <= max_value + tolerance:
            vicinity_coefficients.append(str(coeff))
            vicinity_overall_accuracy.append(accuracy)

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
    plt.title(f'Overall Accuracy vs Coefficients (ScanNet200_global) for {scene_name}')
    plt.xticks(rotation=45, ha="right")
    plt.ylim(-0.1, 1.1)
    plt.grid(True)
    plt.tight_layout()
    plt.show()






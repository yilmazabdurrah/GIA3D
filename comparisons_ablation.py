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
comparisons_dir = os.path.join(base_dir, 'output_global_merger_ablation/') # scene0032_00, scene0118_01, scene0441_00, scene0498_00, scene0690_01, scene0703_01
is_sorted = True

is3D = False

isVerbose = True

if isVerbose:
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

nyu40_color_to_class_id = {v["id"]: k for k, v in nyu40_colors_to_class.items()}
ScanNet20_color_to_class_id = {v["index"]: k for k, v in ScanNet20_colors_to_class.items()}
ScanNet200_color_to_class_id = {v["index"]: k for k, v in ScanNet200_colors_to_class.items()}

nyu40_color_to_class_id_list = list(nyu40_color_to_class_id)
ScanNet20_color_to_class_id_list = list(ScanNet20_color_to_class_id)
ScanNet200_color_to_class_id_list = list(ScanNet200_color_to_class_id)

scene_files = {}

for scene_folder in os.listdir(comparisons_dir):
    scene_folder_path = os.path.join(comparisons_dir, scene_folder)
    
    if os.path.isdir(scene_folder_path):  # Check if it's a directory
        # List and filter the processed files in the current scene folder
        processed_files = [f for f in os.listdir(scene_folder_path) if f.endswith('.pth') and not (f.endswith('baseline.pth') or f.endswith('global.pth'))]
        print(f"processed_files in {scene_folder}: {processed_files}")
        
        for filename in processed_files:
            # Extract the scene name part (e.g., 'scene0013_00')
            scene_name = '_'.join(filename.split('_')[:2])
            if scene_name not in scene_files:
                scene_files[scene_name] = []
            # Append the filename with its full path to the list
            scene_files[scene_name].append(os.path.join(scene_folder_path, filename))

# Print the dictionary containing scene files
print(f"scene_files: {scene_files}")

#combined_results_dict = {}

threshold_to_optimal_coefficients_scenes = {}

#gt = "ScanNet200"
gt = "Instance"

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

    threshold_to_coefficients_accuracy = {}
    for key, value in temp_combined_dict.items():
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
        
        if is3D and isVerbose:
            
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

        elif isVerbose:
            plt.figure(figsize=(10, 5))
            if is_sorted:
                # Plotting the sorted data
                #plt.plot(sorted_coefficients_str, sorted_overall_accuracy_list, marker='o')
                plt.plot(vicinity_coefficients, vicinity_overall_accuracy, marker='o', label='SAM3D-G')
            else:
                # Plotting the unsorted data
                plt.plot(coefficients_str, overall_accuracy_list, marker='o', label='SAM3D-G')

            # Plot the baseline accuracy as a horizontal line if available
            if baseline_accuracy is not None:
                plt.axhline(y=baseline_accuracy, color='r', linestyle='--', linewidth=2, label='SAM3D (Baseline)')

            plt.xlabel('Coefficients $[\lambda_1,\lambda_2,\lambda_3,\lambda_4]$', fontsize=16)
            plt.ylabel('Mean IoU', fontsize=16)
            plt.title(f'Mean IoU vs Coefficients \nThreshold: {round(threshold,3)}, {scene_name}', fontsize=16)
            plt.xticks(rotation=45, ha="right")
            plt.ylim(-0.1, 1.1)
            plt.legend(fontsize=14)
            plt.grid(True)
            plt.tight_layout()
            plt.show()
    
    if is3D and isVerbose:
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

    filtered_threshold_to_coefficients_accuracy = {}

    for threshold, coeff_dict in threshold_to_coefficients_accuracy.items():
            max_accuracy = max(coeff_dict.values())  # Get the max accuracy for the current threshold
            
            for coeff, accuracy in coeff_dict.items():
                if accuracy > baseline_accuracy and accuracy >= 0.9 * max_accuracy:
                    if threshold not in filtered_threshold_to_coefficients_accuracy:
                        filtered_threshold_to_coefficients_accuracy[threshold] = {}
                    
                    filtered_threshold_to_coefficients_accuracy[threshold][coeff] = accuracy

    threshold_to_mean_coefficients = {}

    for threshold, coeff_dict in filtered_threshold_to_coefficients_accuracy.items():
        if not coeff_dict:
            continue
        
        num_coefficients = len(next(iter(coeff_dict)))  # Number of coefficients in a tuple
        coefficients_sums = [0] * num_coefficients
        coefficients_count = len(coeff_dict)  # Number of coefficient pairs
        
        for coeff in coeff_dict.keys():
            for i in range(num_coefficients):
                coefficients_sums[i] += coeff[i]
        
        # Calculate the mean for each coefficient
        mean_coefficients = [coeff_sum / coefficients_count for coeff_sum in coefficients_sums]
        
        # Store the mean coefficients for the current threshold
        threshold_to_mean_coefficients[threshold] = mean_coefficients
    
    threshold_to_optimal_coefficients_scenes[scene_name] = threshold_to_mean_coefficients

    for th, coef in threshold_to_mean_coefficients.items():
        rounded_coef = [round(c, 3) for c in coef]
        print(f"threshold: {round(th,3)} and optimal coefficients: {rounded_coef}")

    if isVerbose:
        for threshold, coefficients_to_accuracy in filtered_threshold_to_coefficients_accuracy.items():
            coefficients_str = [", ".join([f"{elem:.2f}" for elem in k]) for k in coefficients_to_accuracy.keys()]
            overall_accuracy_list = list(coefficients_to_accuracy.values())

            plt.figure(figsize=(10, 5))
            plt.plot(coefficients_str, overall_accuracy_list, marker='o')
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

threshold_to_aggregated_coefficients = {}

for scene, threshold_to_optimal_coefficients in threshold_to_optimal_coefficients_scenes.items():
    for th, coef in threshold_to_optimal_coefficients.items():
        if th not in threshold_to_aggregated_coefficients:
            # Initialize sum and count for each threshold
            threshold_to_aggregated_coefficients[th] = {
                "sum_coefficients": [0] * len(coef),
                "count": 0
            }
        
        for i in range(len(coef)):
            threshold_to_aggregated_coefficients[th]["sum_coefficients"][i] += coef[i]
        
        threshold_to_aggregated_coefficients[th]["count"] += 1

threshold_to_mean_coefficients = {}

for th, data in threshold_to_aggregated_coefficients.items():
    sum_coefficients = data["sum_coefficients"]
    count = data["count"]
    
    mean_coefficients = [sum_coef / count for sum_coef in sum_coefficients]
    
    threshold_to_mean_coefficients[th] = mean_coefficients

for th, coef in threshold_to_mean_coefficients.items():
    rounded_coef = [round(c, 3) for c in coef]
    print(f"threshold: {round(th,3)} and averaged coefficients: {rounded_coef}")

'''# Prepare the plot with improved visualization
plt.figure(figsize=(10, 6))
plt.grid(True, which='both', linestyle='--', linewidth=0.5)

# Updated styles for better differentiation
colors = ['b', 'g', 'r', 'purple']
styles = ['-', '--', '-.', ':']
markers = ['o', 's', '^', 'D']
labels = [r'$\lambda_1$', r'$\lambda_2$', r'$\lambda_3$', r'$\lambda_4$']  # Greek letters for Lambda

# Fit a line and plot each coefficient with respect to thresholds
for i in range(coefficients.shape[1]):
    x = thresholds.reshape(-1, 1)
    y = coefficients[:, i]
    
    # Linear regression fit
    model = LinearRegression()
    model.fit(x, y)
    y_pred = model.predict(x)
    
    # Plot the original data with unique markers
    plt.plot(thresholds, y, marker=markers[i], color=colors[i], linestyle='None', 
             label=f'{labels[i]} (data)', markersize=8)
    
    # Plot the fitted line with different styles
    plt.plot(thresholds, y_pred, linestyle=styles[i], linewidth=3, color=colors[i], label=f'{labels[i]} (fit)')

# Labeling the plot with larger fonts
plt.xlabel('Threshold', fontsize=16)
plt.ylabel('Coefficient Values', fontsize=16)
plt.title('Coefficient Values vs. Threshold with Linear Fit', fontsize=18)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(fontsize=14)
plt.tight_layout()

# Save the plot with high quality for academic papers
plt.savefig('/mnt/data/coefficient_vs_threshold_final.png', dpi=300)

# Show the plot
plt.show()'''






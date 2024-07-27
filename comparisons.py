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
comparisons_dir = os.path.join(base_dir, 'output_global_merger_comparisons/')

# Processed data paths
processed_files = [f for f in os.listdir(comparisons_dir) if f.endswith('_comparisons_output.pth')]
input_files = [f for f in os.listdir(comparisons_dir) if f.endswith('_comparisons_input.pth')]

nyu40_color_to_class_id = {v["id"]: k for k, v in nyu40_colors_to_class.items()}
ScanNet20_color_to_class_id = {v["index"]: k for k, v in ScanNet20_colors_to_class.items()}
ScanNet200_color_to_class_id = {v["index"]: k for k, v in ScanNet200_colors_to_class.items()}

nyu40_color_to_class_id_list = list(nyu40_color_to_class_id)
ScanNet20_color_to_class_id_list = list(ScanNet20_color_to_class_id)
ScanNet200_color_to_class_id_list = list(ScanNet200_color_to_class_id)



for filename in processed_files:
    print(filename)
    comparisons_filepath = os.path.join(comparisons_dir, filename)
    comp_file = glob.glob(comparisons_filepath)
    # Check if any comparison files are found
    if not comp_file:
        print(f"No comparison file found for {filename}")
        continue
    else:
        comparisons_output_data = torch.load(comparisons_filepath)
        print(f"Comparison file {filename} loaded")
    
    for key, value in comparisons_output_data.items():
        print(f"Main Key {key}")
        if isinstance(value, dict):
            print_nested_dict(value)
        else:
            print(f"Value {value}")

'''
# Reach/Print out coord (XYZ), colors (RGB) and labels (classes) for GT and methods 
for filename in input_files:
    print(filename)
    comparisons_filepath = os.path.join(comparisons_dir, filename)
    comp_file = glob.glob(comparisons_filepath)
    # Check if any comparison files are found
    if not comp_file:
        print(f"No comparison file found for {filename}")
        continue
    else:
        comparisons_output_data = torch.load(comparisons_filepath)
        print(f"Comparison file {filename} loaded")
    
    for key, value in comparisons_output_data.items():
        print(f"Main Key {key}")'''




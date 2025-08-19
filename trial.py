import pandas as pd
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("/home/ayilmaz/Downloads/137-2170-142955760-pointcloud.csv", sep=",", header=None, names=["x", "y", "z", "label"])

# Convert to numpy array
points = df[["x", "y", "z"]].to_numpy()

# Create Open3D PointCloud object
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)

# Optional: Add colors based on label
labels = df["label"].to_numpy()
colors = plt.cm.jet(labels / labels.max())[:, :3]  # Normalize and apply colormap
pcd.colors = o3d.utility.Vector3dVector(colors)

# Visualize
o3d.visualization.draw_geometries([pcd])

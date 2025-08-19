# Run Following code

cd ws_segment_3d/GIA3D/

conda activate sam3d

python sam3d_AY.py --rgb_path /home/ayilmaz/ws_segment_3d/GIA3D/scannet-preprocess/processed_dataset/tasks/rgb_test_data/ \
--data_path /home/ayilmaz/ws_segment_3d/GIA3D/scannet-preprocess/processed_dataset/  \
--save_path /home/ayilmaz/ws_segment_3d/GIA3D/output/  \
--save_2dmask_path /home/ayilmaz/ws_segment_3d/GIA3D/output/  \
--sam_checkpoint_path /home/ayilmaz/ws_segmentation/src/Model_Checkpoints/sam_vit_h_4b8939.pth

python sam3d_AY.py --rgb_path /home/ayilmaz/ws_segment_3d/GIA3D/scannet-preprocess/processed_dataset/tasks/rgb_test_data/  \
--data_path /home/ayilmaz/ws_segment_3d/GIA3D/scannet-preprocess/processed_dataset/  \
--save_path /home/ayilmaz/ws_segment_3d/GIA3D/output_global_merger/  \
--save_2dmask_path /home/ayilmaz/ws_segment_3d/GIA3D/output_global_merger/  \
--sam_checkpoint_path /home/ayilmaz/ws_segmentation/src/Model_Checkpoints/sam_vit_h_4b8939.pth

# GT comparisons in test data

python sam3d_AY.py --rgb_path /home/ayilmaz/ws_segment_3d/GIA3D/scannet-preprocess/processed_dataset/tasks/rgb_test_data/  \
--data_path /home/ayilmaz/ws_segment_3d/GIA3D/scannet-preprocess/processed_dataset/  \
--save_path /home/ayilmaz/ws_segment_3d/GIA3D/output_global_merger/  \
--save_2dmask_path /home/ayilmaz/ws_segment_3d/GIA3D/output_global_merger/  \
--sam_checkpoint_path /home/ayilmaz/ws_segmentation/src/Model_Checkpoints/sam_vit_h_4b8939.pth  \
--gt_data_path /home/ayilmaz/ws_segment_3d/GIA3D/ScanNetv2_Downloads/gt/scans/

# GT comparisons in training and validation data

python sam3d_AY.py --rgb_path /home/ayilmaz/ws_segment_3d/GIA3D/scannet-preprocess/processed_dataset/rgbd/  \
--data_path /home/ayilmaz/ws_segment_3d/GIA3D/scannet-preprocess/processed_dataset/point_cloud/  \
--save_path /home/ayilmaz/ws_segment_3d/GIA3D/output_global_merger_comparisons/  \
--save_2dmask_path /home/ayilmaz/ws_segment_3d/GIA3D/output_global_merger_comparisons/  \
--sam_checkpoint_path /home/ayilmaz/ws_segmentation/src/Model_Checkpoints/sam_vit_h_4b8939.pth  \
--gt_data_path /home/ayilmaz/ws_segment_3d/GIA3D/ScanNetv2_Downloads/gt/scans/

python sam3d_AY.py --rgb_path /home/ayilmaz/ws_segment_3d/GIA3D/ScanNetv2_Downloads/raw_dataset/preprocessed/rgbd_frame_skip_20/  \
--data_path /home/ayilmaz/ws_segment_3d/GIA3D/ScanNetv2_Downloads/raw_dataset/preprocessed/point_cloud/  \
--save_path /home/ayilmaz/ws_segment_3d/GIA3D/output_global_merger_comparisons/  \
--save_2dmask_path /media/ayilmaz/Crucial\ X9/ScanNetv2_Dataset/output_step_1_2/  \
--sam_checkpoint_path /home/ayilmaz/ws_segmentation/src/Model_Checkpoints/sam_vit_h_4b8939.pth  \
--gt_data_path /media/ayilmaz/Crucial\ X9/ScanNetv2_Dataset/gt/processed/

# Ablation Study

python sam3d_AY_ablation.py --rgb_path /home/ayilmaz/ws_segment_3d/GIA3D/scannet-preprocess/processed_dataset/rgbd_/  \
--data_path /home/ayilmaz/ws_segment_3d/GIA3D/scannet-preprocess/processed_dataset/point_cloud/  \
--save_path /home/ayilmaz/ws_segment_3d/GIA3D/output_global_merger_comparisons/  \
--save_2dmask_path /home/ayilmaz/ws_segment_3d/GIA3D/output_global_merger_comparisons/  \
--sam_checkpoint_path /home/ayilmaz/ws_segmentation/src/Model_Checkpoints/sam_vit_h_4b8939.pth  \
--gt_data_path /home/ayilmaz/ws_segment_3d/GIA3D/ScanNetv2_Downloads/gt/scans/

python sam3d_AY_ablation.py --rgb_path /home/ayilmaz/ws_segment_3d/GIA3D/ScanNetv2_Downloads/raw_dataset/preprocessed/rgbd_frame_skip_20/  \
--data_path /home/ayilmaz/ws_segment_3d/GIA3D/ScanNetv2_Downloads/raw_dataset/preprocessed/point_cloud/  \
--save_path /home/ayilmaz/ws_segment_3d/GIA3D/output_global_merger_ablation/  \
--save_2dmask_path /media/ayilmaz/Crucial\ X9/ScanNetv2_Dataset/output_step_1_2/  \
--sam_checkpoint_path /home/ayilmaz/ws_segmentation/src/Model_Checkpoints/sam_vit_h_4b8939.pth  \
--gt_data_path /media/ayilmaz/Crucial\ X9/ScanNetv2_Dataset/gt/processed/


# Single Run for Step 1 and Step 2 Classic and Save Output

python sam3d_AY_stp1_2.py --rgb_path /media/ayilmaz/Crucial\ X9/ScanNetv2_Dataset/raw_dataset/preprocessed/rgbd/  \
--data_path /media/ayilmaz/Crucial\ X9/ScanNetv2_Dataset/raw_dataset/preprocessed/point_cloud/  \
--save_path /media/ayilmaz/Crucial\ X9/ScanNetv2_Dataset/output_step_1_2/  \
--save_2dmask_path /media/ayilmaz/Crucial\ X9/ScanNetv2_Dataset/output_step_1_2/  \
--sam_checkpoint_path /home/ayilmaz/ws_segmentation/src/Model_Checkpoints/sam_vit_h_4b8939.pth  \
--gt_data_path /media/ayilmaz/Crucial\ X9/ScanNetv2_Dataset/gt/scans/


/home/ayilmaz/ws_segment_3d/GIA3D/output.txt # for details on correspondences

/home/ayilmaz/ws_segment_3d/GIA3D/ScanNetv2_Downloads/gt/scans/ # GT point clouds segmented ply

# Save segmented point clouds by our algorithm on GIA3D in ply format

python visualize_save.py

# Data prepare

## Download GT Data

python3 download-scannet.py -o /home/ayilmaz/ws_segment_3d/SegmentAnything3/ScanNetv2_Downloads/gt --type _vh_clean_2.labels.ply

## Download Point Cloud

python scannet-preprocess/preprocess_scannet.py --dataset_root /home/ayilmaz/ws_segment_3d/GIA3D/scannet-preprocess/raw_dataset/raw --output_root /home/ayilmaz/ws_segment_3d/GIA3D/scannet-preprocess/processed_dataset/point_cloud/

### To Fix Issue

python scannet-preprocess/preprocess_scannet.py  \
--dataset_root /home/ayilmaz/ws_segment_3d/GIA3D/scannet-preprocess/raw_dataset/raw_  \
--output_root /home/ayilmaz/ws_segment_3d/GIA3D/scannet-preprocess/raw_dataset/raw_/point_cloud

## Download RGBD

python3 download-scannet.py -o /home/ayilmaz/ws_segment_3d/GIA3D/scannet-preprocess/raw_dataset/raw/ --type .sens

python3 download-scannet.py -o /media/ayilmaz/Crucial\ X9/ScanNetv2_Dataset/raw_dataset/raw/ --type .sens

python scannet-preprocess/prepare_2d_data/prepare_2d_data.py  \
--scannet_path /home/ayilmaz/ws_segment_3d/GIA3D/scannet-preprocess/raw_dataset/raw/scans  \
--output_path /home/ayilmaz/ws_segment_3d/GIA3D/scannet-preprocess/processed_dataset/rgbd/  \
--export_label_images --label_map_file /home/ayilmaz/ws_segment_3d/GIA3D/scannet-preprocess/raw_dataset/raw/scannetv2-labels.combined.tsv

python scannet-preprocess/prepare_2d_data/prepare_2d_data.py --scannet_path /media/ayilmaz/Crucial\ X9/ScanNetv2_Dataset/raw_dataset/raw/scans  \
--output_path /media/ayilmaz/Crucial\ X9/ScanNetv2_Dataset/raw_dataset/preprocessed/rgbd/  \
--export_label_images --label_map_file /media/ayilmaz/Crucial\ X9/ScanNetv2_Dataset/raw_dataset/raw/scannetv2-labels.combined.tsv

or

python scannet-preprocess/prepare_2d_data/prepare_2d_data.py --scannet_path /home/ayilmaz/ws_segment_3d/GIA3D/scannet-preprocess/raw_dataset/raw/scans --output_path /home/ayilmaz/ws_segment_3d/GIA3D/scannet-preprocess/processed_dataset/rgbd/

python scannet-preprocess/prepare_2d_data/prepare_2d_data.py --scannet_path /media/ayilmaz/Crucial\ X9/ScanNetv2_Dataset/raw_dataset/raw/scans 
--output_path /media/ayilmaz/Crucial\ X9/ScanNetv2_Dataset/raw_dataset/preprocessed/rgbd/ 

# Running on Matterport Dataset

python matterport_GT_extraction.py \
--scans_file_path /home/ayilmaz/ws_segment_3d/GIA3D/datasets/Matterport_dataset/scans.txt \
--scans_root /home/ayilmaz/ws_segment_3d/GIA3D/datasets/Matterport_dataset/dataset/v1/scans \
--save_path /home/ayilmaz/ws_segment_3d/GIA3D/datasets/Matterport_dataset/output/trial01 \
--save_2dmask_path /home/ayilmaz/ws_segment_3d/GIA3D/datasets/Matterport_dataset/output/trial01/output_step_1_2/ \
--sam_checkpoint_path /home/ayilmaz/ws_segmentation/src/Model_Checkpoints/sam_vit_h_4b8939.pth \
--category_mapping_path /home/ayilmaz/ws_segment_3d/GIA3D/datasets/Matterport_dataset/category_mapping.tsv

python gia3d_matterport.py \
--scans_file_path /home/ayilmaz/ws_segment_3d/GIA3D/datasets/Matterport_dataset/scans.txt \
--scans_root /home/ayilmaz/ws_segment_3d/GIA3D/datasets/Matterport_dataset/dataset/v1/scans \
--save_path /home/ayilmaz/ws_segment_3d/GIA3D/datasets/Matterport_dataset/output/trial01 \
--save_2dmask_path /home/ayilmaz/ws_segment_3d/GIA3D/datasets/Matterport_dataset/output/trial01/ \
--sam_checkpoint_path /home/ayilmaz/ws_segmentation/src/Model_Checkpoints/sam_vit_h_4b8939.pth \
--category_mapping_path /home/ayilmaz/ws_segment_3d/GIA3D/datasets/Matterport_dataset/category_mapping.tsv \
--gt_data_path /home/ayilmaz/ws_segment_3d/GIA3D/datasets/Matterport_dataset/dataset/v1/gt/

pip install -e /home/ayilmaz/ws_segmentation/src/segment-anything
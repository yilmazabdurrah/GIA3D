import open3d as o3d
import numpy as np
from sklearn.metrics import average_precision_score
from collections import defaultdict
import glob
import os
from scipy.spatial import cKDTree

def read_ply(file_path):
    pcd = o3d.io.read_point_cloud(file_path)
    points = np.asarray(pcd.points)
    colors = (np.asarray(pcd.colors) * 255).astype(np.uint8)
    return points, colors

def get_instance_map_from_colors(colors):
    color_labels, instance_ids = np.unique(colors, axis=0, return_inverse=True)
    instance_map = defaultdict(list)
    for idx, inst_id in enumerate(instance_ids):
        instance_map[inst_id].append(idx)
    return instance_map

def compute_tp_fp_fn(pred_instance_map, gt_instance_map, iou_threshold):
    matched_gt = set()
    matched_ious = []
    unmatched_ious = []

    for pred_id, pred_indices in pred_instance_map.items():
        pred_indices = set(pred_indices)
        best_iou = 0
        best_gt_id = None
        for gt_id, gt_indices in gt_instance_map.items():
            if gt_id in matched_gt:
                continue
            gt_indices = set(gt_indices)
            intersection = len(pred_indices & gt_indices)
            union = len(pred_indices | gt_indices)
            iou = intersection / union if union > 0 else 0
            if iou > best_iou:
                best_iou = iou
                best_gt_id = gt_id

        if best_iou >= iou_threshold:
            matched_gt.add(best_gt_id)
            matched_ious.append(best_iou)
        else:
            unmatched_ious.append(best_iou)

    tp = len(matched_ious)
    fp = len(unmatched_ious)
    fn = len(gt_instance_map) - len(matched_gt)

    return tp, fp, fn, matched_ious, unmatched_ious

def compute_ap(tp, fp, fn, matched_ious, unmatched_ious):
    y_true = [1] * tp + [0] * fp + [0] * fn
    y_scores = matched_ious + unmatched_ious + [0] * fn
    if len(y_true) != len(y_scores):
        raise ValueError(f"Mismatched y_true ({len(y_true)}) and y_scores ({len(y_scores)})")
    if not y_true or sum(y_true) == 0:
        return 0.0
    return average_precision_score(y_true, y_scores)

def compute_ap_metrics(gt_ply_path, pred_ply_path):
    pred_points, pred_colors = read_ply(pred_ply_path)
    gt_points, gt_colors = read_ply(gt_ply_path)

    if len(pred_points) != len(gt_points):
        raise ValueError("Prediction and GT point clouds do not match in size.")

    gt_instance_map = get_instance_map_from_colors(gt_colors)
    pred_instance_map = get_instance_map_from_colors(pred_colors)

    ap25_tp, ap25_fp, ap25_fn, ap25_matched, ap25_unmatched = compute_tp_fp_fn(pred_instance_map, gt_instance_map, 0.25)
    ap50_tp, ap50_fp, ap50_fn, ap50_matched, ap50_unmatched = compute_tp_fp_fn(pred_instance_map, gt_instance_map, 0.50)

    ap25 = compute_ap(ap25_tp, ap25_fp, ap25_fn, ap25_matched, ap25_unmatched)
    ap50 = compute_ap(ap50_tp, ap50_fp, ap50_fn, ap50_matched, ap50_unmatched)

    thresholds = np.arange(0.5, 1.0, 0.05)
    aps = []
    for t in thresholds:
        tp, fp, fn, matched, unmatched = compute_tp_fp_fn(pred_instance_map, gt_instance_map, t)
        aps.append(compute_ap(tp, fp, fn, matched, unmatched))
    map_score = np.mean(aps) if aps else 0.0

    return {
        'AP25': ap25,
        'AP50': ap50,
        'mAP': map_score,
        'IoU_per_instance': ap50_matched + ap50_unmatched,
        'num_fn': ap50_fn
    }

def evaluate_dataset(gt_dir, pred_dir):
    gt_files = sorted(glob.glob(os.path.join(gt_dir, "*instance_gt.ply")))
    results = {}
    all_ap25, all_ap50, all_map = [], [], []

    for gt_file in gt_files:
        scene_name = os.path.basename(gt_file).split("_gt_instance_gt")[0]
        pred_pattern = os.path.join(pred_dir, f"{scene_name}*global_with_labels.ply")
        print(f"pred_pattern: {pred_pattern}")
        pred_matches = glob.glob(pred_pattern)
        print(f"pred_matches: {pred_matches}")

        if not pred_matches:
            print(f"No prediction found for {scene_name}")
            continue

        pred_file = pred_matches[0]  # assuming one match per scene
        print(f"scene_name: {scene_name}")
        print(f"gt_file: {gt_file}")
        print(f"pred_file: {pred_file}")
        try:
            metrics = compute_ap_metrics(gt_file, pred_file)
            results[scene_name] = metrics
            all_ap25.append(metrics['AP25'])
            all_ap50.append(metrics['AP50'])
            all_map.append(metrics['mAP'])
            print(f"Scene {scene_name}: AP25={metrics['AP25']:.4f}, AP50={metrics['AP50']:.4f}, mAP={metrics['mAP']:.4f}")
            print(f"IoU_per_instance={metrics['IoU_per_instance']}")
        except ValueError as e:
            print(f"Error processing {scene_name}: {e}")

        if all_ap25:
            print("\nOverall Dataset Performance:")
            print(f"Mean AP25: {np.mean(all_ap25):.4f}")
            print(f"Mean AP50: {np.mean(all_ap50):.4f}")
            print(f"Mean mAP:  {np.mean(all_map):.4f}")

    return results

if __name__ == "__main__":
    gt_dir = "/home/ayilmaz/ws_segment_3d/GIA3D/datasets/ScanNetv2_Downloads/gt_PCs"
    pred_dir = "/home/ayilmaz/ws_segment_3d/GIA3D/output_global_merger_comparisons/PCs_"
    evaluate_dataset(gt_dir, pred_dir)

    
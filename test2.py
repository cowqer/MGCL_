import os
import numpy as np
from PIL import Image

def compute_binary_iou(pred_mask, gt_mask):
    pred_mask = np.array(pred_mask).astype(bool)
    gt_mask = np.array(gt_mask).astype(bool)
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    if union == 0:
        return 1.0
    else:
        return intersection / union

def batch_iou(pred_folder, gt_folder):
    iou_results = {}
    pred_files = sorted(os.listdir(pred_folder))
    
    for fname in pred_files:
        pred_path = os.path.join(pred_folder, "mining00_pred.png")
        gt_path = os.path.join(gt_folder, "mining00_gt.png")
        
        if not os.path.exists(gt_path):
            print(f"[WARN] GT not found for {fname}, skipping.")
            continue
        
        # 读取 mask
        pred_mask = Image.open(pred_path).convert("1")  # 二值化
        gt_mask = Image.open(gt_path).convert("1")      # 二值化

        iou = compute_binary_iou(pred_mask, gt_mask)
        iou_results[fname] = iou

    return iou_results

# 使用示例
pred_folder = "/data/seekyou/Algos/MGCL/vis_results"
gt_folder = "/data/seekyou/Algos/MGCL/vis_results"

results = batch_iou(pred_folder, gt_folder)
for fname, iou_val in results.items():
    print(f"{fname}: IoU = {iou_val:.4f}")

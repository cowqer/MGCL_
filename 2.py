import os
import pickle
import numpy as np
from PIL import Image

def compare_label_png_and_pkl(png_path, pkl_path):
    # 读取png标签图
    png_label = np.array(Image.open(png_path))
    print(f"PNG label unique values: {np.unique(png_label)}")

    # 读取pkl文件里的label
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    pkl_label = data['label']
    print(f"PKL label unique values: {np.unique(pkl_label)}")

    # 计算两个label的差异（是否完全相等）
    if png_label.shape != pkl_label.shape:
        print("Shape mismatch between PNG label and PKL label!")
    else:
        equal_pixels = np.sum(png_label == pkl_label)
        total_pixels = png_label.size
        print(f"Pixel-wise equal count: {equal_pixels} / {total_pixels} ({equal_pixels/total_pixels*100:.2f}%)")

# 示例调用，替换为你自己的文件路径
png_label_path = "/data/seekyou/Data/DLRSD_split/train/Labels/agricultural00.png"
pkl_label_path = "/data/seekyou/Data/DLRSD_split/train/sam_mask_vit_h_t64_p16_s50/agricultural00.pkl"

compare_label_png_and_pkl(png_label_path, pkl_label_path)


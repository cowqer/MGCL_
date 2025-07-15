import tifffile
import numpy as np
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import torch

# =========================
# 1. 加载多波段TIF
tif_path = "./A1.tif"
image = tifffile.imread(tif_path)
print("Original shape:", image.shape)

# 可能是 (5, H, W) 也可能是 (H, W, 5)
if image.ndim == 3 and image.shape[0] < 10:
    image = np.transpose(image, (1, 2, 0))
print("Shape for processing:", image.shape)

# =========================
# 2. 选3个波段组合
# 这里最简单方案是取前3波段
pseudo_rgb = image[..., :3].astype(np.float32)

# 更专业的false color（自己改映射）
# pseudo_rgb = np.stack([image[..., 4], image[..., 3], image[..., 2]], axis=-1).astype(np.float32)

# =========================
# 3. 归一化拉伸到0-255
def scale_to_uint8(arr):
    arr -= arr.min()
    arr /= (arr.max() + 1e-8)
    return (arr * 255).clip(0, 255).astype(np.uint8)

pseudo_rgb = scale_to_uint8(pseudo_rgb)
print("Prepared pseudo_rgb shape:", pseudo_rgb.shape)

# =========================
# 4. 初始化SAM
model_type = "vit_h"
checkpoint_path = "./segment-anything-main/checkpoints/sam_vit_h_4b8939.pth"
device = "cuda" if torch.cuda.is_available() else "cpu"

sam = sam_model_registry[model_type](checkpoint=checkpoint_path).to(device)
mask_generator = SamAutomaticMaskGenerator(
    sam,
    points_per_side=16,
    stability_score_thresh=0.5
)

# =========================
# 5. 生成mask
masks = mask_generator.generate(pseudo_rgb)
print(f"Generated {len(masks)} masks")

# =========================
# 6. 可视化或保存结果
# 这里最简单可视化
plt.figure(figsize=(10, 10))
plt.imshow(pseudo_rgb)
for ann in masks:
    plt.imshow(ann["segmentation"], alpha=0.3)
plt.axis('off')
plt.savefig("./vis/sam_mask_overlay.png")
plt.close()

# 也可以逐个二值mask保存
import os
os.makedirs("./vis", exist_ok=True)
for idx, ann in enumerate(masks):
    seg = (ann["segmentation"] * 255).astype(np.uint8)
    tifffile.imwrite(f"./vis/mask_{idx+1}.tif", seg)

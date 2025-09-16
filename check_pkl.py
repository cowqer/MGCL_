# import pickle
# import numpy as np

# # 打开 pkl 文件
# import pickle
# import numpy as np

# import matplotlib.pyplot as plt
# pkl_path = "/data/seekyou/Data/DLRSD_split/val/sam_mask_vit_h_t64_p16_s50/airplane93.pkl"  # 替换为你的文件路径
# # /data/seekyou/Data/DLRSD_split/train/sam_mask_vit_h_t64_p32_s50
# with open(pkl_path, "rb") as f:
#     data = pickle.load(f)

# print("Type of data:", type(data))
# print("Keys in data:", list(data.keys()))
# print("\n--- Detailed content ---\n")

# for k, v in data.items():
#     print(f"Key: {k}")
#     print(f"  Type: {type(v)}")
    
#     if isinstance(v, np.ndarray):
#         print(f"  Shape: {v.shape}")
#         print(f"  Dtype: {v.dtype}")
#         print(f"  Min: {v.min()}")
#         print(f"  Max: {v.max()}")
#         print(f"  Value: {np.unique(v)}")
#         print(f"  First few elements:\n{v.flatten()[:10]}")
#     elif isinstance(v, (tuple, list)):
#         print(f"  Length: {len(v)}")
#         print(f"  Content: {v}")
#     else:
#         print(f"  Value: {v}")
    


pkl_path = "/data/seekyou/Data/DLRSD_split/val/sam_mask_vit_h_t64_p16_s50/airplane93.pkl"

import pickle
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. 读取 pkl 文件
# pkl_path = "/data/seekyou/Algos/MGCL/your_file.pkl"
with open(pkl_path, "rb") as f:
    data = pickle.load(f)

print("Keys in pkl:", data.keys())

# 2. 获取 masks
masks = data["masks"]  # 具体 shape 你可以 print 看一下
print("masks shape:", np.array(masks).shape, type(masks))
masks = data["masks"]

# 如果是 list → 转 numpy
if isinstance(masks, list):
    masks = np.array(masks)

# 如果还是 object 类型（说明子元素 shape 不统一）
if masks.dtype == object:
    print("masks 是 object 类型，可能每个 mask 尺寸不一样")
    # 取第一个看看
    mask0 = np.array(masks[0], dtype=np.float32)
else:
    mask0 = masks[0].astype(np.float32)
# 3. 保存到可视化文件夹
save_dir = "./vis_masks"
os.makedirs(save_dir, exist_ok=True)

# 假设 masks 是 [N, H, W] 或 [H, W]
if isinstance(masks, list):
    masks = np.array(masks)

if masks.ndim == 2:
    # 单个 mask
    mask = (masks * 255).astype(np.uint8)
    cv2.imwrite(os.path.join(save_dir, "mask.png"), mask)
elif masks.ndim == 3:
    # 多个 mask
    for i, mask in enumerate(masks):
        mask = (mask * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(save_dir, f"mask_{i}.png"), mask)
else:
    print("Unsupported masks shape:", masks.shape)

# 4. 也可以用 matplotlib 直接可视化
plt.imshow(masks[0], cmap="gray")
plt.title("Mask[0]")
plt.savefig(os.path.join(save_dir, "mask0_matplotlib.png"))
plt.close()
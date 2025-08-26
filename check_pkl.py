import pickle
import numpy as np

# 打开 pkl 文件
import pickle
import numpy as np

import matplotlib.pyplot as plt
pkl_path = "/data/seekyou/Data/DLRSD_split/val/sam_mask_vit_h_t64_p16_s50/airplane93.pkl"  # 替换为你的文件路径
# /data/seekyou/Data/DLRSD_split/train/sam_mask_vit_h_t64_p32_s50
with open(pkl_path, "rb") as f:
    data = pickle.load(f)

print("Type of data:", type(data))
print("Keys in data:", list(data.keys()))
print("\n--- Detailed content ---\n")

for k, v in data.items():
    print(f"Key: {k}")
    print(f"  Type: {type(v)}")
    
    if isinstance(v, np.ndarray):
        print(f"  Shape: {v.shape}")
        print(f"  Dtype: {v.dtype}")
        print(f"  Min: {v.min()}")
        print(f"  Max: {v.max()}")
        print(f"  Value: {np.unique(v)}")
        print(f"  First few elements:\n{v.flatten()[:10]}")
    elif isinstance(v, (tuple, list)):
        print(f"  Length: {len(v)}")
        print(f"  Content: {v}")
    else:
        print(f"  Value: {v}")
    
    print("-"*50)

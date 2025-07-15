import os
import numpy as np
import tifffile
import matplotlib.pyplot as plt

# ========================
# 设置路径
image_path = "./A1.tif"    # ⚠️ 改成你的图像路径
output_dir = "./vis"
os.makedirs(output_dir, exist_ok=True)

# ========================
# 加载多波段图像
image = tifffile.imread(image_path)
print(f"Loaded image shape: {image.shape}")

# 如果是 (C, H, W)，转为 (H, W, C)
if image.ndim == 3 and image.shape[0] < 10:
    image = np.transpose(image, (1, 2, 0))
    print(f"Transposed to: {image.shape}")

bands = image.shape[-1]

# ========================
# 单独保存每个波段
for i in range(bands):
    band = image[..., i]
    plt.imshow(band, cmap='gray')
    plt.title(f'Band {i+1}')
    plt.colorbar()
    plt.savefig(os.path.join(output_dir, f'band_{i+1}.png'))
    plt.close()
    print(f"Saved band {i+1}")

# ========================
# 做个伪RGB合成（取前三波段）
if bands >= 3:
    pseudo_rgb = image[..., :3].astype(np.float32)
    pseudo_rgb -= pseudo_rgb.min()
    pseudo_rgb /= (pseudo_rgb.max() + 1e-8)
    plt.imshow(pseudo_rgb)
    plt.title('Pseudo RGB (Bands 1-3)')
    plt.axis('off')
    plt.savefig(os.path.join(output_dir, 'pseudo_rgb.png'))
    plt.close()
    print("Saved pseudo_rgb.png")
else:
    print("Not enough bands for RGB visualization.")

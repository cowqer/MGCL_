import os
import shutil

log_dir = "./logs"
for name in os.listdir(log_dir):
    path = os.path.join(log_dir, name)
    if not os.path.isdir(path):
        continue

    # 拆分前缀和编号
    if "_" in name:
        parts = name.split("_")
        prefix = "_".join(parts[:-1])  # 前缀
        suffix = parts[-1]             # 编号

        if suffix.isdigit():  # 确认最后一段是数字
            new_dir = os.path.join(log_dir, prefix)
            os.makedirs(new_dir, exist_ok=True)
            shutil.move(path, os.path.join(new_dir, suffix))

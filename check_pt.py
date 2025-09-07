import torch
from net import myNetwork   # TODO: 换成你的网络类，比如 SegmentationHead_FBC_3_v1

pt_path = "/data/seekyou/Algos/MGCL/logs/MGCD+FGE+amp_43.11/FBC_3_2_0828__03_5555.log/best_model.pt"   # 你的 .pt 文件路径

# 1. 加载 state_dict
state_dict = torch.load(pt_path, map_location="cpu")

print("="*50)
print("📌 State dict keys & tensor shapes in checkpoint:")
for k, v in state_dict.items():
    print(f"{k:50s} {tuple(v.shape)}")
print("="*50)

# 2. 构建模型实例
args = None  # 如果你的网络构造需要参数，可以在这里传
model = myNetwork(args)

# 3. 加载权重
model.load_state_dict(state_dict, strict=False)  # strict=False 防止缺层时报错

print("\n📌 Model Structure:")
print(model)

# 4. （可选）更详细结构：每层输入输出 shape
try:
    from torchsummary import summary
    summary(model, input_size=(3, 224, 224))  # TODO: 输入大小改成你的数据尺寸
except Exception as e:
    print("\n⚠️ torchsummary 未安装或输入尺寸不对，跳过详细 summary")

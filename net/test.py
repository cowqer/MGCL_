import torch
from modules import HyperGraphBuilder  # 假设你的类保存在 hypergraph.py 中


if __name__ == "__main__":
    # 模拟输入特征图：3个层级，每个 batch=2，通道数不同
    B = 2
    feat1 = torch.randn(B, 16, 256, 256)  # C1 = 16
    feat2 = torch.randn(B, 32, 16, 16)  # C2 = 32
    feat3 = torch.randn(B, 64, 8, 8)    # C3 = 64

    feature_list = [feat1, feat2, feat3]

    # 初始化超图构建器
    builder = HyperGraphBuilder(epsilon=1.0, device='cpu')

    # 融合特征图
    fused = builder.fuse_multiscale_features(feature_list)  # → [B, C_total, H, W]

    print(f"Fused feature shape: {fused.shape}")  # (B, C1+C2+C3, 32, 32)

    # 构建超图关联矩阵
    hypergraphs = builder.build_hypergraph(fused)  # List of B tensors, each (N, E)

    for i, H in enumerate(hypergraphs):
        print(f"Sample {i}: H shape = {H.shape}")  # H ∈ [N, E]
        sparsity = H.sum().item() / H.numel()
        print(f"  Hyperedge density (sparsity): {sparsity:.4f}")
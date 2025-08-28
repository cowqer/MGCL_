import torch
from blocks.mamba_blocks import VSSM

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # 随机输入
    B, C, H, W = 2, 256, 64, 64
    q_feat = torch.randn(B, C, H, W, device=device)
    s_feat = torch.randn(B, C, H, W, device=device)

    # 构建模型
    model = VSSM(
        depths=[2],
        dims=[256],
        mlp_ratio=1,
        d_state=16,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.,
    ).to(device)   # <<< 关键：把模型放到 GPU

    # 前向传播
    out_q, out_s = model(q_feat, s_feat)

    print("q_feat out:", out_q.shape)
    print("s_feat out:", out_s.shape)

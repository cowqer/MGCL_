import torch
import torch.nn as nn

class MaskPriorDecoder(nn.Module):
    def __init__(self, in_channels, mask_channels=128):
        super().__init__()
        # 学习每个mask的权重
        self.mask_weight = nn.Parameter(torch.ones(mask_channels))  # [128]
        # 将 mask 压缩到特征通道
        self.mask_proj = nn.Conv2d(mask_channels, in_channels, kernel_size=1)

    def forward(self, query_feat, sam_masks):
        """
        query_feat: [B, C, H, W]
        sam_masks: [B, 128, H, W]
        """
        # mask加权
        weighted_masks = sam_masks * self.mask_weight.view(1, -1, 1, 1)  # [B,128,H,W]
        # 压缩到query特征通道
        mask_feat = self.mask_proj(weighted_masks)  # [B, C, H, W]

        # 将 mask 特征与 query 特征融合
        enhanced_feat = query_feat + mask_feat

        return enhanced_feat

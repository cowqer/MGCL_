import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet
from torchvision.models import vgg
from typing_extensions import override
import PIL.Image as Image
import torchvision.transforms.v2 as transforms2
from torch.utils.data import Dataset
from .net_tools import MGCLNetwork, SegmentationHead, MGCDModule, MGFEModule


class PromptAttentionModule(nn.Module):
    def __init__(self, in_channels, prompt_dim, num_heads=4):
        super().__init__()
        self.in_channels = in_channels
        self.prompt_dim = prompt_dim
        self.num_heads = num_heads
        
        self.query_proj = nn.Conv2d(in_channels, prompt_dim, kernel_size=1)
        self.key_proj = nn.Linear(prompt_dim, prompt_dim)
        self.value_proj = nn.Linear(prompt_dim, prompt_dim)
        self.out_proj = nn.Conv2d(prompt_dim, in_channels, kernel_size=1)

    def forward(self, feats, prompt_tokens):
        """
        feats: [B, C, H, W] - backbone特征图
        prompt_tokens: [B, M, C] - 从 mask 区域池化而来
        """
        B, C, H, W = feats.shape
        M = prompt_tokens.shape[1]

        # 1. 将 query 从图上投影出来（[B, H*W, D]）
        Q = self.query_proj(feats).flatten(2).transpose(1, 2)  # [B, HW, D]
        K = self.key_proj(prompt_tokens)                       # [B, M, D]
        V = self.value_proj(prompt_tokens)                     # [B, M, D]

        # 2. 计算 attention map（[B, HW, M]）
        attn = torch.matmul(Q, K.transpose(1, 2)) / (self.prompt_dim ** 0.5)
        attn = F.softmax(attn, dim=-1)

        # 3. 加权求和：把 prompt 融入图上每一点
        out = torch.matmul(attn, V)  # [B, HW, D]
        out = out.transpose(1, 2).view(B, self.prompt_dim, H, W)
        out = self.out_proj(out)     # [B, C, H, W]

        return feats + out  # 残差增强

###############>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>SegmentationHead-Start<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<################

class SegmentationHead_MGSA(SegmentationHead):
    def __init__(self, in_channels_list, prompt_dim=256, num_heads=4):
        """
        in_channels_list: list[int], 每层query_feats的通道数
        """
        super().__init__()
        self.num_layers = len(in_channels_list)
        self.prompt_dim = prompt_dim

        # 初始化每层对应的PromptAttentionModule
        self.prompt_attentions = nn.ModuleList([
            PromptAttentionModule(in_ch, prompt_dim, num_heads)
            for in_ch in in_channels_list
        ])
        
    def get_prompt_tokens(self, support_feats, support_label, M=5):
        """
        简单示例：根据support_label在support_feats上做区域池化，得到M个token。
        这里直接做全局池化，假设前景mask和背景mask。
        你可以根据实际设计，采样多个区域池化。

        Args:
            support_feats: list[tensor], 每层支持图特征 [B,C,H,W]
            support_label: tensor [B,H,W], 前景标签掩码
            M: int, token数目，示例固定值

        Returns:
            prompt_tokens: tensor [B,M,D]，D=self.prompt_dim
        """
        B = support_label.size(0) if support_label.dim() == 4 else support_label.shape[0]
        device = support_feats[0].device

        # 这里示例只用第一层support_feats做池化
        sf = support_feats[0]  # [B,C,H,W]
        B, C, H, W = sf.shape

        # 对support_label做resize到sf大小，假设support_label形状是[B,H,W]
        support_label_resized = F.interpolate(support_label.unsqueeze(1).float(), size=(H, W), mode='nearest').squeeze(1)  # [B,H,W]

        # 对前景区域做平均池化作为prompt token第1个token
        fg_mask = (support_label_resized > 0.5).float().unsqueeze(1)  # [B,1,H,W]
        fg_token = (sf * fg_mask).sum(dim=[2,3]) / (fg_mask.sum(dim=[2,3]) + 1e-6)  # [B,C]

        # 对背景区域做平均池化作为prompt token第2个token
        bg_mask = (support_label_resized <= 0.5).float().unsqueeze(1)
        bg_token = (sf * bg_mask).sum(dim=[2,3]) / (bg_mask.sum(dim=[2,3]) + 1e-6)  # [B,C]

        # 如果M > 2，用零向量补齐
        if self.prompt_dim != C:
            # 用线性层映射sf通道到prompt_dim
            mapper = nn.Linear(C, self.prompt_dim).to(device)
            fg_token = mapper(fg_token)
            bg_token = mapper(bg_token)

        pad_tokens = torch.zeros(B, M-2, self.prompt_dim, device=device)
        prompt_tokens = torch.cat([fg_token.unsqueeze(1), bg_token.unsqueeze(1), pad_tokens], dim=1)  # [B,M,D]

        return prompt_tokens

    def forward(self, query_feats, support_feats, support_label, query_mask, support_masks):
        # MGFE
        _query_feats, _support_feats = MGFEModule.update_feature(
            query_feats, support_feats, query_mask, support_masks)
        
        query_feats = [torch.concat([one, two], dim=1) for one, two in zip(query_feats, _query_feats)]
        support_feats = [torch.concat([one, two], dim=1) for one, two in zip(support_feats, _support_feats)]

        # 假设你从 support_label 里生成 prompt_tokens，具体方式根据你设计
        # 这里演示用 support_feats[0] 做平均池化模拟 prompt_tokens（B, M, C）
        # 实际你要基于mask区域池化
        # 生成prompt tokens
        prompt_tokens = self.get_prompt_tokens(support_feats, support_label)

        # 对query_feats每层用prompt attention融合
        enhanced_query_feats = []
        for i, feat in enumerate(query_feats):
            enhanced = self.prompt_attentions[i](feat, prompt_tokens)
            enhanced_query_feats.append(enhanced)

        # 后续流程
        support_feats_fg = [self.label_feature(
            support_feat, support_label.clone()) for support_feat in support_feats]
        support_feats_bg = [self.label_feature(
            support_feat, (1 - support_label).clone()) for support_feat in support_feats]
        corr_fg = self.multilayer_correlation(query_feats, support_feats_fg)
        corr_bg = self.multilayer_correlation(query_feats, support_feats_bg)
        corr = [torch.concatenate([fg_one[:, None], bg_one[:, None]],
                                  dim=1) for fg_one, bg_one in zip(corr_fg, corr_bg)]

        logit = self.mgcd(corr[::-1], query_mask)
        return logit


###############>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>SegmentationHead-Over<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<################

###############>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>Network-Start<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<################
class MGSANetwork(MGCLNetwork):
    """
    MGSANet is a subclass of MGCLNetwork that implements the Multi-Granularity Attention Network.
    It uses the same backbone and segmentation head but focuses on multi-granularity attention mechanisms.
    """
    def __init__(self, args):
        super().__init__(args)
        self.segmentation_head = SegmentationHead_MGSA()  # Reuse SegmentationHead for MGSANet
        pass

    # Additional methods specific to MGSANet can be added here
###############>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>Network-Over<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<################
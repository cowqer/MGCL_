import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet
from torchvision.models import vgg
from typing_extensions import override
import PIL.Image as Image
import torchvision.transforms.v2 as transforms2
from torch.utils.data import Dataset
from .modules import *
from .net_tools import SegmentationHead, MGCDModule, MGFEModule, MGCLNetwork


class SegmentationHead_FBC(SegmentationHead):
    """
    SegmentationHead_FBC_MGCL is a subclass of SegmentationHead_FBC that implements the Multi-Granularity Attention Network.
    It uses the same backbone and segmentation head but focuses on multi-granularity attention mechanisms.
    """
    def __init__(self):
        super().__init__()
        self.mgcd = MGCDModule([2, 2, 2])  # Initialize MGCDModule with specific parameters
        pass
    
    def forward(self, query_feats, support_feats, support_label, query_mask, support_masks):
            # MGFE 
            _query_feats, _support_feats = MGFEModule.update_feature(query_feats, support_feats, query_mask, support_masks)
            
            query_feats = [torch.concat([one, two], dim=1) for one, two in zip(query_feats, _query_feats)]
            support_feats = [torch.concat([one, two], dim=1) for one, two in zip(support_feats, _support_feats)]
            
            label = support_label.unsqueeze(1)  # [B,1,H0,W0]

            feat = support_feats[2]  # shape: [16, 2048, 8, 8]

            proto_fg = masked_avg_pool(feat, label)            # 前景 prototype: [16, 2048]
            proto_bg = masked_avg_pool(feat, 1 - label)        # 背景 prototype: [16, 2048]

            support_prototypes_fg = proto_fg
            support_prototypes_bg = proto_bg

            alpha = 0.5
            query_feat = query_feats[2]  # 取最后一层特征
            
            prior_fg, prior_bg = compute_query_prior(query_feat, support_prototypes_fg, support_prototypes_bg, temperature=1.0)
            prior = torch.sigmoid(prior_fg - prior_bg)
            
            query_prototypes_fg = get_query_foreground_prototype(query_feat, prior)  # [B, C]
            prototype_fg = 0.5 * query_prototypes_fg + (1.0 - alpha) * support_prototypes_fg

            prototype_fg = prototype_fg.unsqueeze(-1).unsqueeze(-1)   # [16, 4096, 1, 1]
            prototype_fg = prototype_fg.expand(-1, -1, 8, 8)    
            
            support_feats[2]= support_feats[2] + prototype_fg
            query_feats[2] = query_feats[2] + prototype_fg
            # print("query_feats", [f.shape for f in query_feats])  # Debugging line to check shapes
            
            # FBC
            support_feats_fg = [self.label_feature(
                support_feat, support_label.clone())for support_feat in support_feats]
            support_feats_bg = [self.label_feature(
                support_feat, (1 - support_label).clone())for support_feat in support_feats]
            
            corr_fg = self.multilayer_correlation(query_feats, support_feats_fg)
            corr_bg = self.multilayer_correlation(query_feats, support_feats_bg)
            corr = [torch.concatenate([fg_one[:, None], bg_one[:, None]],
                                    dim=1) for fg_one, bg_one in zip(corr_fg, corr_bg)]

            # MGCD
            logit = self.mgcd(corr[::-1], query_mask)
            return logit
        
class SegmentationHead_FBC_1(SegmentationHead):

    def __init__(self):
        super().__init__()
        self.mgcd = MGCDModule([2, 2, 2])

        pass

    def forward(self, query_feats, support_feats, support_label, query_mask, support_masks):
            # MGFE 
            _query_feats, _support_feats = MGFEModule.update_feature(query_feats, support_feats, query_mask, support_masks)
            
            query_feats = [torch.concat([one, two], dim=1) for one, two in zip(query_feats, _query_feats)]
            support_feats = [torch.concat([one, two], dim=1) for one, two in zip(support_feats, _support_feats)]
            
            label = support_label.unsqueeze(1)  # [B,1,H0,W0]

            feat = support_feats[2]  # shape: [16, 2048, 8, 8]

            proto_fg = masked_avg_pool(feat, label)            # 前景 prototype: [16, 2048]
            proto_bg = masked_avg_pool(feat, 1 - label)        # 背景 prototype: [16, 2048]

            support_prototypes_fg = proto_fg
            support_prototypes_bg = proto_bg

            alpha = 0.5
            query_feat = query_feats[2]  # 取最后一层特征
            
            prior_fg, prior_bg = compute_query_prior(query_feat, support_prototypes_fg, support_prototypes_bg, temperature=1.0)
            prior = torch.sigmoid(prior_fg - prior_bg)
            
            query_prototypes_fg = get_query_foreground_prototype(query_feat, prior)  # [B, C]
            prototype_fg = 0.5 * query_prototypes_fg + (1.0 - alpha) * support_prototypes_fg

            prototype_fg = prototype_fg.unsqueeze(-1).unsqueeze(-1)   # [16, 4096, 1, 1]
            prototype_fg = prototype_fg.expand(-1, -1, 8, 8)    
            
            support_feats[2]= support_feats[2] + prototype_fg
            query_feats[2] = query_feats[2] + prototype_fg
            
            support_feats[2] = F.dropout(support_feats[2], p=0.2, training=self.training)
            query_feats[2] = F.dropout(query_feats[2], p=0.2, training=self.training)
            # print("query_feats", [f.shape for f in query_feats])  # Debugging line to check shapes
            
            # FBC
            support_feats_fg = [self.label_feature(
                support_feat, support_label.clone())for support_feat in support_feats]
            support_feats_bg = [self.label_feature(
                support_feat, (1 - support_label).clone())for support_feat in support_feats]
            
            corr_fg = self.multilayer_correlation(query_feats, support_feats_fg)
            corr_bg = self.multilayer_correlation(query_feats, support_feats_bg)
            corr = [torch.concatenate([fg_one[:, None], bg_one[:, None]],
                                    dim=1) for fg_one, bg_one in zip(corr_fg, corr_bg)]

            # MGCD
            logit = self.mgcd(corr[::-1], query_mask)
            return logit


class SegmentationHead_FBC_2(SegmentationHead):

    def __init__(self):
        super().__init__()
        self.mgcd = MGCDModule([2, 2, 2])
        
        self.self_gate = nn.Sequential(
        nn.AdaptiveAvgPool2d(1),  # [B, C, 1, 1] —— 全局平均池化
        nn.Conv2d(4096, 1, kernel_size=1),  # 从通道数 C→1，C是query_feats[2]的通道数
        nn.Sigmoid()  # 输出α∈[0,1]
)
        pass

    def forward(self, query_feats, support_feats, support_label, query_mask, support_masks):
            # MGFE 
            _query_feats, _support_feats = MGFEModule.update_feature(query_feats, support_feats, query_mask, support_masks)
            
            query_feats = [torch.concat([one, two], dim=1) for one, two in zip(query_feats, _query_feats)]
            support_feats = [torch.concat([one, two], dim=1) for one, two in zip(support_feats, _support_feats)]
            
            label = support_label.unsqueeze(1)  # [B,1,H0,W0]

            feat = support_feats[2]  # shape: [16, 2048, 8, 8]

            proto_fg = masked_avg_pool(feat, label)            # 前景 prototype: [16, 2048]
            proto_bg = masked_avg_pool(feat, 1 - label)        # 背景 prototype: [16, 2048]

            support_prototypes_fg = proto_fg
            support_prototypes_bg = proto_bg

            alpha = self.self_gate(query_feats[2])  # [B, 1, 1, 1]
            alpha = alpha.view(-1, 1)  # reshape to [B, 1] for prototype blending
            
            query_feat = query_feats[2]  # 取最后一层特征
            
            prior_fg, prior_bg = compute_query_prior(query_feat, support_prototypes_fg, support_prototypes_bg, temperature=1.0)
            prior = torch.sigmoid(prior_fg - prior_bg)
            
            query_prototypes_fg = get_query_foreground_prototype(query_feat, prior)  # [B, C]
            prototype_fg = 0.5 * query_prototypes_fg + (1.0 - alpha) * support_prototypes_fg

            prototype_fg = prototype_fg.unsqueeze(-1).unsqueeze(-1)   # [16, 4096, 1, 1]
            prototype_fg = prototype_fg.expand(-1, -1, 8, 8)    
            
            support_feats[2]= support_feats[2] + prototype_fg
            query_feats[2] = query_feats[2] + prototype_fg
            # print("query_feats", [f.shape for f in query_feats])  # Debugging line to check shapes
            
            # FBC
            support_feats_fg = [self.label_feature(
                support_feat, support_label.clone())for support_feat in support_feats]
            support_feats_bg = [self.label_feature(
                support_feat, (1 - support_label).clone())for support_feat in support_feats]
            
            corr_fg = self.multilayer_correlation(query_feats, support_feats_fg)
            corr_bg = self.multilayer_correlation(query_feats, support_feats_bg)
            corr = [torch.concatenate([fg_one[:, None], bg_one[:, None]],
                                    dim=1) for fg_one, bg_one in zip(corr_fg, corr_bg)]

            # MGCD
            logit = self.mgcd(corr[::-1], query_mask)
            return logit


class MG_FBCNetwork(MGCLNetwork):
    """
    MGSANet is a subclass of MGCLNetwork that implements the Multi-Granularity Attention Network.
    It uses the same backbone and segmentation head but focuses on multi-granularity attention mechanisms.
    """
    def __init__(self, args):
        super().__init__(args)
        self.segmentation_head = SegmentationHead_FBC()  # Reuse SegmentationHead for MGSANet
        pass
    
class MG_FBC_1Network(MGCLNetwork):
    """
    MGSANet is a subclass of MGCLNetwork that implements the Multi-Granularity Attention Network.
    It uses the same backbone and segmentation head but focuses on multi-granularity attention mechanisms.
    """
    def __init__(self, args):
        super().__init__(args)
        self.segmentation_head = SegmentationHead_FBC_1()  # Reuse SegmentationHead for MGSANet
        pass
    
class MG_FBC_2Network(MGCLNetwork):
    """
    MGSANet is a subclass of MGCLNetwork that implements the Multi-Granularity Attention Network.
    It uses the same backbone and segmentation head but focuses on multi-granularity attention mechanisms.
    """
    def __init__(self, args):
        super().__init__(args)
        self.segmentation_head = SegmentationHead_FBC_2()  # Reuse SegmentationHead for MGSANet
        pass
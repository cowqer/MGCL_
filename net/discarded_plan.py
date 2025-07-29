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
## 7.28 discarded plan
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

        alphas = [0.5, 0.5, 0.5]
        spatial_sizes = [32, 16, 8]
        
        query_prototypes_fg = []
        support_prototypes_fg = []
        
        for i in range(3):
        # Support prototypes
            fg_proto = masked_avg_pool(support_feats[i], label)
            bg_proto = masked_avg_pool(support_feats[i], 1 - label)

            # Query priors
            prior_fg, prior_bg = compute_query_prior(query_feats[i], fg_proto, bg_proto)
            prior = torch.sigmoid(prior_fg - prior_bg)

            # Query prototypes
            query_proto_fg = get_query_foreground_prototype(query_feats[i], prior)

            # Combine prototypes
            proto_fg_support = alphas[i] * query_proto_fg + (1.0 - alphas[i]) * fg_proto
            proto_fg_query = alphas[i] * fg_proto + (1.0 - alphas[i]) * query_proto_fg
            
            
            # Broadcast and add to features
            proto_fg = proto_fg.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, spatial_sizes[i], spatial_sizes[i])
            support_feats[i] = support_feats[i] + proto_fg_support
            query_feats[i] = query_feats[i] + proto_fg_query

            query_prototypes_fg.append(query_proto_fg)
            support_prototypes_fg.append(fg_proto)
  
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

        alphas = [0.5, 0.5, 0.5]
        spatial_sizes = [32, 16, 8]
        
        query_prototypes_fg = []
        support_prototypes_fg = []
        
        for i in range(3):
        # Support prototypes
            fg_proto = masked_avg_pool(support_feats[i], label)
            bg_proto = masked_avg_pool(support_feats[i], 1 - label)

            # Query priors
            prior_fg, prior_bg = compute_query_prior(query_feats[i], fg_proto, bg_proto)
            prior = torch.sigmoid(prior_fg - prior_bg)

            # Query prototypes
            query_proto_fg = get_query_foreground_prototype(query_feats[i], prior)

            # Combine prototypes
            proto_fg_support = alphas[i] * query_proto_fg + (1.0 - alphas[i]) * fg_proto
            proto_fg_query = alphas[i] * fg_proto + (1.0 - alphas[i]) * query_proto_fg
            
            
            # Broadcast and add to features
            proto_fg_support = proto_fg_support.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, spatial_sizes[i], spatial_sizes[i])
            proto_fg_query = proto_fg_query.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, spatial_sizes[i], spatial_sizes[i])
            if i != 2:
                support_feats[i] = support_feats[i] + proto_fg_support
                query_feats[i] = query_feats[i] + proto_fg_query

            query_prototypes_fg.append(query_proto_fg)
            support_prototypes_fg.append(fg_proto)
  
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
            
            prior_fg, prior_bg = compute_query_prior(query_feat, support_prototypes_fg, support_prototypes_bg)
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
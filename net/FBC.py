import torch
import torch.nn as nn
import torch.nn.functional as F

from .CD import SSCDModule_v1, MGCDModule_v3, MGCDModule_v2, MGCDModule_v1, SSCDModule_v3
from .modules import *
from .net_tools import SegmentationHead, MGCDModule, MGFEModule, MGCLNetwork
from .net_tools_pro import CDModule , myNetwork
from .blocks.mask_enhance import MaskEnhancer ,priority_decay_masks
import torch
import torch.nn.functional as F

def weighted_masked_avg_pool(feat, mask):
    """
    feat: [B, C, H, W]
    mask: [B, 1, H, W] 或 [B, H, W]（概率值 0~1）
    return: [B, C]
    使用软权重的加权平均（适用于 pseudo mask）
    """
    if mask.dim() == 3:
        mask = mask.unsqueeze(1)
    b, c, h, w = feat.shape
    mask_flat = mask.view(b, 1, -1)                     # B x 1 x HW
    feat_flat = feat.view(b, c, -1)                     # B x C x HW

    weighted_sum = torch.bmm(feat_flat, mask_flat.permute(0, 2, 1)).squeeze(-1)  # B x C
    area = mask_flat.sum(dim=2).clamp(min=1e-6)         # B x 1
    proto = weighted_sum / area                        # B x C
    return proto

def batched_cosine_sim(a, b, eps=1e-8):
    """
    a: [B, C] or [B, K, C]
    b: [B, C]
    return:
      if a is [B, K, C]: returns [B, K] (cosine between each k and b)
      if a is [B, C]: returns [B] (cosine)
    """
    if a.dim() == 3:
        # [B, K, C] vs [B, C] -> [B, K]
        b_exp = b.unsqueeze(1)                         # B x 1 x C
        num = (a * b_exp).sum(dim=2)                   # B x K
        a_norm = a.norm(dim=2)                         # B x K
        b_norm = b.norm(dim=1).unsqueeze(1)            # B x 1
        return num / (a_norm * b_norm + eps)
    else:
        num = (a * b).sum(dim=1)
        denom = a.norm(dim=1) * b.norm(dim=1) + eps
        return num / denom

def resize_mask(mask, target_h, target_w):
    """
    mask: [B, M, H, W]
    输出: [B, M, target_h, target_w]
    """
    mask = mask.float()
    return F.interpolate(mask, size=(target_h, target_w), mode="bilinear", align_corners=False)

class SegmentationHead_mgcd_fbc(SegmentationHead):
    """
    SegmentationHead_FBC_MGCL is a subclass of SegmentationHead_FBC that implements the Multi-Granularity Attention Network.
    It uses the same backbone and segmentation head but focuses on multi-granularity attention mechanisms.
    """
    def __init__(self):
        super().__init__()
        self.mgcd = MGCDModule([2, 2, 2])  # Initialize MGCDModule with specific parameters
        pass
    
    def forward(self, query_feats, support_feats, support_label, query_mask, support_masks):
        
            # query_feats, support_feats = FGE(query_feats, support_feats, support_label, query_mask, alpha=0.5)
        
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
        
class SegmentationHead_mgcd_fge(SegmentationHead):
    """
    SegmentationHead_FBC_MGCL is a subclass of SegmentationHead_FBC that implements the Multi-Granularity Attention Network.
    It uses the same backbone and segmentation head but focuses on multi-granularity attention mechanisms.
    """
    def __init__(self):
        super().__init__()
        self.mgcd = MGCDModule([2, 2, 2])  # Initialize MGCDModule with specific parameters
        pass
    
    def forward(self, query_feats, support_feats, support_label, query_mask, support_masks):
        
            query_feats, support_feats = FGE(query_feats, support_feats, support_label, query_mask, alpha=0.5)
        
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

class SegmentationHead_w_o_sam(SegmentationHead):
    """
    SegmentationHead_FBC_MGCL is a subclass of SegmentationHead_FBC that implements the Multi-Granularity Attention Network.
    It uses the same backbone and segmentation head but focuses on multi-granularity attention mechanisms.
    """
    def __init__(self):
        super().__init__()
        self.mgcd = CDModule([2, 2, 2])  # Initialize MGCDModule with specific parameters
        pass
    
    def forward(self, query_feats, support_feats, support_label, query_mask, support_masks):
 
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
        

class SegmentationHead_w_omgfe(SegmentationHead):
    """
    SegmentationHead_FBC_MGCL is a subclass of SegmentationHead_FBC that implements the Multi-Granularity Attention Network.
    It uses the same backbone and segmentation head but focuses on multi-granularity attention mechanisms.
    """
    def __init__(self):
        super().__init__()
        self.mgcd = MGCDModule([2, 2, 2])  # Initialize MGCDModule with specific parameters
        pass
    
    def forward(self, query_feats, support_feats, support_label, query_mask, support_masks):

            # Foregroud Background Correlation (FBC)
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
        

class SegmentationHead_FBC_3(SegmentationHead):

    def __init__(self):
        super().__init__()
        self.mgcd = MGCDModule([2, 2, 2])

        self.alpha = nn.Parameter(torch.tensor(0.0))  # 初始为 0，经sigmoid后为0.5
        self.beta = nn.Parameter(torch.tensor(0.0))
        pass

    def forward(self, query_feats, support_feats, support_label, query_mask, support_masks):
            # MGFE 
            # _query_feats, _support_feats = MGFEModule.update_feature(query_feats, support_feats, query_mask, support_masks)
            
            # query_feats = [torch.concat([one, two], dim=1) for one, two in zip(query_feats, _query_feats)]
            # support_feats = [torch.concat([one, two], dim=1) for one, two in zip(support_feats, _support_feats)]
            
            label = support_label.unsqueeze(1)  # [B,1,H0,W0]

            feat = support_feats[2]  # shape: [16, 2048, 8, 8]
            
            proto_fg = masked_avg_pool(feat, label)            # 前景 prototype: [16, 2048]
            proto_bg = masked_avg_pool(feat, 1 - label)        # 背景 prototype: [16, 2048]

            support_prototypes_fg = proto_fg
            support_prototypes_bg = proto_bg

            query_feat_2 = query_feats[2]  # 取最后一层特征
            support_feat_2 = support_feats[2]  # 取最后一层特征
            
            alpha = 0.5
            beta = 1.0 - alpha

            prior_fg, prior_bg = compute_query_prior(query_feat_2, support_prototypes_fg, support_prototypes_bg, temperature=1.1)
            prior = torch.sigmoid(prior_fg - prior_bg)
            
            query_prototypes_fg = get_query_foreground_prototype(query_feat_2, prior)  # [B, C]
            prototype_fg = alpha * query_prototypes_fg + beta * support_prototypes_fg

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

class SegmentationHead_FBC_3_v1(SegmentationHead):

    def __init__(self):
        super().__init__()
        self.mgcd = MGCDModule([2, 2, 2])

        self.alpha = nn.Parameter(torch.tensor(0.0))# 初始为 0，经sigmoid后为0.5
        # self.beta = nn.Parameter(torch.tensor(0.0))
        pass

    def forward(self, query_feats, support_feats, support_label, query_mask, support_masks):
            # MGFE 

            label = support_label.unsqueeze(1)  # [B,1,H0,W0]

            feat = support_feats[2]  # shape: [16, 2048, 8, 8]
            
            proto_fg = masked_avg_pool(feat, label)            # 前景 prototype: [16, 2048]
            proto_bg = masked_avg_pool(feat, 1 - label)        # 背景 prototype: [16, 2048]

            support_prototypes_fg = proto_fg
            support_prototypes_bg = proto_bg

            query_feat_2 = query_feats[2]  # 取最后一层特征
            support_feat_2 = support_feats[2]  # 取最后一层特征
            
            alpha = torch.sigmoid(self.alpha)   # 初始值 0.5
            beta = 1.0 - alpha

            prior_fg, prior_bg = compute_query_prior(query_feat_2, support_prototypes_fg, support_prototypes_bg, temperature=1.1)
            prior = torch.sigmoid(prior_fg - prior_bg)
            
            query_prototypes_fg = get_query_foreground_prototype(query_feat_2, prior)  # [B, C]
            prototype_fg = alpha * query_prototypes_fg + beta * support_prototypes_fg

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

class SegmentationHead_FBC_3_v2(SegmentationHead):

    def __init__(self):
        super().__init__()
        # self.mgcd = MGCDModule([2, 2, 2])
        self.mgcd = SSCDModule_v1([2, 2, 2])
        self.alpha = nn.Parameter(torch.tensor(0.0))  # 初始为 0，经sigmoid后为0.5
        self.beta = nn.Parameter(torch.tensor(0.0))
        pass

    def forward(self, query_feats, support_feats, support_label, query_mask, support_masks):
            
            label = support_label.unsqueeze(1)  # [B,1,H0,W0]

            feat = support_feats[2]  # shape: [16, 2048, 8, 8]
            
            proto_fg = masked_avg_pool(feat, label)            # 前景 prototype: [16, 2048]
            proto_bg = masked_avg_pool(feat, 1 - label)        # 背景 prototype: [16, 2048]

            support_prototypes_fg = proto_fg
            support_prototypes_bg = proto_bg

            query_feat_2 = query_feats[2]  # 取最后一层特征
            support_feat_2 = support_feats[2]  # 取最后一层特征
            
            alpha = 0.5
            beta = 1.0 - alpha

            prior_fg, prior_bg = compute_query_prior(query_feat_2, support_prototypes_fg, support_prototypes_bg, temperature=1.1)
            prior = torch.sigmoid(prior_fg - prior_bg)
            
            query_prototypes_fg = get_query_foreground_prototype(query_feat_2, prior)  # [B, C]
            prototype_fg = alpha * query_prototypes_fg + beta * support_prototypes_fg

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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class SegmentationHead_FBC_3_v3(SegmentationHead):
    

    #*在2的基础上引入了两个masks做增强
    def __init__(self):
        super().__init__()
        # self.mgcd = MGCDModule([2, 2, 2])
        self.mgcd = SSCDModule_v1([2, 2, 2])
        self.alpha = nn.Parameter(torch.tensor(0.0))  # 初始为 0，经sigmoid后为0.5
        self.beta = nn.Parameter(torch.tensor(0.0))

        pass

    def forward(self, query_feats, support_feats, support_label, query_mask, support_masks):
            
            # print("query_feats_0", query_feats[0].shape) # [16, 512, 32, 32]
            # print("query_feats_1", query_feats[1].shape) # [16, 1024, 16, 16]
            # print("query_feats_2", query_feats[2].shape) # [16, 2048, 8, 8]

            # print("query_mask", query_mask.shape) # [16, 128, 256, 256]
            # print("support_label", support_label.shape) # [16, 256, 256]
            #!#################MASK_ENHACE_BEGIN###################
            support_masks = priority_decay_masks(support_masks, decay=0.5)
            query_mask = priority_decay_masks(query_mask, decay=0.5)
            q_low = query_feats[0]
            s_low = support_feats[0]

            B, C, H, W = q_low.shape

            # 将 mask 下采样到低维特征大小
            q_mask_resized = resize_mask(query_mask, H, W)   # [B, M, H, W]
            s_mask_resized = resize_mask(support_masks, H, W)

            # 简单优先级衰减，避免重叠过强干扰
            q_mask_clean = priority_decay_masks(q_mask_resized, decay=0.5).mean(1, keepdim=True)  # [B,1,H,W]
            s_mask_clean = priority_decay_masks(s_mask_resized, decay=0.5).mean(1, keepdim=True)

            # 利用 mask 对特征做加权增强
            query_feats[0] = q_low * (1 + q_mask_clean)
            support_feats[0] = s_low * (1 + s_mask_clean)
            #!#################MASK_ENHANCE_END####################
            
            #!#################FGE BEGIN###################
            label = support_label.unsqueeze(1)  # [B,1,H0,W0]
            feat = support_feats[2]  # shape: [16, 2048, 8, 8]取高维信息作为先验输入
            
            proto_fg = masked_avg_pool(feat, label)            # 前景 prototype: [16, 2048]
            proto_bg = masked_avg_pool(feat, 1 - label)        # 背景 prototype: [16, 2048]
            
            support_prototypes_fg = proto_fg
            support_prototypes_bg = proto_bg
            
            query_feat_2 = query_feats[2]  # 取最后一层特征
            support_feat_2 = support_feats[2]  # 取最后一层特征
            
            alpha = 0.5
            beta = 1.0 - alpha

            prior_fg, prior_bg = compute_query_prior(query_feat_2, support_prototypes_fg, support_prototypes_bg, temperature=1.1)
            prior = torch.sigmoid(prior_fg - prior_bg)
            
            query_prototypes_fg = get_query_foreground_prototype(query_feat_2, prior)  # [B, C]
            prototype_fg = alpha * query_prototypes_fg + beta * support_prototypes_fg

            prototype_fg = prototype_fg.unsqueeze(-1).unsqueeze(-1)   # [16, 4096, 1, 1]
            prototype_fg = prototype_fg.expand(-1, -1, 8, 8)    
            
            support_feats[2]= support_feats[2] + prototype_fg
            query_feats[2] = query_feats[2] + prototype_fg
            
            #!#################FGE ENDDING###################

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
class SegmentationHead_FBC_3_v4(SegmentationHead):
    #*SSCDv1+mgfev2
    def __init__(self):
        super().__init__()
        # self.mgcd = MGCDModule([2, 2, 2])
        self.mgcd = SSCDModule_v3([2, 2, 2])
        self.alpha = nn.Parameter(torch.tensor(0.0))  # 初始为 0，经sigmoid后为0.5
        self.beta = nn.Parameter(torch.tensor(0.0))

        pass

    def forward(self, query_feats, support_feats, support_label, query_mask, support_masks):
            
            # print("query_feats_0", query_feats[0].shape) # [16, 512, 32, 32]
            # print("query_feats_1", query_feats[1].shape) # [16, 1024, 16, 16]
            # print("query_feats_2", query_feats[2].shape) # [16, 2048, 8, 8]

            # print("query_mask", query_mask.shape) # [16, 128, 256, 256]
            # print("support_label", support_label.shape) # [16, 256, 256]
            #!#################MASK_ENHACE_BEGIN###################
            
            #!#################MASK_ENHANCE_END####################
            
            #!#################FGE BEGIN###################
            label = support_label.unsqueeze(1)  # [B,1,H0,W0]
            feat = support_feats[2]  # shape: [16, 2048, 8, 8]取高维信息作为先验输入
            
            proto_fg = masked_avg_pool(feat, label)            # 前景 prototype: [16, 2048]
            proto_bg = masked_avg_pool(feat, 1 - label)        # 背景 prototype: [16, 2048]
            
            support_prototypes_fg = proto_fg
            support_prototypes_bg = proto_bg
            
            query_feat_2 = query_feats[2]  # 取最后一层特征
            support_feat_2 = support_feats[2]  # 取最后一层特征
            
            alpha = 0.5
            beta = 1.0 - alpha

            prior_fg, prior_bg = compute_query_prior(query_feat_2, support_prototypes_fg, support_prototypes_bg, temperature=1.1)
            prior = torch.sigmoid(prior_fg - prior_bg)
            
            query_prototypes_fg = get_query_foreground_prototype(query_feat_2, prior)  # [B, C]
            prototype_fg = alpha * query_prototypes_fg + beta * support_prototypes_fg

            prototype_fg = prototype_fg.unsqueeze(-1).unsqueeze(-1)   # [16, 4096, 1, 1]
            prototype_fg = prototype_fg.expand(-1, -1, 8, 8)    
            
            support_feats[2]= support_feats[2] + prototype_fg
            query_feats[2] = query_feats[2] + prototype_fg
            
            #!#################FGE ENDDING###################

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
        
class SegmentationHead_FBC_3_v5(SegmentationHead):
    #*参考了PANet + 
    def __init__(self):
        super().__init__()
        # self.mgcd = MGCDModule([2, 2, 2])
        self.mgcd = SSCDModule_v1([2, 2, 2])
        self.alpha = nn.Parameter(torch.tensor(0.0))  # 初始为 0，经sigmoid后为0.5
        self.beta = nn.Parameter(torch.tensor(0.0))

        pass

    def forward(self, query_feats, support_feats, support_label, query_mask, support_masks):
            
            # print("query_feats_0", query_feats[0].shape) # [16, 512, 32, 32]
            # print("query_feats_1", query_feats[1].shape) # [16, 1024, 16, 16]
            # print("query_feats_2", query_feats[2].shape) # [16, 2048, 8, 8]

            # print("query_mask", query_mask.shape) # [16, 128, 256, 256]
            # print("support_label", support_label.shape) # [16, 256, 256]
            
            #!#################FGE BEGIN###################
            label = support_label.unsqueeze(1)  # [B,1,H0,W0]
            feat = support_feats[2]  # shape: [16, 2048, 8, 8]取高维信息作为先验输入
            
            support_prototypes_fg = masked_avg_pool(feat, label)            # 前景 prototype: [16, 2048]
            support_prototypes_bg = masked_avg_pool(feat, 1 - label)        # 背景 prototype: [16, 2048]
            
            query_feat_2 = query_feats[2]  # 取最后一层特征
            support_feat_2 = support_feats[2]  # 取最后一层特征
            
            alpha = 0.5
            beta = 1.0 - alpha

            prior_fg, prior_bg = compute_query_prior(query_feat_2, support_prototypes_fg, support_prototypes_bg, temperature=1.1)
            prior = torch.sigmoid(prior_fg - prior_bg)
            
            query_prototypes_fg = get_query_foreground_prototype(query_feat_2, prior)  # [B, C]
            prototype_fg = alpha * query_prototypes_fg + beta * support_prototypes_fg

            prototype_fg = prototype_fg.unsqueeze(-1).unsqueeze(-1)   # [16, 4096, 1, 1]
            prototype_fg = prototype_fg.expand(-1, -1, 8, 8)    
            
            support_feats[2]= support_feats[2] + prototype_fg
            query_feats[2] = query_feats[2] + prototype_fg
            
            #!#################FGE ENDDING###################

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
            coarse_logit = self.mgcd(corr[::-1], query_mask)   # [B, 2, Hq, Wq] 假设二分类 (bg, fg)

            # 将 coarse_logit 当作 pseudo mask 的来源
            # 取前景概率 (soft)
            coarse_prob = torch.softmax(coarse_logit, dim=1)   # [B, 2, Hq, Wq]
            pseudo_fg_prob = coarse_prob[:, 1:2, :, :]         # [B,1,Hq,Wq]

            # 1) 计算 query 伪前景原型 (伪原型)

            query_feat_for_pool = query_feats[2]               # [B, C, h, w]

            if pseudo_fg_prob.shape[2:] != query_feat_for_pool.shape[2:]:
                pseudo_fg_resized = F.interpolate(pseudo_fg_prob, size=query_feat_for_pool.shape[2:], mode='bilinear', align_corners=False)
            else:
                pseudo_fg_resized = pseudo_fg_prob

          # confidence filtering
            conf_thresh = 0.7
            conf_mask = (pseudo_fg_resized > conf_thresh).float()
            pseudo_fg_filtered = pseudo_fg_resized * conf_mask

            # normalize to avoid all-zero
            norm = pseudo_fg_filtered.sum(dim=(2,3), keepdim=True) + 1e-6
            pseudo_fg_filtered = pseudo_fg_filtered / norm

            # detach to avoid gradient flow back to coarse
            pseudo_fg_filtered = pseudo_fg_filtered.detach()
  
            
            pseudo_query_proto = weighted_masked_avg_pool(query_feat_for_pool, pseudo_fg_filtered)  # [B, C]

            # 2) 准备 support 原型（你之前计算得到 support_prototypes_fg）
            # support_prototypes_fg 可能是 [B, C]（K=1）或 [B, K, C]
            support_proto = support_prototypes_fg  # 你之前的变量名
            # 如果你的 support_prototypes_fg 是 [B, C], 下面代码也能工作（视为 K=1）

            # 3) 计算 affinity scores（每个 support 与 query 伪原型的相似度）
            # 使用余弦相似度并指数化（论文是 exp(cos)）
            # 输出 weights_sum->1 across K
            if support_proto.dim() == 2:  # [B, C] -> 单支持
                # 转为 [B, 1, C] 以统一后续处理
                support_proto_k = support_proto.unsqueeze(1)      # [B,1,C]
            else:
                support_proto_k = support_proto                      # [B,K,C]

            cos_sim = batched_cosine_sim(support_proto_k, pseudo_query_proto)  # [B, K]
            affinity = torch.exp(cos_sim)                                     # [B,K]
            affinity = affinity / (affinity.sum(dim=1, keepdim=True) + 1e-6)  # softmax-like normalization

            # 4) rectified prototype: weighted sum over K supports
            # support_proto_k: [B, K, C], affinity: [B, K]
            affinity_exp = affinity.unsqueeze(-1)                               # [B,K,1]
            rectified_proto = (support_proto_k * affinity_exp).sum(dim=1)       # [B, C]
            rectified_proto = rectified_proto.detach()  # 不反传梯度
            
            # 5) 用 rectified_proto 替换原先用于匹配的 support prototype，重做一次 FGE / 后续流程
            # 将 rectified_proto 注入到原来 prototype 生成功能里（扩展并加到 feature）
            rectified_proto_spatial = rectified_proto.unsqueeze(-1).unsqueeze(-1)  # [B,C,1,1]
            rectified_proto_spatial = rectified_proto_spatial.expand(-1, -1, query_feat_for_pool.shape[2], query_feat_for_pool.shape[3])

            # 将 rectified prototype 融入到 support_feats[2] / query_feats[2]（视你原先的设计）
            support_feats[2] = support_feats[2] + rectified_proto_spatial
            query_feats[2] = query_feats[2] + rectified_proto_spatial

            # 6) 重新计算 support_feats_fg/bg, correlation, mgcd，得到 refined_logit
            support_feats_fg = [self.label_feature(support_feat, support_label.clone()) for support_feat in support_feats]
            support_feats_bg = [self.label_feature(support_feat, (1 - support_label).clone()) for support_feat in support_feats]

            corr_fg = self.multilayer_correlation(query_feats, support_feats_fg)
            corr_bg = self.multilayer_correlation(query_feats, support_feats_bg)
            corr = [torch.concatenate([fg_one[:, None], bg_one[:, None]], dim=1) for fg_one, bg_one in zip(corr_fg, corr_bg)]

            refined_logit = self.mgcd(corr[::-1], query_mask)  # 最终输出

            return coarse_logit, refined_logit

class SegmentationHead_FBC_3_MGCD_v1(SegmentationHead):

    def __init__(self):
        super().__init__()
        self.mgcd = MGCDModule_v1([2, 2, 2])
        # self.alpha = nn.Parameter(torch.tensor(0.0))  # 初始为 0，经sigmoid后为0.5
        # self.beta = nn.Parameter(torch.tensor(0.0))
        pass

    def forward(self, query_feats, support_feats, support_label, query_mask, support_masks):
            
            label = support_label.unsqueeze(1)  # [B,1,H0,W0]

            feat = support_feats[2]  # shape: [16, 2048, 8, 8]
            
            proto_fg = masked_avg_pool(feat, label)            # 前景 prototype: [16, 2048]
            proto_bg = masked_avg_pool(feat, 1 - label)        # 背景 prototype: [16, 2048]

            support_prototypes_fg = proto_fg
            support_prototypes_bg = proto_bg

            query_feat_2 = query_feats[2]  # 取最后一层特征
            support_feat_2 = support_feats[2]  # 取最后一层特征
            
            alpha = 0.5
            beta = 1.0 - alpha

            prior_fg, prior_bg = compute_query_prior(query_feat_2, support_prototypes_fg, support_prototypes_bg, temperature=1.1)
            prior = torch.sigmoid(prior_fg - prior_bg)
            
            query_prototypes_fg = get_query_foreground_prototype(query_feat_2, prior)  # [B, C]
            prototype_fg = alpha * query_prototypes_fg + beta * support_prototypes_fg

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

class SegmentationHead_FBC_3_MGCD_v2(SegmentationHead):

    def __init__(self):
        super().__init__()
        self.mgcd = MGCDModule_v2([2, 2, 2])
        # self.alpha = nn.Parameter(torch.tensor(0.0))  # 初始为 0，经sigmoid后为0.5
        # self.beta = nn.Parameter(torch.tensor(0.0))
        pass

    def forward(self, query_feats, support_feats, support_label, query_mask, support_masks):
            
            label = support_label.unsqueeze(1)  # [B,1,H0,W0]

            feat = support_feats[2]  # shape: [16, 2048, 8, 8]
            
            proto_fg = masked_avg_pool(feat, label)            # 前景 prototype: [16, 2048]
            proto_bg = masked_avg_pool(feat, 1 - label)        # 背景 prototype: [16, 2048]

            support_prototypes_fg = proto_fg
            support_prototypes_bg = proto_bg

            query_feat_2 = query_feats[2]  # 取最后一层特征
            support_feat_2 = support_feats[2]  # 取最后一层特征
            
            alpha = 0.5
            beta = 1.0 - alpha

            prior_fg, prior_bg = compute_query_prior(query_feat_2, support_prototypes_fg, support_prototypes_bg, temperature=1.1)
            prior = torch.sigmoid(prior_fg - prior_bg)
            
            query_prototypes_fg = get_query_foreground_prototype(query_feat_2, prior)  # [B, C]
            prototype_fg = alpha * query_prototypes_fg + beta * support_prototypes_fg

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

class SegmentationHead_FBC_3_MGCD_v3(SegmentationHead):

    def __init__(self):
        super().__init__()
        self.mgcd = MGCDModule_v3([2, 2, 2])
        # self.alpha = nn.Parameter(torch.tensor(0.0))  # 初始为 0，经sigmoid后为0.5
        # self.beta = nn.Parameter(torch.tensor(0.0))
        pass

    def forward(self, query_feats, support_feats, support_label, query_mask, support_masks):
            
            label = support_label.unsqueeze(1)  # [B,1,H0,W0]

            feat = support_feats[2]  # shape: [16, 2048, 8, 8]
            
            proto_fg = masked_avg_pool(feat, label)            # 前景 prototype: [16, 2048]
            proto_bg = masked_avg_pool(feat, 1 - label)        # 背景 prototype: [16, 2048]

            support_prototypes_fg = proto_fg
            support_prototypes_bg = proto_bg

            query_feat_2 = query_feats[2]  # 取最后一层特征
            support_feat_2 = support_feats[2]  # 取最后一层特征
            
            alpha = 0.5
            beta = 1.0 - alpha

            prior_fg, prior_bg = compute_query_prior(query_feat_2, support_prototypes_fg, support_prototypes_bg, temperature=1.1)
            prior = torch.sigmoid(prior_fg - prior_bg)
            
            query_prototypes_fg = get_query_foreground_prototype(query_feat_2, prior)  # [B, C]
            prototype_fg = alpha * query_prototypes_fg + beta * support_prototypes_fg

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

###############从8.3之后开始的FBC网络，重新命名，按照数字顺序#####################

class MG_FBC_3Network(MGCLNetwork):
    """
    MGSANet is a subclass of MGCLNetwork that implements the Multi-Granularity Attention Network.
    It uses the same backbone and segmentation head but focuses on multi-granularity attention mechanisms.
    """
    def __init__(self, args):
        super().__init__(args)
        self.segmentation_head = SegmentationHead_FBC_3()  # Reuse SegmentationHead for MGSANet
        pass

class MG_FBC_3_v1Network(MGCLNetwork):
    """
    MGSANet is a subclass of MGCLNetwork that implements the Multi-Granularity Attention Network.
    It uses the same backbone and segmentation head but focuses on multi-granularity attention mechanisms.
    """
    def __init__(self, args):
        super().__init__(args)
        self.segmentation_head = SegmentationHead_FBC_3_v1()  # Reuse SegmentationHead for MGSANet
        pass

class MG_FBC_3_v2Network(MGCLNetwork):
    """
    """
    def __init__(self, args):
        super().__init__(args)
        self.segmentation_head = SegmentationHead_FBC_3_v2()  # Reuse SegmentationHead for MGSANet
        pass
    
class MG_FBC_3_v3Network(MGCLNetwork):
    """
    """
    def __init__(self, args):
        super().__init__(args)
        self.segmentation_head = SegmentationHead_FBC_3_v3()  # Reuse SegmentationHead for MGSANet
        pass
    
class MG_FBC_3_v4Network(MGCLNetwork):
    """
    """
    def __init__(self, args):
        super().__init__(args)
        self.segmentation_head = SegmentationHead_FBC_3_v4()  # Reuse SegmentationHead for MGSANet
        pass
    
class MG_FBC_3_v5Network(MGCLNetwork):
    """
    """
    def __init__(self, args):
        super().__init__(args)
        self.segmentation_head = SegmentationHead_FBC_3_v5()  # Reuse SegmentationHead for MGSANet
        pass
    
    def forward(self, query_img, support_img, support_label, query_mask, support_masks):
        if self.finetune_backbone:
            query_feats = self.extract_feats(query_img, self.backbone)
            support_feats = self.extract_feats(support_img, self.backbone)
        else:
            with torch.no_grad():
                query_feats = self.extract_feats(query_img, self.backbone)
                support_feats = self.extract_feats(support_img, self.backbone)
                pass
            pass
        # MGFE, FBC, MGCD
        coarse_logit, refined_logit = self.segmentation_head(query_feats, support_feats, support_label.clone(), query_mask, support_masks)
        coarse_logit = F.interpolate(coarse_logit, support_img.size()[2:], mode='bilinear', align_corners=True)
        refined_logit = F.interpolate(refined_logit, support_img.size()[2:], mode='bilinear', align_corners=True)
        return coarse_logit, refined_logit 
    
    def predict_nshot(self, batch):
        nshot = batch["support_imgs"].shape[1]
        logit_label_agg = 0
        for s_idx in range(nshot):
            coarse_logit, refined_logit = self.forward(
                batch['query_img'], batch['support_imgs'][:, s_idx],  batch['support_labels'][:, s_idx],
                query_mask=batch['query_mask'] if 'query_mask' in batch and self.args.mask else None,
                support_masks=batch['support_masks'][:, s_idx] if 'support_masks' in batch and self.args.mask else None)
            # alpha = 0.5  # 可以调节，取值范围 [0,1]
            # logit_label = alpha * coarse_logit + (1 - alpha) * refined_logit
            logit_label = refined_logit
            result_i = logit_label.argmax(dim=1).clone()
            logit_label_agg += result_i

            # One-Shot
            if nshot == 1: return result_i.float()
            pass

        # Few-Shot
        # Average & quantize predictions given threshold (=0.5)
        bsz = logit_label_agg.size(0)
        max_vote = logit_label_agg.view(bsz, -1).max(dim=1)[0]
        max_vote = torch.stack([max_vote, torch.ones_like(max_vote).long()])
        max_vote = max_vote.max(dim=0)[0].view(bsz, 1, 1)
        pred_label = logit_label_agg.float() / max_vote
        threshold = 0.4
        pred_label[pred_label < threshold] = 0
        pred_label[pred_label >= threshold] = 1
        return pred_label
    
class MGCD_v1Network(MGCLNetwork):
    """
    """
    def __init__(self, args):
        super().__init__(args)
        self.segmentation_head = SegmentationHead_FBC_3_MGCD_v1()  # Reuse SegmentationHead for MGSANet
        pass

class MGCD_v2Network(MGCLNetwork):
    """
    """
    def __init__(self, args):
        super().__init__(args)
        self.segmentation_head = SegmentationHead_FBC_3_MGCD_v2()  # Reuse SegmentationHead for MGSANet
        pass
    
class MGCD_v3Network(MGCLNetwork):
    """
    """
    def __init__(self, args):
        super().__init__(args)
        self.segmentation_head = SegmentationHead_FBC_3_MGCD_v3()  # Reuse SegmentationHead for MGSANet
        pass


    
class Test_Network(MGCLNetwork):
    """
    MGSANet is a subclass of MGCLNetwork that implements the Multi-Granularity Attention Network.
    It uses the same backbone and segmentation head but focuses on multi-granularity attention mechanisms.
    """
    def __init__(self, args):
        super().__init__(args)
        self.segmentation_head = SegmentationHead_w_omgfe()  # Reuse SegmentationHead for MGSANet
        pass

    def forward(self, query_img, support_img, support_label, query_mask, support_masks):
        if self.finetune_backbone:

            mask_query_prior = torch.mean(query_mask.float(), dim=1, keepdim=True)  # [B,1,H,W]
            mask_query_feat = mask_query_prior.repeat(1, 3, 1, 1)                   # [B,3,H,W]
            query_img = query_img + mask_query_feat
            
            mask_support_prior = torch.mean(support_masks.float(), dim=1, keepdim=True)
            mask_support_feat = mask_support_prior.repeat(1, 3, 1, 1)               # [B,3,H,W]
            support_img = support_img + mask_support_feat
            
            query_feats = self.extract_feats(query_img, self.backbone)
            support_feats = self.extract_feats(support_img, self.backbone)
        else:
            with torch.no_grad():
                query_feats = self.extract_feats(query_img, self.backbone)
                support_feats = self.extract_feats(support_img, self.backbone)
                pass
            pass

        logit = self.segmentation_head(query_feats, support_feats, support_label.clone(), query_mask, support_masks)
        logit = F.interpolate(logit, support_img.size()[2:], mode='bilinear', align_corners=True)
        return logit


class Test_samv1_Network(myNetwork):
    """
    MGSANet is a subclass of MGCLNetwork that implements the Multi-Granularity Attention Network.
    It uses the same backbone and segmentation head but focuses on multi-granularity attention mechanisms.
    """
    def __init__(self, args):
        super().__init__(args)
        self.segmentation_head = SegmentationHead_w_omgfe()  # Reuse SegmentationHead for MGSANet
        pass

    def forward(self, query_img, support_img, support_label, query_mask, support_masks):
        if self.finetune_backbone:

            mask_query_prior = torch.mean(query_mask.float(), dim=1, keepdim=True)  # [B,1,H,W]
            mask_query_feat = mask_query_prior.repeat(1, 3, 1, 1)                   # [B,3,H,W]
            query_img = query_img + mask_query_feat
            
            mask_support_prior = torch.mean(support_masks.float(), dim=1, keepdim=True)
            mask_support_feat = mask_support_prior.repeat(1, 3, 1, 1)               # [B,3,H,W]
            support_img = support_img + mask_support_feat
            
            query_feats = self.extract_feats(query_img, self.backbone)
            support_feats = self.extract_feats(support_img, self.backbone)
        else:
            with torch.no_grad():
                query_feats = self.extract_feats(query_img, self.backbone)
                support_feats = self.extract_feats(support_img, self.backbone)
                pass
            pass

        logit = self.segmentation_head(query_feats, support_feats, support_label.clone(), query_mask, support_masks)
        logit = F.interpolate(logit, support_img.size()[2:], mode='bilinear', align_corners=True)
        return logit
    

class nosam_Network(MGCLNetwork):
    
    def __init__(self, args):
        super().__init__(args)
        self.segmentation_head = SegmentationHead_w_o_sam()  # Reuse SegmentationHead for MGSANet
        pass
    
class mgcd_fge_Network(MGCLNetwork):
    
    def __init__(self, args):
        super().__init__(args)
        self.segmentation_head = SegmentationHead_mgcd_fge()  # Reuse SegmentationHead for MGSANet
        pass
    
class mgcd_fbc_Network(MGCLNetwork):
    
    def __init__(self, args):
        super().__init__(args)
        self.segmentation_head = SegmentationHead_mgcd_fbc()  # Reuse SegmentationHead for MGSANet
        pass
    

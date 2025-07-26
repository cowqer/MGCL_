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
            
            prior_fg, prior_bg = compute_query_prior(query_feat, support_prototypes_fg, support_prototypes_bg)
            prior = torch.sigmoid(prior_fg - prior_bg)
            
            # w_fg = torch.sigmoid(prior_fg)
            # w_bg = torch.sigmoid(prior_bg)
            # prior = w_fg / (w_fg + w_bg + 1e-8)
            
            alpha = 0.5
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

        ##query_feats shapes: [torch.Size([16, 512, 32, 32]), torch.Size([16, 1024, 16, 16]), torch.Size([16, 2048, 8, 8])]
        ##support_feats shapes: [torch.Size([16, 512, 32, 32]), torch.Size([16, 1024, 16, 16]), torch.Size([16, 2048, 8, 8])]
        
        ##query_feats after MGFEModule [torch.Size([16, 1024, 32, 32]), torch.Size([16, 2048, 16, 16]), torch.Size([16, 4096, 8, 8])]
        ##support_feats after MGFEModule [torch.Size([16, 1024, 32, 32]), torch.Size([16, 2048, 16, 16]), torch.Size([16, 4096, 8, 8])]
        
        _query_feats, _support_feats = MGFEModule.update_feature(query_feats, support_feats, query_mask, support_masks)
        
        query_feats = [torch.concat([one, two], dim=1) for one, two in zip(query_feats, _query_feats)]
        support_feats = [torch.concat([one, two], dim=1) for one, two in zip(support_feats, _support_feats)]
        
        # print("query_feats after MGFEModule", [f.shape for f in query_feats])
        # print("support_feats after MGFEModule", [f.shape for f in support_feats])
        label = support_label.unsqueeze(1)  # [B,1,H0,W0]

        query_feat_2 = query_feats[2]  # 取最后一层特征
        supp_feat_2 = support_feats[2]  # shape: [16, 2048, 8, 8]
        query_feat_1 = query_feats[1]  # 取最后一层特征
        supp_feat_1 = support_feats[1]  # shape: [16, 2048, 8, 8]
        query_feat_0 = query_feats[0]  # 取最后一层特征
        supp_feat_0 = support_feats[0]  # shape: [16, 2048, 8, 8]

        support_prototypes_fg_2 = masked_avg_pool(supp_feat_2, label)            # 前景 prototype: [16, 2048]
        support_prototypes_bg_2 = masked_avg_pool(supp_feat_2, 1 - label)        # 背景 prototype: [16, 2048]
        support_prototypes_fg_1 = masked_avg_pool(supp_feat_1, label)            # 前景 prototype: [16, 2048]
        support_prototypes_bg_1 = masked_avg_pool(supp_feat_1, 1 - label)        # 背景 prototype: [16, 2048]
        support_prototypes_fg_0 = masked_avg_pool(supp_feat_0, label)            # 前景 prototype: [16, 2048]
        support_prototypes_bg_0 = masked_avg_pool(supp_feat_0, 1 - label)        # 背景 prototype: [16, 2048]

        prior_fg_2, prior_bg_2 = compute_query_prior(query_feat_2, support_prototypes_fg_2, support_prototypes_bg_2)
        prior_fg_1, prior_bg_1 = compute_query_prior(query_feat_1, support_prototypes_fg_1, support_prototypes_bg_1)
        prior_fg_0, prior_bg_0 = compute_query_prior(query_feat_0, support_prototypes_fg_0, support_prototypes_bg_0)
        
        prior2 = torch.sigmoid(prior_fg_2 - prior_bg_2)
        prior1 = torch.sigmoid(prior_fg_1 - prior_bg_1)
        prior0 = torch.sigmoid(prior_fg_0 - prior_bg_0)
            
        print("prior2", prior2.shape, "prior1", prior1.shape, "prior0", prior0.shape)
            
        alpha = 0.5
        query_prototypes_fg_2 = get_query_foreground_prototype(query_feat_2, prior2)  # [B, C]
        prototype_fg = 0.5 * query_prototypes_fg_2 + (1.0 - alpha) * support_prototypes_fg_2

        prototype_fg = prototype_fg.unsqueeze(-1).unsqueeze(-1)   # [16, 4096, 1, 1]
        prototype_fg = prototype_fg.expand(-1, -1, 8, 8)    
        
        support_feats[2]= support_feats[2] + prototype_fg
        query_feats[2] = query_feats[2] + prototype_fg


        # print("query_feats", [f.shape for f in query_feats])
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

    pass

class SegmentationHead_FBC_2(SegmentationHead):

    def __init__(self):
        super().__init__()
        self.mgcd = MGCDModule([2, 2, 2])

        pass

    def forward(self, query_feats, support_feats, support_label, query_mask, support_masks):
        # MGFE

        _query_feats, _support_feats = MGFEModule.update_feature(query_feats, support_feats, query_mask, support_masks)
        
        query_feats = [torch.concat([one, two], dim=1) for one, two in zip(query_feats, _query_feats)]
        support_feats = [torch.concat([one, two], dim=1) for one, two in zip(support_feats, _support_feats)]
        
        # print("query_feats after MGFEModule", [f.shape for f in query_feats])
        # print("support_feats after MGFEModule", [f.shape for f in support_feats])
        label = support_label.unsqueeze(1)  # [B,1,H0,W0]

        feat_1, feat_2, feat_3 = support_feats[:3] # shape: [16, 2048, 8, 8]

        proto_fg_1 = masked_avg_pool(feat_1, label)            # 前景 prototype: [16, 2048]
        proto_bg_1 = masked_avg_pool(feat_1, 1 - label)        # 背景 prototype: [16, 2048]
        proto_fg_2 = masked_avg_pool(feat_2, label)            # 前景 prototype: [16, 2048]
        proto_bg_2 = masked_avg_pool(feat_2, 1 - label)        # 背景 prototype: [16, 2048]
        proto_fg_3 = masked_avg_pool(feat_3, label)            # 前景 prototype: [16, 2048]
        proto_bg_3 = masked_avg_pool(feat_3, 1 - label)        # 背景 prototype: [16, 2048]
        ###
        
        support_prototypes_fg = [proto_fg_1, proto_fg_2, proto_fg_3]
        support_prototypes_bg = [proto_bg_1, proto_bg_2, proto_bg_3]

        alpha = 0.5
        new_query_feats   = []
        new_support_feats = []
        priors = []
        for i, (q_feat, s_feat, supp_fg, supp_bg) in enumerate(zip(
                query_feats, support_feats,
                support_prototypes_fg, support_prototypes_bg)):

            # 1. 计算 prior
            prior_fg, prior_bg = compute_query_prior(q_feat, supp_fg, supp_bg)
            prior = torch.sigmoid(prior_fg - prior_bg)
            priors.append(prior)
            # 2. 计算 query 上的前景 prototype
            query_proto_fg = get_query_foreground_prototype(q_feat, prior)  # [B, C]

            # 3. 融合 support 和 query prototype
            #    这里把 (1-α) 改成 1.0−alpha，对应你原来 0.5 * Q + 0.5 * S
            proto_fg = alpha * query_proto_fg + (1.0 - alpha) * supp_fg  # [B, C]

            # 4. reshape → [B, C, H, W]
            B, C, H, W = q_feat.shape
            proto_map = proto_fg.view(B, C, 1, 1).expand(B, C, H, W)

            # 5. 累加到 query_feats[i] 和 support_feats[i]
            new_query_feats.append(q_feat + proto_map)
            new_support_feats.append(s_feat + proto_map)

        # 最后把更新后的 list 赋回去
        print("priors", [p.shape for p in priors])
        query_feats   = new_query_feats
        support_feats = new_support_feats

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

    pass

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
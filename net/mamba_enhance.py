import torch
import torch.nn as nn

from .blocks.mamba_blocks import VSSM  # 假设你已经有 VSSM 实现
from .modules import *
from .net_tools import SegmentationHead, MGCDModule, MGFEModule, MGCLNetwork
from .net_tools_pro import *
from  torch.cuda.amp import autocast, GradScaler
device = torch.device("cuda:0")

def Weighted_GAP(supp_feat, mask):
    supp_feat = supp_feat * mask
    feat_h, feat_w = supp_feat.shape[-2:][0], supp_feat.shape[-2:][1]
    area = F.avg_pool2d(mask, (supp_feat.size()[2], supp_feat.size()[3])) * feat_h * feat_w + 0.0005
    supp_feat = F.avg_pool2d(input=supp_feat, kernel_size=supp_feat.shape[-2:]) * feat_h * feat_w / area
    return supp_feat

class FSQ_Mamba(nn.Module):
    """
    Few-shot 特征交互模块
    输入: F_q, F_s
    处理: 压缩 -> 融合 -> Mamba
    输出: F_q', F_s'
    """

    def __init__(self, fea_dim=1536, reduce_dim=256, blocks=8):
        """
        Args:
            fea_dim: 输入特征的维度 (resnet: layer2+layer3 -> 1024+512)
            reduce_dim: 压缩后的维度
            blocks: Mamba 堆叠层数
        """
        super(FSQ_Mamba, self).__init__()

        # 压缩 query 特征
        self.down_query = nn.Sequential(
            nn.Conv2d(fea_dim, reduce_dim, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5)
        )

        # 压缩 support 特征
        self.down_supp = nn.Sequential(
            nn.Conv2d(fea_dim, reduce_dim, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5)
        )

        # 融合 query 特征 (concat 后再压缩)
        self.init_merge_query = nn.Sequential(
            nn.Conv2d(reduce_dim * 2, reduce_dim, kernel_size=1, bias=False),
            nn.ReLU(inplace=True)
        )

        # 融合 support 特征
        self.init_merge_supp = nn.Sequential(
            nn.Conv2d(reduce_dim * 2, reduce_dim, kernel_size=1, bias=False),
            nn.ReLU(inplace=True)
        )

        # ----------------------------
        # Mamba Block (核心交互模块)
        # ----------------------------

        self.mamba = VSSM(
            depths=8,
            dims=[256],
            mlp_ratio=1
        )

    def forward(self, F_q, F_s):
        """
        Args:
            F_q: [B, C, H, W] query 特征
            F_s: [B, C, H, W] support 特征

        Returns:
            F_q_new, F_s_new : 经过 Mamba 交互后的特征
        """
        # 1. 压缩
        F_q = self.down_query(F_q)   # [B, 256, H, W]
        F_s = self.down_supp(F_s)    # [B, 256, H, W]

        # 2. 融合 (这里简单 concat)
        F_q_merge = self.init_merge_query(torch.cat([F_q, F_s], dim=1))
        F_s_merge = self.init_merge_supp(torch.cat([F_s, F_q], dim=1))

        # 3. Mamba 交互
        F_q_new, F_s_new = self.mamba(F_q_merge, F_s_merge)

        return F_q_new, F_s_new
    
class SegmentationHeadV3(SegmentationHead):
    """
    SegmentationHead_FBC_MGCL is a subclass of SegmentationHead_FBC that implements the Multi-Granularity Attention Network.
    It uses the same backbone and segmentation head but focuses on multi-granularity attention mechanisms.
    """
    def __init__(self):
        super().__init__()
        self.mgcd = MGCDModule([2, 2, 2])  # Initialize MGCDModule with specific parameters
        # self.spatial_attn = SpatialAttention(kernel_size=7) 

        pass
    
    def forward(self, query_feats, support_feats, support_label, query_mask, support_masks):
        # MGFE
        # 1️⃣ 定义你的增强模块
        
        support_feat0, support_feat1,support_feat2 = support_feats[0],support_feats[1],support_feats[2]
        query_feats0, support_feat1, support_feat2 = query_feats[0],query_feats[1],query_feats[2]

        # 2️⃣ 调用 update_feature
        _query_feats, _support_feats = MGFEModule.update_feature(query_feats, support_feats, query_mask, support_masks)
        # _query_feats, _support_feats = MGFEModuleV2.update_feature(query_feats, support_feats, query_mask, support_masks,attn_support, attn_query)

        query_feats = [torch.concat([one, two], dim=1) for one, two in zip(query_feats, _query_feats)]
        support_feats = [torch.concat([one, two], dim=1) for one, two in zip(support_feats, _support_feats)]
        
        # print("query_feats after MGFEModule", [f.shape for f in query_feats])
        # print("support_feats after MGFEModule", [f.shape for f in support_feats])
        
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

class SegmentationHead_Mamba(SegmentationHead):

    def __init__(self):
        super().__init__()
        self.mgcd = MGCDModule([2, 2, 2])
        fea_dim = 3072
        reduce_dim = 512
        self.down_query = nn.Sequential(
            nn.Conv2d(fea_dim, reduce_dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5)
        )
        self.down_supp = nn.Sequential(
            nn.Conv2d(fea_dim, reduce_dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5)
        )
        self.init_merge_query = nn.Sequential(
            nn.Conv2d(reduce_dim * 2 , 512, kernel_size=1, padding=0, bias=False),
        )
        self.init_merge_supp = nn.Sequential(
            nn.Conv2d(reduce_dim * 2 , 512, kernel_size=1, padding=0, bias=False),
        )
        self.mamba = VSSM(
            depths=[8],
            dims=[512],
            mlp_ratio=1
        )
        self.relu = nn.ReLU(inplace=True)
        self.alpha = nn.Parameter(torch.tensor(0.0))  # 初始为 0，经sigmoid后为0.5
        self.beta = nn.Parameter(torch.tensor(0.0))
        pass

    def forward(self, query_feats, support_feats, support_label, query_mask, support_masks):
            #!
            size = (32, 32)

            query_feat2_upsampled = F.interpolate(query_feats[2], size=query_feats[1].shape[-2:], mode='bilinear', align_corners=True)
            
            query_feats_mid = torch.cat([query_feats[1], query_feat2_upsampled], dim=1) # [B, 3072, 16, 16]
            query_feats_mid =self.down_query(query_feats_mid) # [B, 512, 16, 16]
            query_feat_m = F.interpolate(query_feats_mid, size=size, mode='bilinear', align_corners=True) # [B, 512, 64, 64]
            
            fts_size = query_feat_m.size()[-2:] # (64, 64)
            
            sepport_feats_upsampled = F.interpolate(support_feats[2], size=support_feats[1].shape[-2:], mode='bilinear', align_corners=True)
            support_feats_mid = torch.cat([support_feats[1], sepport_feats_upsampled], dim=1) # [B, 3072, 16, 16]
            support_feats_mid = self.down_supp(support_feats_mid) # [B, 512, 16, 16]
            support_feat_m = F.interpolate(support_feats_mid, size=size, mode='bilinear', align_corners=True) # [B, 512, 64, 64]
            
            mask = support_label.unsqueeze(1) # [B,1,256,256]
            mask = F.interpolate(mask.float(), size=fts_size, mode='nearest') # 
            # print("mask", mask.shape)
            # mask = F.interpolate(mask, size=fts_size, mode='bilinear', align_corners=True) 
            
            bs = support_feat_m.size()[0]
            
            # support_feat_m = support_feat_m.view(bs, -1, fts_size[0], fts_size[1]) # [B,512,64,64]
            support_pro = Weighted_GAP(support_feat_m, mask) #
            support_pro = support_pro.expand_as(query_feat_m)  # [B,512,64,64]
            # print("support_pro", support_pro.shape)
            
            # query_cat = torch.cat([query_feat_m, support_pro, corr_mask], dim=1) #* bs, 512, 60, 60, C: reduce_dim * 2 + 2=>reduce_dim
            query_cat = torch.cat([query_feat_m, support_pro], dim=1)#([B, 1024, 64, 64])

            query_feat_m = self.init_merge_query(query_cat)#([4, 512, 64, 64])
       
            # supp_cat = torch.cat([support_feat_m, support_pro, supp_mask], dim=1)  #* bs, 512, 60, 60, C: reduce_dim * 2 + 1 => reduce_dim
            supp_cat = torch.cat([support_feat_m, support_pro], dim=1)
            supp_feat_m = self.init_merge_supp(supp_cat)  # bs, 256, 60, 60 
            
            query_feat_m, support_feat_m = self.mamba(query_feat_m, supp_feat_m)
            
            #!
            # MGFE 
            query_feats[0]= query_feats[0] + query_feat_m
            support_feats[0] = support_feats[0] + support_feat_m
            #
            # print("query_feats before MGFEModule", [f.shape for f in query_feats])
            # print("support_feats before MGFEModule", [f.shape for f in support_feats])
            _query_feats, _support_feats = MGFEModule.update_feature(query_feats, support_feats, query_mask, support_masks)
            
            query_feats = [torch.concat([one, two], dim=1) for one, two in zip(query_feats, _query_feats)]
            support_feats = [torch.concat([one, two], dim=1) for one, two in zip(support_feats, _support_feats)]
            
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
    
class MG_mambaNetwork(MGCLNetwork):
    """
    MGSANet is a subclass of MGCLNetwork that implements the Multi-Granularity Attention Network.
    It uses the same backbone and segmentation head but focuses on multi-granularity attention mechanisms.
    """
    def __init__(self, args):
        super().__init__(args)
        self.segmentation_head = SegmentationHead_Mamba()  # Reuse SegmentationHead for MGSANet
        pass
    

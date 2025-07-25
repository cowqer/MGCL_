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

class CrossAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.query_proj = nn.Conv2d(in_channels, in_channels, 1)
        self.key_proj = nn.Conv2d(in_channels, in_channels, 1)
        self.value_proj = nn.Conv2d(in_channels, in_channels, 1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, query, support):
        B, C, H, W = query.shape

        Q = self.query_proj(query).view(B, C, -1)  # B x C x HW
        K = self.key_proj(support).view(B, C, -1)
        V = self.value_proj(support).view(B, C, -1)

        attn = torch.bmm(Q.transpose(1, 2), K) / (C ** 0.5)  # B x HW_q x HW_s
        attn = self.softmax(attn)

        out = torch.bmm(attn, V.transpose(1, 2))  # B x HW_q x C
        out = out.transpose(1, 2).view(B, C, H, W)
        return out + query


    
class SegmentationHead_MG_FBC(SegmentationHead):
    def forward(self, query_feats, support_feats, support_label, query_mask, support_masks):
        # MGFE
        _query_feats, _support_feats = MGFEModule.update_feature(
            query_feats, support_feats, query_mask, support_masks)
        query_feats = [torch.concat([one, two], dim=1) for one, two in zip(query_feats, _query_feats)]
        support_feats = [torch.concat([one, two], dim=1) for one, two in zip(support_feats, _support_feats)]

        # FBC
        support_feats_fg = [self.label_feature(
            support_feat, support_label.clone())for support_feat in support_feats]
        support_feats_bg = [self.label_feature(
            support_feat, (1 - support_label).clone())for support_feat in support_feats]
        corr_fg = self.multilayer_correlation(query_feats, support_feats_fg)
        corr_bg = self.multilayer_correlation(query_feats, support_feats_bg)
        print("corr_fg", corr_fg[0].shape, corr_bg[0].shape)
        
        corr = [torch.concatenate([fg_one[:, None], bg_one[:, None]],
                                  dim=1) for fg_one, bg_one in zip(corr_fg, corr_bg)]

        # MGCD
        logit = self.mgcd(corr[::-1], query_mask)
        return logit
    
###############>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>SegmentationHead-Over<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<################

###############>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>Network-Start<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<################

class MG_FBCNetwork(MGCLNetwork):
    """
    MGANet is a subclass of MGCLNetwork that implements the Multi-Granularity Attention Network.
    It uses the same backbone and segmentation head but focuses on multi-granularity attention mechanisms.
    """
    def __init__(self, args):
        super().__init__(args)
        self.segmentation_head = SegmentationHead_MG_FBC()  
        pass   
###############>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>Network-Over<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<################
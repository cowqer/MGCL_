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
from .net_tools_pro import *

device = torch.device("cuda:0")

class SegmentationHeadV2(SegmentationHead):
    """
    SegmentationHead_FBC_MGCL is a subclass of SegmentationHead_FBC that implements the Multi-Granularity Attention Network.
    It uses the same backbone and segmentation head but focuses on multi-granularity attention mechanisms.
    """
    def __init__(self):
        super().__init__()
        self.mgcd = MGCDModule([2, 2, 2])  # Initialize MGCDModule with specific parameters
        self.spatial_attn = SpatialAttention(kernel_size=7) 
        pass
    
    def forward(self, query_feats, support_feats, support_label, query_mask, support_masks):
        # MGFE
        # 1️⃣ 定义你的增强模块
        
        # support_feat0, support_feat1,support_feat2 = support_feats[0],support_feats[1],support_feats[2]
        # query_feats0, support_feat1, support_feat2 = query_feats[0],query_feats[1],query_feats[2]
        MGFEModuleV2.pool_mode = 'topk'
        MGFEModuleV2.topk = 10
        # 2️⃣ 调用 update_feature
        _query_feats, _support_feats = MGFEModuleV2.update_feature(query_feats, support_feats, query_mask, support_masks)
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
                
class MG_SSENetwork(MGCLNetwork):
    """
    MGSANet is a subclass of MGCLNetwork that implements the Multi-Granularity Attention Network.
    It uses the same backbone and segmentation head but focuses on multi-granularity attention mechanisms.
    """
    def __init__(self, args):
        super().__init__(args)
        self.segmentation_head = SegmentationHeadV2()  # Reuse SegmentationHead for MGSANet
        pass
    

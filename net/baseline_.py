import torch
import torch.nn as nn
import torch.nn.functional as F

from .net_tools_pro import myNetwork, SegmentationHead_baseline
from .modules import FGE

class SegmentationHead_FGE(SegmentationHead_baseline):
    """
    FGE + FBC + CD
    """
    def __init__(self):
        super().__init__()
        pass
    
    def forward(self, query_feats, support_feats, support_label, query_mask, support_masks):
        
        query_feats, support_feats = FGE(
        query_feats, support_feats, support_label, query_mask, alpha=0.5
        )
        
        # FBC
        support_feats_fg = [self.label_feature(
            support_feat, support_label.clone())for support_feat in support_feats]
        support_feats_bg = [self.label_feature(
            support_feat, (1 - support_label).clone())for support_feat in support_feats]
        corr_fg = self.multilayer_correlation(query_feats, support_feats_fg)
        corr_bg = self.multilayer_correlation(query_feats, support_feats_bg)
        corr = [torch.concatenate([fg_one[:, None], bg_one[:, None]],
                                  dim=1) for fg_one, bg_one in zip(corr_fg, corr_bg)]
        # CD Correlation Decoder
        logit = self.cd(corr[::-1])
        return logit

    pass

class FGE_baseline_Network(myNetwork):
    """
    MGSANet is a subclass of MGCLNetwork that implements the Multi-Granularity Attention Network.
    It uses the same backbone and segmentation head but focuses on multi-granularity attention mechanisms.
    """
    def __init__(self, args):
        super().__init__(args)
        self.segmentation_head = SegmentationHead_FGE()  # Reuse SegmentationHead for MGSANet
        pass

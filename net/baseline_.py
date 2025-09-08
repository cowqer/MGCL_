import torch
import torch.nn as nn
import torch.nn.functional as F
from .net_tools import SegmentationHead,MGCLNetwork
from .net_tools_pro import myNetwork, SegmentationHead_baseline
from .modules import FGE
from .CD import SSCDModule, MCDModule,SSCDModule_v1,SSCDModule_v2

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

class SegmentationHead_FGE_SSCD(SegmentationHead_baseline):
    """
    FGE + FBC + SSCD
    """
    def __init__(self):
        super().__init__()
        self.cd = SSCDModule([2, 2, 2])
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
        logit = self.cd(corr[::-1], query_mask)
        return logit

    pass

class SegmentationHead_FGE_SSCDv1(SegmentationHead_baseline):
    """
    FGE + FBC + SSCD
    """
    def __init__(self):
        super().__init__()
        self.cd = SSCDModule_v1([2, 2, 2])
        pass
    
    def forward(self, query_feats, support_feats, support_label, query_mask, support_masks):
        # print("querymask",query_mask)
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
        # print("querymask1",query_mask)
        logit = self.cd(corr[::-1], query_mask)
        return logit

    pass

class SegmentationHead_FGE_SSCDv2(SegmentationHead_baseline):
    """
    FGE + FBC + SSCD
    """
    def __init__(self):
        super().__init__()
        self.cd = SSCDModule_v2([2, 2, 2])
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
        logit = self.cd(corr[::-1], query_mask)
        return logit
    
class SegmentationHead_FGE_SSCDv3(SegmentationHead_baseline):
    """
    FGE + FBC + SSCD
    """
    def __init__(self):
        super().__init__()
        self.cd = SSCDModule_v3([2, 2, 2])
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
        logit = self.cd(corr[::-1], query_mask)
        return logit
class SegmentationHead_FGE_MCD(SegmentationHead_baseline):
    """
    FGE + FBC + MCD
    """
    def __init__(self):
        super().__init__()
        self.cd = MCDModule([2, 2, 2])
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
        logit = self.cd(corr[::-1], query_mask)
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
    
class FGE_SSCD_Network(myNetwork):
    
    def __init__(self, args):
        super().__init__(args)
        self.segmentation_head = SegmentationHead_FGE_SSCD()  # Reuse SegmentationHead for MGSANet
        pass
    
class FGE_SSCDv1_Network(myNetwork):
    
    def __init__(self, args):
        super().__init__(args)
        self.segmentation_head = SegmentationHead_FGE_SSCDv1()  # Reuse SegmentationHead for MGSANet
        pass
    
class FGE_SSCDv2_Network(myNetwork):
    
    def __init__(self, args):
        super().__init__(args)
        self.segmentation_head = SegmentationHead_FGE_SSCDv2()  # Reuse SegmentationHead for MGSANet
        pass
    
class FGE_SSCDv3_Network(myNetwork):
    
    def __init__(self, args):
        super().__init__(args)
        self.segmentation_head = SegmentationHead_FGE_SSCDv3()  # Reuse SegmentationHead for MGSANet
        pass
    
class FGE_MCD_Network(
    myNetwork):
    
    def __init__(self, args):
        super().__init__(args)
        self.segmentation_head = SegmentationHead_FGE_MCD()  # Reuse SegmentationHead for MGSANet
        pass
    pass


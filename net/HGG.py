import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet
from torchvision.models import vgg
from typing_extensions import override
import PIL.Image as Image
import torchvision.transforms.v2 as transforms2
from torch.utils.data import Dataset
from .modules import CenterPivotConv4d, HyperGraphBuilder 
from .net_tools import  SegmentationHead, MGCDModule, MGFEModule, MGCLNetwork

class HyperGraphGenerator(nn.Module):
    """
    HyperGraphGenerator is a module that generates hypergraphs from input features.
    It uses CenterPivotConv4d for 4D convolution operations.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(HyperGraphGenerator, self).__init__()
        self.conv = CenterPivotConv4d(in_channels, out_channels, kernel_size, stride, padding)

    def forward(self, x):
        return self.conv(x)
    
    
class SegmentationHead_HGG(SegmentationHead):

    @override
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
        corr = [torch.concatenate([fg_one[:, None], bg_one[:, None]],
                                  dim=1) for fg_one, bg_one in zip(corr_fg, corr_bg)]

        # MGCD
        logit = self.mgcd(corr[::-1], query_mask)
        return logit

class MGCL_HGGNetwork(MGCLNetwork):
    """
    MGANet is a subclass of MGCLNetwork that implements the Multi-Granularity Attention Network.
    It uses the same backbone and segmentation head but focuses on multi-granularity attention mechanisms.
    """
    def __init__(self, args):
        super().__init__(args)
        self.segmentation_head = SegmentationHead_HGG()  # Reuse SegmentationHead for MGANet
        pass
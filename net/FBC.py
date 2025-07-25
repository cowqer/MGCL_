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


class SegmentationHead_FBC(nn.Module):

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

        feat = support_feats[2]  # shape: [16, 2048, 8, 8]

        proto_fg = masked_avg_pool(feat, label)            # 前景 prototype: [16, 2048]
        proto_bg = masked_avg_pool(feat, 1 - label)        # 背景 prototype: [16, 2048]

        support_prototypes_fg = proto_fg
        support_prototypes_bg = proto_bg

        query_feat = query_feats[2]  # 对应 query 的第三个尺度特征: [16, 2048, 8, 8]

        prior_fg, prior_bg = compute_query_prior(query_feat, support_prototypes_fg, support_prototypes_bg)
        prior = torch.sigmoid(prior_fg - prior_bg)
        
        # w_fg = torch.sigmoid(prior_fg)
        # w_bg = torch.sigmoid(prior_bg)
        # prior = w_fg / (w_fg + w_bg + 1e-8)
        
        query_prototypes_fg = get_query_foreground_prototype(query_feat, prior)  # [B, C]
        prototype_fg = 0.5 * query_prototypes_fg + 0.5 * support_prototypes_fg
        # print("prototype_fg", prototype_fg.shape)##([16, 4096])

        prototype_fg = prototype_fg.unsqueeze(-1).unsqueeze(-1)   # [16, 4096, 1, 1]
        prototype_fg = prototype_fg.expand(-1, -1, 8, 8)    
         
        support_feats[2]= support_feats[2] + prototype_fg
        
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

    @staticmethod
    def label_feature(feature, label):
        label = F.interpolate(label.unsqueeze(1).float(), feature.size()[2:],
                             mode='bilinear', align_corners=True)
        return feature * label

    @staticmethod
    def multilayer_correlation(query_feats, support_feats):
        corrs = []
        for idx, (query_feat, support_feat) in enumerate(zip(query_feats, support_feats)):
            bsz, ch, hb, wb = support_feat.size()
            support_feat = support_feat.view(bsz, ch, -1)
            support_feat = support_feat / (support_feat.norm(dim=1, p=2, keepdim=True) + 1e-5)

            bsz, ch, ha, wa = query_feat.size()
            query_feat = query_feat.view(bsz, ch, -1)
            query_feat = query_feat / (query_feat.norm(dim=1, p=2, keepdim=True) + 1e-5)

            corr = torch.bmm(query_feat.transpose(1, 2), support_feat).view(bsz, ha, wa, hb, wb)
            corr = corr.clamp(min=0)
            corrs.append(corr)
            pass

        return corrs

    pass


    @staticmethod
    def my_masked_average_pooling(feature, mask):
        b, c, w, h = feature.shape
        _, m, _, _ = mask.shape

        _mask = mask.view(b, m, -1)
        _feature = feature.view(b, c, -1).permute(0, 2, 1).contiguous()
        feature_sum = _mask @ _feature
        masked_sum = torch.sum(_mask, dim=2, keepdim=True)

        masked_average_pooling = torch.div(feature_sum, masked_sum + 1e-8)
        return masked_average_pooling

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
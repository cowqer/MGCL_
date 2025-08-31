import torch
import torch.nn as nn
import torch.nn.functional as F
from .modules import *
import cv2
from .net_tools_pro import  myNetwork, SegmentationHead_baseline
# SegmentationHead_baseline = FBC_origin + CD

class SegmentationHead_sam_test_v1(SegmentationHead_baseline):
    """
    引入sobel算子 得到基于sam mask 的边缘信息，将sobel结果与原始mask拼接后作为辅助信息输入到segmentation head中
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


class samv1_Network(myNetwork):
    """
    FGE + FBC + CD + Sobel
    """
    def __init__(self, args):
        super().__init__(args)
        self.segmentation_head = SegmentationHead_sam_test_v1()  # Reuse SegmentationHead for MGSANet
        
        self.alpha_mask = nn.Parameter(torch.tensor(1.0))
        self.alpha_sobel_mask = nn.Parameter(torch.tensor(1.0))
        self.alpha_sobel_img = nn.Parameter(torch.tensor(1.0))
        sobel_x = torch.tensor([[1, 0, -1],
                        [2, 0, -2],
                        [1, 0, -1]], dtype=torch.float32)
        sobel_y = torch.tensor([[1, 2, 1],
                                [0, 0, 0],
                                [-1,-2,-1]], dtype=torch.float32)

        # 转成 conv2d 权重 [out_channels=1, in_channels=1, kH=3, kW=3]
        self.register_buffer('sobel_x', sobel_x.view(1, 1, 3, 3))
        self.register_buffer('sobel_y', sobel_y.view(1, 1, 3, 3))

        pass
    
    def compute_sobel(self, x):
        """
        x: [B,C,H,W]  mask 或图像
        """
        # 确保是 float
        if not x.is_floating_point():
            x = x.float()

        B, C, H, W = x.shape
        if C > 1:  # 多通道图像
            x_gray = x.mean(dim=1, keepdim=True)
        else:      # 单通道 mask
            x_gray = x

        gx = F.conv2d(x_gray, self.sobel_x, padding=1)
        gy = F.conv2d(x_gray, self.sobel_y, padding=1)
        grad = torch.sqrt(gx**2 + gy**2 + 1e-6)
        return grad

    
    def forward(self, query_img, support_img, support_label, query_mask, support_masks):
        if self.finetune_backbone:
            #* size of mask: ([16, 128, 256, 256])            
            # mask 均值先验
            mask_query_prior = torch.mean(query_mask.float(), dim=1, keepdim=True)
            mask_support_prior = torch.mean(support_masks.float(), dim=1, keepdim=True)

            # # Sobel 特征
            sobel_mask_query = self.compute_sobel(query_mask)#([16, 1, 256, 256])
            sobel_mask_support = self.compute_sobel(support_masks)#([16, 1, 256, 256])
            
            sobel_query_img = self.compute_sobel(query_img)#([16, 1, 256, 256])
            sobel_support_img = self.compute_sobel(support_img)#([16, 1, 256, 256])


            # 加权叠加
            query_img = query_img \
                            + self.alpha_mask * mask_query_prior.repeat(1,3,1,1) \
                            + self.alpha_sobel_mask * sobel_mask_query.repeat(1,3,1,1) \
                            + self.alpha_sobel_img * sobel_query_img.repeat(1,3,1,1)

            support_img = support_img \
                              + self.alpha_mask * mask_support_prior.repeat(1,3,1,1) \
                              + self.alpha_sobel_mask * sobel_mask_support.repeat(1,3,1,1) \
                              + self.alpha_sobel_img * sobel_support_img.repeat(1,3,1,1)
            
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
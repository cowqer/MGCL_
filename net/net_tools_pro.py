import torch
import torch.nn as nn
import torch.nn.functional as F
from .modules import CenterPivotConv4d,SSblock
import math
from torchvision.models import resnet
from torchvision.models import vgg

import torch
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

class MGFEModule_v1(object):
    """Mask Guided Feature Enhancement + 可选择 Mask"""

    @classmethod
    def update_feature_one(cls, query_feat, query_mask, mask_score=None):
        '''单特征增强'''
        return cls.enabled_feature([query_feat], query_mask, mask_score)[0]

    @classmethod
    def update_feature(cls, query_feats, support_feats, query_mask, support_masks, 
                       query_mask_score=None, support_mask_score=None):
        '''批量特征增强'''
        query_feats = cls.enabled_feature(query_feats, query_mask, query_mask_score)
        support_feats = cls.enabled_feature(support_feats, support_masks, support_mask_score)
        return query_feats, support_feats

    @classmethod
    def enabled_feature(cls, feats, masks, mask_score=None, threshold=0.2):
        """
        feats: list of [B,C,H,W]
        masks: [B,M,H,W] 二值 mask
        mask_score: [B,M] 每个 mask 的置信/重要性分数，可选
        threshold: 选择阈值，小于阈值 mask 被抑制
        """
        b, m, h, w = masks.shape
        # 每个像素只属于一个区域
        index_mask = torch.full_like(masks[:, 0], m, dtype=torch.long)
        for i in range(m):
            index_mask[masks[:, i] == 1] = i
        masks_onehot = F.one_hot(index_mask, num_classes=m+1)[..., :m].permute(0,3,1,2).contiguous().float()

        # mask 选择：根据 mask_score 或均匀 threshold
        if mask_score is not None:
            score_mask = (mask_score > threshold).float().view(b, m, 1, 1)
            masks_onehot = masks_onehot * score_mask
        else:
            # 按 mask 覆盖比例过滤：mask 占比 < threshold 则抑制
            mask_area = masks_onehot.view(b, m, -1).sum(-1) / (h*w)  # [B,M]
            score_mask = (mask_area > threshold).float().view(b, m, 1, 1)
            masks_onehot = masks_onehot * score_mask

        enabled_feats = []
        for feat in feats:
            b, c, h_feat, w_feat = feat.shape
            target_masks = F.interpolate(masks_onehot, (h_feat, w_feat), mode='nearest')
            
            # 区域平均特征
            map_features = cls.my_masked_average_pooling(feat, target_masks)

            # 投影回像素
            _map_features = map_features.permute(0,2,1).contiguous()  # [B,C,M]
            feature_sum = torch.bmm(_map_features, target_masks.view(b, m, -1))
            feature_sum = feature_sum.view(b, c, h_feat, w_feat)

            sum_mask = target_masks.sum(dim=1, keepdim=True)
            enabled_feat = feature_sum / (sum_mask + 1e-8)
            enabled_feats.append(enabled_feat)
        
        return enabled_feats

    @staticmethod
    def my_masked_average_pooling(feature, mask):
        b, c, h, w = feature.shape
        _, m, _, _ = mask.shape
        _mask = mask.view(b, m, -1)
        _feature = feature.view(b, c, -1).permute(0, 2, 1).contiguous()
        feature_sum = torch.bmm(_mask, _feature)
        masked_sum = _mask.sum(dim=2, keepdim=True)
        return feature_sum / (masked_sum + 1e-8)


class SobelConv(nn.Module):
    """通用 Sobel 卷积"""
    def __init__(self, in_channels, kernel_size=3):
        super().__init__()
        assert kernel_size in [3,5]
        if kernel_size==3:
            sobel_x = torch.tensor([[-1.,0.,1.],[-2.,0.,2.],[-1.,0.,1.]])
        else:
            sobel_x = torch.tensor([[-5,-4,0,4,5],
                                    [-8,-10,0,10,8],
                                    [-10,-20,0,20,10],
                                    [-8,-10,0,10,8],
                                    [-5,-4,0,4,5]])
        sobel_y = sobel_x.T
        kernel = torch.stack([sobel_x,sobel_y]).unsqueeze(1).repeat(in_channels,1,1,1)
        self.conv = nn.Conv2d(in_channels, 2*in_channels, kernel_size=kernel_size, 
                              stride=1, padding=kernel_size//2, bias=False, groups=in_channels)
        self.conv.weight.data = kernel
        self.conv.weight.requires_grad = False
    def forward(self,x):
        return self.conv(x)

class SSblock(nn.Module):
    """Sobel + Attention 模块"""
    def __init__(self, dim):
        super().__init__()
        self.sobel3 = SobelConv(dim,3)
        self.sobel5 = SobelConv(dim,5)
        self.conv3 = nn.Conv2d(2*dim, dim//2,1)
        self.conv5 = nn.Conv2d(2*dim, dim//2,1)
        self.conv_squeeze = nn.Conv2d(2,2,7,padding=3)
        self.conv_fuse = nn.Conv2d(dim//2, dim,1)
    def forward(self,x):
        feat3 = self.conv3(self.sobel3(x))
        feat5 = self.conv5(self.sobel5(x))
        attn_cat = torch.cat([feat3,feat5],dim=1)
        avg_attn = torch.mean(attn_cat,dim=1,keepdim=True)
        max_attn,_ = torch.max(attn_cat,dim=1,keepdim=True)
        agg = torch.cat([avg_attn,max_attn],dim=1)
        sig = self.conv_squeeze(agg).sigmoid()
        fused = feat3*sig[:,0:1,:,:] + feat5*sig[:,1:2,:,:]
        fused = self.conv_fuse(fused)
        return fused  # 注意：这里只输出增强特征，不做残差

class MGFEModule_v2(object):
    """Mask Guided Feature Enhancement + Sobel Attention"""
    SSblock_module = None  # 可在外部初始化一次，复用
    @classmethod
    def update_feature_one(cls, query_feat, query_mask):
        return cls.enabled_feature([query_feat], query_mask)[0]

    @classmethod
    def update_feature(cls, query_feats, support_feats, query_mask, support_masks):
        query_feats = cls.enabled_feature(query_feats, query_mask)
        support_feats = cls.enabled_feature(support_feats, support_masks)
        return query_feats, support_feats
    @classmethod
    def set_sobel_block(cls, dim):
        cls.SSblock_module = SSblock(dim).cuda() if torch.cuda.is_available() else SSblock(dim)

    @classmethod
    def enabled_feature(cls, feats, masks):
        b, m, h_mask, w_mask = masks.shape
        index_mask = torch.argmax(masks, dim=1)
        masks_onehot = F.one_hot(index_mask, num_classes=m)[..., :m].permute(0,3,1,2).contiguous() # [B,M,Hm,Wm]

        enabled_feats = []
        for feat in feats:
            b, c, h_feat, w_feat = feat.shape
            target_masks = F.interpolate(masks_onehot.float(), (h_feat,w_feat), mode='nearest')
            # 区域平均特征
            map_features = cls.my_masked_average_pooling(feat,target_masks)
            # 投影回像素
            _map_features = map_features.permute(0,2,1).contiguous()  # [B,C,M]
            feature_sum = torch.bmm(_map_features, target_masks.view(b,m,-1)).view(b,c,h_feat,w_feat)
            sum_mask = target_masks.sum(dim=1,keepdim=True)
            mask_feat = feature_sum / (sum_mask+1e-8)

            # ---- Sobel Attention branch ----
            if cls.SSblock_module is not None:
                sobel_feat = cls.SSblock_module(feat)
                # 插值到和 mask_feat 一致
                if sobel_feat.shape[2:] != mask_feat.shape[2:]:
                    sobel_feat = F.interpolate(sobel_feat, mask_feat.shape[2:], mode='bilinear', align_corners=False)
                # 融合
                alpha = 0.3  # Sobel 权重，可调/可学习
                mask_feat = mask_feat + alpha * sobel_feat

            enabled_feats.append(mask_feat)
        return enabled_feats

    @staticmethod
    def my_masked_average_pooling(feature, mask):
        b,c,h,w = feature.shape
        _,m,_,_ = mask.shape
        _mask = mask.view(b,m,-1)
        _feature = feature.view(b,c,-1).permute(0,2,1).contiguous()
        feature_sum = torch.bmm(_mask,_feature)
        masked_sum = _mask.sum(dim=2,keepdim=True)
        return feature_sum/(masked_sum+1e-8)

class MGFEModule_v3(object):
    """
    Enhanced Mask-Guided Feature Enrichment (v3)

    Key ideas:
    - Keep masked-average pooling prototype extraction (backward compatible)
    - Multi-scale aggregation across feat levels
    - Prototype <-> pixel cross-attention (lightweight Transformer-like)
    - Channel attention (Squeeze-and-Excite) and spatial gating derived from prototypes
    - Residual connection + learnable gates initialized to zero (to avoid hurting baseline)
    - All ops are batch-friendly and support list of feature maps (multi-level)
    """

    @classmethod
    def update_feature_one(cls, query_feat, query_mask, **kwargs):
        return cls.enabled_feature([query_feat], query_mask, **kwargs)[0]

    @classmethod
    def update_feature(cls, query_feats, support_feats, query_mask, support_masks, **kwargs):
        q = cls.enabled_feature(query_feats, query_mask, **kwargs)
        s = cls.enabled_feature(support_feats, support_masks, **kwargs)
        return q, s

    @classmethod
    def enabled_feature(cls, feats, masks, 
                        prototype_dim=None,
                        use_cross_attn=True,
                        n_attn_heads=4,
                        attn_hidden=128,
                        use_channel_attn=True,
                        multi_scale_pool=True,
                        gate_learnable=True):
        """
        feats: list of tensors [B, C, H, W]
        masks: [B, M, Hm, Wm]  (binary masks per region)
        Returns: list of enhanced features (same shapes as feats)
        """

        device = feats[0].device
        b, m, h_mask, w_mask = masks.shape

        # --- step0: compute one-hot region assignment as in original ---
        index_mask = torch.full_like(masks[:, 0], m, dtype=torch.long, device=device)
        for i in range(m):
            index_mask = torch.where(masks[:, i] == 1, torch.full_like(index_mask, i), index_mask)
        masks_onehot = F.one_hot(index_mask, num_classes=m+1)[..., :m]   # [B, Hm, Wm, M]
        masks_onehot = masks_onehot.permute(0, 3, 1, 2).contiguous().float()  # [B, M, Hm, Wm]

        # Prepare per-level outputs
        enhanced_feats = []

        # Shared small MLPs / transform for prototypes & cross-attn
        # we create lightweight modulators in-module to avoid global state
        # prototype_dim default = channel of first feature or attn_hidden
        C0 = feats[0].shape[1]
        prototype_dim = prototype_dim or min(attn_hidden, C0)
        # create small projection layers (bias-free for stability)
        proj_proto = nn.Linear(C0, prototype_dim).to(device)
        proj_query = nn.Linear(C0, prototype_dim).to(device)
        proto_to_channel = nn.Linear(prototype_dim, C0).to(device)
        # LayerNorms for stable training
        ln_proto = nn.LayerNorm(prototype_dim).to(device)
        ln_query = nn.LayerNorm(prototype_dim).to(device)

        # Attention parameters (lightweight multi-head attention implemented manually)
        # We'll implement a small cross-attention: prototypes (M) attend to pixel tokens (H*W)
        # For efficiency we operate on reduced spatial resolution if needed.
        def lightweight_cross_attn(proto_feats, query_tokens, num_heads=n_attn_heads):
            # proto_feats: [B, M, D], query_tokens: [B, HW, D]
            B, M, D = proto_feats.shape
            _, HW, _ = query_tokens.shape
            head_dim = D // num_heads
            if head_dim == 0:
                # fallback: single-head
                q = proto_feats.unsqueeze(2)  # [B, M, 1, D]
                k = query_tokens.unsqueeze(1) # [B, 1, HW, D]
                attn = torch.einsum('bmnd,bkhd->bmnk', q, k)  # expensive shape fallback
                # but we avoid this by forcing D multiple of heads externally
            # project to queries/keys/values by small linear maps (use einsum-friendly)
            # For speed use linear layers on last dim (we have small dims)
            Wq = nn.Linear(D, D).to(device)
            Wk = nn.Linear(D, D).to(device)
            Wv = nn.Linear(D, D).to(device)
            q = Wq(proto_feats)  # [B, M, D]
            k = Wk(query_tokens) # [B, HW, D]
            v = Wv(query_tokens) # [B, HW, D]
            # reshape for heads
            qh = q.view(B, M, num_heads, head_dim).permute(0,2,1,3)  # [B, H, M, Hd]
            kh = k.view(B, HW, num_heads, head_dim).permute(0,2,1,3) # [B, H, HW, Hd]
            vh = v.view(B, HW, num_heads, head_dim).permute(0,2,1,3) # [B, H, HW, Hd]
            attn_scores = torch.matmul(qh, kh.transpose(-2, -1))  # [B, H, M, HW]
            attn_scores = attn_scores / math.sqrt(head_dim + 1e-8)
            attn_prob = F.softmax(attn_scores, dim=-1)  # softmax over HW
            outh = torch.matmul(attn_prob, vh)  # [B, H, M, Hd]
            out = outh.permute(0,2,1,3).contiguous().view(B, M, D)  # [B, M, D]
            return out

        # Channel attention (Squeeze-and-Excite style) helper
        def channel_se(pool_proto, num_channels):
            # pool_proto: [B, M, D_proto] -> produce [B, C] channel scaling
            # Use a small MLP
            x = pool_proto.mean(dim=1)  # [B, D_proto], average over M regions
            fc1 = nn.Linear(x.shape[-1], max(8, x.shape[-1]//4)).to(device)
            fc2 = nn.Linear(fc1.out_features, num_channels).to(device)
            x = F.relu(fc1(x))
            x = torch.sigmoid(fc2(x))  # [B, C]
            return x.view(-1, num_channels, 1, 1)

        # For each feature level
        for feat in feats:
            B, C, H, W = feat.shape
            # step1: resize masks to feature size (nearest)
            target_masks = F.interpolate(masks_onehot, size=(H, W), mode='nearest')  # [B, M, H, W]

            # step2: multi-scale pooling (optional):
            # we compute prototypes at original scale and optionally at pooled scales
            prototypes = cls.my_masked_average_pooling(feat, target_masks)  # [B, M, C]
            if multi_scale_pool:
                # small pyramid: 1x1, 3x3 avgpool then masked pooling on downsampled features
                # average-pool feature and masks then masked-average-pooling
                feat_down = F.adaptive_avg_pool2d(feat, (H//2 if H//2>0 else 1, W//2 if W//2>0 else 1))
                mask_down = F.adaptive_avg_pool2d(target_masks, (feat_down.shape[2], feat_down.shape[3]))
                # binarize mask_down (threshold 0.5)
                mask_down = (mask_down >= 0.5).float()
                proto_down = cls.my_masked_average_pooling(feat_down, mask_down)  # [B, M, C]
                # upsample proto_down via linear project and concat
                prototypes = (prototypes + F.interpolate(proto_down, size=(prototypes.shape[2],), mode='nearest'))/2.0

            # prototypes: [B, M, C]
            # project prototypes to proto space and queries to proto space
            p = prototypes  # [B, M, C]
            p_proj = ln_proto(proj_proto(p)) if hasattr(proj_proto, 'weight') else proj_proto(p)  # [B, M, D]
            # pixel tokens: flatten spatial dims
            pixel_tokens = feat.view(B, C, H*W).permute(0,2,1).contiguous()  # [B, HW, C]
            q_proj = ln_query(proj_query(pixel_tokens))  # [B, HW, D]

            # cross-attention to enrich prototypes from query pixels (proto <- attn(pixel))
            if use_cross_attn:
                # Use lightweight cross-attention (proto attends to pixels)
                proto_attn_enh = lightweight_cross_attn(p_proj, q_proj)  # [B, M, D]
                # fuse: residual + small MLP
                fuse_mlp = nn.Sequential(
                    nn.Linear(proto_attn_enh.shape[-1], proto_attn_enh.shape[-1]).to(device),
                    nn.ReLU(),
                    nn.Linear(proto_attn_enh.shape[-1], proto_attn_enh.shape[-1]).to(device)
                )
                p_fused = p_proj + fuse_mlp(proto_attn_enh)  # [B, M, D]
            else:
                p_fused = p_proj

            # map fused prototypes back to pixel space:
            # Reconstruct feature per pixel as weighted sum of prototypes using target_masks
            # First project prototypes back to C-channel (proto_to_channel)
            proto_back = proto_to_channel(p_fused)  # [B, M, C]
            # weight-sum via masks: proto_back (B, M, C) * mask (B, M, H, W) -> sum over M
            # expand dims to multiply
            proto_back = proto_back.permute(0,2,1).contiguous()  # [B, C, M]
            mask_flat = target_masks.view(B, m, H*W)  # [B, M, HW]
            # feature_sum = proto_back @ mask_flat  -> [B, C, HW]
            feature_sum = torch.bmm(proto_back, mask_flat)  # [B, C, HW]
            feature_sum = feature_sum.view(B, C, H, W)

            # normalize by mask coverage
            sum_mask = target_masks.sum(dim=1, keepdim=True)  # [B,1,H,W]
            enhanced = feature_sum / (sum_mask + 1e-8)  # [B, C, H, W]

            # channel attention from prototypes
            if use_channel_attn:
                ch_scale = channel_se(p_fused, C)  # [B, C, 1, 1]
                enhanced = enhanced * ch_scale

            # residual connection + learnable gate: y = x + g * enhanced
            if gate_learnable:
                # gate per-channel for stability. Initialize to zeros so early training ~ identity.
                gate = nn.Parameter(torch.zeros(1, C, 1, 1)).to(device)
                # apply
                out = feat + gate * enhanced
            else:
                out = feat + enhanced

            # optional normalization to keep statistics stable
            out = F.layer_norm(out, out.shape[1:])  # layer-norm across (C,H,W) per sample

            enhanced_feats.append(out)

        return enhanced_feats

    @staticmethod
    def my_masked_average_pooling(feature, mask):
        """
        feature: [B, C, H, W]
        mask: [B, M, H, W] (binary or soft)
        returns: [B, M, C]
        """
        b, c, h, w = feature.shape
        _, m, _, _ = mask.shape
        _mask = mask.view(b, m, -1)  # [B, M, H*W]
        _feature = feature.view(b, c, -1).permute(0, 2, 1).contiguous()  # [B, H*W, C]
        # masked sum: (B, M, H*W) @ (B, H*W, C) -> (B, M, C)
        feature_sum = torch.bmm(_mask, _feature)  # [B, M, C]
        masked_sum = torch.sum(_mask, dim=2, keepdim=True)  # [B, M, 1]
        masked_avg = feature_sum / (masked_sum + 1e-8)
        return masked_avg

class SegmentationHead_baseline(nn.Module):
    """
    SegmentationHead_FBC_MGCL is a subclass of SegmentationHead_FBC that implements the Multi-Granularity Attention Network.
    It uses the same backbone and segmentation head but focuses on multi-granularity attention mechanisms.
    """
    def __init__(self):
        super().__init__()
        self.cd = CDModule([2, 2, 2])  # Initialize CDModule with specific parameters
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

        # CD Correlation Decoder
        logit = self.cd(corr[::-1])
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

class CDModule(nn.Module):
    """
    CD (Correlation Decoder)：
    - 输入：来自多层(FPN式)的 4D 相关张量金字塔 hypercorr_pyramid，形状依次为 [B, C, Ha, Wa, Hb, Wb]
      其中 (Ha, Wa) 是 Query 空间，(Hb, Wb) 是 Support 空间。
    - 作用：
        1) Squeezing：对每层 4D 相关张量做多层 4D 卷积降维/提取表征；
        2) Mixing：把高层 4D 表征对齐到低层 Query 尺度，逐级融合；
        3) Reduce 4D -> 2D：把 support 维度 (Hb, Wb) 做统计（mean），得到 2D 的 Query 特征；
        4) Decoder：2D 卷积 + 上采样，输出二分类的语义分割 logit。
    - 输出：logit，形状为 [B, 2, H, W]（最终分割的前景/背景两通道）。
    """

    def __init__(self, inch):
        """
        参数
        ----
        inch : list[int]
            与输入金字塔各层通道数对应的列表（例如 [2, 2, 2]），即每层 4D 相关张量的 C。
        """
        super().__init__()

        def make_building_block(in_channel, out_channels, kernel_sizes, spt_strides, group=4):
            """
            构造由若干个 4D 卷积 + GroupNorm + ReLU 组成的子序列。
            - in_channel: 输入通道 C_in（4D 卷积的输入通道）
            - out_channels: List，每个阶段的输出通道数
            - kernel_sizes: List，每个阶段 4D 卷积的空间核大小（这里对 4 个维度共用同一个标量）
            - spt_strides: List，每个阶段在“Query 空间维 (Ha, Wa)”上的 stride（前两个 stride 固定为1）
            - group: GroupNorm 的分组数（要求 out_channel 能被 group 整除）
            
            形状约定（CenterPivotConv4d）：
            输入: [B, C_in, Ha, Wa, Hb, Wb]
            卷积核: (k_a, k_a, k_b, k_b)  # 这里展开为 ksz4d=(ksz, ksz, ksz, ksz)
            stride: (1, 1, stride, stride) # 只在 Query 空间(Ha, Wa)做下采样；Support 空间保持
            padding: (ksz//2, ksz//2, ksz//2, ksz//2) # 保持两空间“同长对齐”的填充
            输出: [B, C_out, Ha', Wa', Hb, Wb]
            """
            assert len(out_channels) == len(kernel_sizes) == len(spt_strides)

            building_block_layers = []
            for idx, (outch, ksz, stride) in enumerate(zip(out_channels, kernel_sizes, spt_strides)):
                inch = in_channel if idx == 0 else out_channels[idx - 1]
                ksz4d = (ksz,) * 4
                str4d = (1, 1) + (stride,) * 2
                pad4d = (ksz // 2,) * 4

                building_block_layers.append(CenterPivotConv4d(inch, outch, ksz4d, str4d, pad4d))
                building_block_layers.append(nn.GroupNorm(group, outch))
                building_block_layers.append(nn.ReLU(inplace=True))

            return nn.Sequential(*building_block_layers)

        outch1, outch2, outch3, outch4 = 16, 32, 64, 128

        # Squeezing building blocks
        self.encoder_layer4 = make_building_block(inch[0], [outch1, outch2, outch3], [3, 3, 3], [2, 2, 2])
        self.encoder_layer3 = make_building_block(inch[1], [outch1, outch2, outch3], [5, 3, 3], [4, 2, 2])
        self.encoder_layer2 = make_building_block(inch[2], [outch1, outch2, outch3], [5, 5, 3], [4, 4, 2])

        # Mixing building blocks
        self.encoder_layer4to3 = make_building_block(outch3, [outch3, outch3, outch3], [3, 3, 3], [1, 1, 1])
        self.encoder_layer3to2 = make_building_block(outch3, [outch3, outch3, outch3], [3, 3, 3], [1, 1, 1])

        # Decoder layers
        self.decoder1 = nn.Sequential(
            nn.Conv2d(outch4, outch3, (3, 3), padding=(1, 1), bias=True), nn.ReLU(),
            nn.Conv2d(outch3, outch2, (3, 3), padding=(1, 1), bias=True), nn.ReLU())
        #! 别删 之前训练的有部分初始化了这段 评估报错就取消这个注释
        self.decoder1_my = nn.Sequential(
            nn.Conv2d(64, 32, (3, 3), padding=(1, 1), bias=True), nn.ReLU(),
            nn.Conv2d(32, 32, (3, 3), padding=(1, 1), bias=True), nn.ReLU())
        
        self.decoder2 = nn.Sequential(
            nn.Conv2d(outch2, outch1, (3, 3), padding=(1, 1), bias=True), nn.ReLU(),
            nn.Conv2d(outch1, 2, (3, 3), padding=(1, 1), bias=True))
        
        self.SSblock = None
        
        pass
    
    def interpolate_support_dims(self, hypercorr, spatial_size=None):
        bsz, ch, ha, wa, hb, wb = hypercorr.size()
        hypercorr = hypercorr.permute(0, 4, 5, 1, 2, 3).contiguous().view(bsz * hb * wb, ch, ha, wa)
        hypercorr = F.interpolate(hypercorr, spatial_size, mode='bilinear', align_corners=True)
        o_hb, o_wb = spatial_size
        hypercorr = hypercorr.view(bsz, hb, wb, ch, o_hb, o_wb).permute(0, 3, 4, 5, 1, 2).contiguous()
        return hypercorr

    def forward(self, hypercorr_pyramid, query_mask=None):
        # Encode hypercorrelations from each layer (Squeezing building blocks)
        hypercorr_sqz4 = self.encoder_layer4(hypercorr_pyramid[0])
        hypercorr_sqz3 = self.encoder_layer3(hypercorr_pyramid[1])
        hypercorr_sqz2 = self.encoder_layer2(hypercorr_pyramid[2])

        # Propagate encoded 4D-tensor (Mixing building blocks)
        hypercorr_sqz4 = self.interpolate_support_dims(hypercorr_sqz4, hypercorr_sqz3.size()[-4:-2])
        hypercorr_mix43 = hypercorr_sqz4 + hypercorr_sqz3
        hypercorr_mix43 = self.encoder_layer4to3(hypercorr_mix43)

        hypercorr_mix43 = self.interpolate_support_dims(hypercorr_mix43, hypercorr_sqz2.size()[-4:-2])
        hypercorr_mix432 = hypercorr_mix43 + hypercorr_sqz2
        hypercorr_mix432 = self.encoder_layer3to2(hypercorr_mix432)

        bsz, ch, ha, wa, hb, wb = hypercorr_mix432.size()
        hypercorr_encoded = hypercorr_mix432.view(bsz, ch, ha, wa, -1).mean(dim=-1)
        #
            
        if query_mask is not None:
            hypercorr_encoded = torch.concat([hypercorr_encoded, hypercorr_encoded], dim=1)
        else:
            print("query_mask",query_mask)
            hypercorr_encoded = torch.concat([hypercorr_encoded, hypercorr_encoded], dim=1)
            pass
        #! SSblock
        hypercorr_encoded = self.SSblock(hypercorr_encoded) if self.SSblock is not None else hypercorr_encoded

        hypercorr_decoded = self.decoder1(hypercorr_encoded)
        upsample_size = (hypercorr_decoded.size(-1) * 2,) * 2
        hypercorr_decoded = F.interpolate(hypercorr_decoded, upsample_size,
                                          mode='bilinear', align_corners=True)
        logit = self.decoder2(hypercorr_decoded)
        return logit

    pass

class myNetwork(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.backbone_type = args.backbone
        self.finetune_backbone = args.finetune_backbone if hasattr(args, "finetune_backbone") else False

        if "vgg" in self.backbone_type:
            self.backbone = vgg.vgg16(pretrained=True)
            self.extract_feats = self.extract_feats_vgg
        elif "50" in self.backbone_type:
            self.backbone = resnet.resnet50(pretrained=True)
            # self.backbone.conv1.in_channels = 4
            self.extract_feats = self.extract_feats_res
        else:
            self.backbone = resnet.resnet101(pretrained=True)
            self.extract_feats = self.extract_feats_res
            pass

        self.segmentation_head = SegmentationHead_baseline()
        self.cross_entropy_loss = nn.CrossEntropyLoss()
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
        # Modules forward
        logit = self.segmentation_head(query_feats, support_feats, support_label.clone(), query_mask, support_masks)
        logit = F.interpolate(logit, support_img.size()[2:], mode='bilinear', align_corners=True)
        return logit

    def predict_nshot(self, batch):
        nshot = batch["support_imgs"].shape[1]
        logit_label_agg = 0
        for s_idx in range(nshot):
            logit_label = self.forward(
                batch['query_img'], batch['support_imgs'][:, s_idx],  batch['support_labels'][:, s_idx],
                query_mask=batch['query_mask'] if 'query_mask' in batch and self.args.mask else None,
                support_masks=batch['support_masks'][:, s_idx] if 'support_masks' in batch and self.args.mask else None)

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

    def compute_objective(self, logit_label, gt_label):
        bsz = logit_label.size(0)
        logit_label = logit_label.view(bsz, 2, -1)
        gt_label = gt_label.view(bsz, -1).long()

        return self.cross_entropy_loss(logit_label, gt_label)

    def train_mode(self):
        self.train()
        self.backbone.eval()  # to prevent BN from learning data statistics with exponential averaging
        pass

    @staticmethod
    def extract_feats_vgg(img, backbone):
        feat_ids = [16, 23, 30]
        feats = []
        feat = img
        for lid, module in enumerate(backbone.features):
            feat = module(feat)
            if lid in feat_ids:
                feats.append(feat.clone())
        return feats

    @staticmethod
    def extract_feats_res(img, backbone):
        x = backbone.maxpool(backbone.relu(backbone.bn1(backbone.conv1(img))))

        feats = []
        x = backbone.layer1(x)
        x = backbone.layer2(x)
        feats.append(x.clone())
        x = backbone.layer3(x)
        feats.append(x.clone())
        x = backbone.layer4(x)
        feats.append(x.clone())
        return feats

    pass

class baseline_Network(myNetwork):
    """
    MGSANet is a subclass of MGCLNetwork that implements the Multi-Granularity Attention Network.
    It uses the same backbone and segmentation head but focuses on multi-granularity attention mechanisms.
    """
    def __init__(self, args):
        super().__init__(args)
        self.segmentation_head = SegmentationHead_baseline()  # Reuse SegmentationHead for MGSANet
        pass

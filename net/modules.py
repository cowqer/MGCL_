import torch
import torch.nn as nn
import torch.nn.functional as F

class MaskSobelFusion(nn.Module):
    def __init__(self, init_alpha=1.0, init_beta=0.5):
        super().__init__()
        # 可学习参数，初始值可调
        
        self.alpha = nn.Parameter(torch.tensor(init_alpha, dtype=torch.float32))
        self.beta  = nn.Parameter(torch.tensor(init_beta, dtype=torch.float32))

    def forward(self, mask):
        """
        mask: [B, N, H, W]
        return: [B,1,H,W] 融合结果
        """
        grad_mag = self.mask_sobel(mask)  # [B,N,H,W]

        # 确保权重为正，可选归一化
        alpha = torch.clamp(self.alpha, 0, 1)
        beta  = torch.clamp(self.beta, 0, 1)

        combined = alpha * mask + beta * grad_mag
        combined_merged = combined.mean(dim=1, keepdim=True)  # 多 mask 合并
        return combined_merged

    @staticmethod
    def mask_sobel(mask):
        sobel_x = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=torch.float32, device=mask.device).view(1,1,3,3)
        sobel_y = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=torch.float32, device=mask.device).view(1,1,3,3)

        B, N, H, W = mask.shape
        mask = mask.view(B*N, 1, H, W)

        grad_x = F.conv2d(mask, sobel_x, padding=1)
        grad_y = F.conv2d(mask, sobel_y, padding=1)

        grad_mag = torch.sqrt(grad_x**2 + grad_y**2)
        grad_mag = grad_mag.view(B, N, H, W)
        return grad_mag
    
def FGE( query_feats, support_feats, support_label, query_mask, alpha=0.5):
    """
    Foreground-Guilded Enhancement (FPG)
    """
    # [B,1,H,W]
    label = support_label.unsqueeze(1)

    # 取最后一层 support/query 特征
    query_feat_2 = query_feats[2]        # [B, C, H, W]
    support_feat_2 = support_feats[2]    # [B, C, H, W]

    # 支持图前景/背景 prototype
    proto_fg = masked_avg_pool(support_feat_2, label)       # [B, C]
    proto_bg = masked_avg_pool(support_feat_2, 1 - label)   # [B, C]

    # Query prior
    prior_fg, prior_bg = compute_query_prior(query_feat_2, proto_fg, proto_bg, temperature=1.1)
    prior = torch.sigmoid(prior_fg - prior_bg)

    # query prototype
    query_proto_fg = get_query_foreground_prototype(query_feat_2, prior)  # [B, C]

    # 融合 support & query prototype
    beta = 1.0 - alpha
    prototype_fg = alpha * query_proto_fg + beta * proto_fg   # [B, C]

    # 映射成 feature map
    B, C, H, W = query_feat_2.shape
    prototype_fg = prototype_fg.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)  # [B, C, H, W]

    # 融合到原始特征
    support_feats[2] = support_feats[2] + prototype_fg
    query_feats[2] = query_feats[2] + prototype_fg

    return query_feats, support_feats


def compute_similarity_alpha(query_feat, support_feat):
    # Global average pooling
    query_vec = F.adaptive_avg_pool2d(query_feat, 1).squeeze(-1).squeeze(-1)  # [B, C]
    support_vec = F.adaptive_avg_pool2d(support_feat, 1).squeeze(-1).squeeze(-1)  # [B, C]

    # Normalize for cosine
    query_norm = F.normalize(query_vec, dim=1)
    support_norm = F.normalize(support_vec, dim=1)

    # Cosine similarity: [B]
    cos_sim = (query_norm * support_norm).sum(dim=1)  # Higher is more similar → closer to 1

    # Euclidean distance: [B]
    l2_dist = (query_vec - support_vec).pow(2).sum(dim=1).sqrt()  # Higher is more dissimilar

    # Normalize l2 distance to [0,1]
    l2_dist = (l2_dist - l2_dist.min()) / (l2_dist.max() - l2_dist.min() + 1e-8)

    # Combine (you can tune weights)
    sim_score = 0.5 * cos_sim + 0.5 * (1 - l2_dist)  # Higher = more similar

    # Convert to alpha: if sim high → alpha near 0.5, if sim low → alpha near 1
    alpha = 1.0 - sim_score  # Higher sim → smaller alpha

    # Clamp alpha to range [0.5, 1.0]
    alpha = (alpha + 0.5) / 2 

    return alpha  # shape: [B]

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

def masked_avg_pool(feat, mask):
    # feat: [B, C, H, W]
    # mask: [B, 1, H_img, W_img] -> 需要下采样到 [B, 1, H, W]
    B, C, H, W = feat.shape
    mask = F.interpolate(mask, size=(H, W), mode='bilinear', align_corners=False)

    masked_feat = feat * mask  # [B, C, H, W]
    sum_feat = masked_feat.sum(dim=(2, 3))  # [B, C]
    mask_sum = mask.sum(dim=(2, 3)) + 1e-5  # [B, 1]
    avg_feat = sum_feat / mask_sum  # [B, C]
    return avg_feat  # [B, C]

def compute_cosine_similarity(query_feats, prototype):
    # query_feats: [B, C, H, W], prototype: [B, C]
    # print("compute_cosine_similarity", query_feats.shape, prototype.shape)
    B, C, H, W = query_feats.shape
    query_feats = F.normalize(query_feats, dim=1)  # [B, C, H, W]
    prototype = F.normalize(prototype, dim=1).view(B, C, 1, 1)  # [B, C, 1, 1]
    sim_map = (query_feats * prototype).sum(dim=1, keepdim=True)  # [B, 1, H, W]
    return sim_map

def compute_query_prior(query_feats, prototype_fg, prototype_bg, temperature=1.0):

    sim_fg = compute_cosine_similarity(query_feats, prototype_fg)  # [B, 1, H, W]
    sim_bg = compute_cosine_similarity(query_feats, prototype_bg)  # [B, 1, H, W]
    sim = torch.cat([sim_fg, sim_bg], dim=1)  # [B, 2, H, W]
    prior = F.softmax(sim / temperature, dim=1)  # [B, 2, H, W]
    return prior[:, 0:1], prior[:, 1:2]  # foreground_prior, background_prior

def get_query_foreground_prototype(query_feats, fg_prior):
    B, C, H, W = query_feats.shape
    weighted_feat = query_feats * fg_prior  # [B, C, H, W]
    proto = weighted_feat.sum(dim=(2, 3)) / (fg_prior.sum(dim=(2, 3)) + 1e-5)  # [B, C]
    return proto

class CenterPivotConv4d(nn.Module):
    r""" CenterPivot 4D conv"""
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=True):
        super(CenterPivotConv4d, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size[:2], stride=stride[:2],
                               bias=bias, padding=padding[:2])
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size[2:], stride=stride[2:],
                               bias=bias, padding=padding[2:])

        self.stride34 = stride[2:]
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.idx_initialized = False

    def prune(self, ct):
        bsz, ch, ha, wa, hb, wb = ct.size()
        if not self.idx_initialized:
            idxh = torch.arange(start=0, end=hb, step=self.stride[2:][0], device=ct.device)
            idxw = torch.arange(start=0, end=wb, step=self.stride[2:][1], device=ct.device)
            self.len_h = len(idxh)
            self.len_w = len(idxw)
            self.idx = (idxw.repeat(self.len_h, 1) + idxh.repeat(self.len_w, 1).t() * wb).view(-1)
            self.idx_initialized = True
        ct_pruned = ct.view(bsz, ch, ha, wa, -1).index_select(4, self.idx).view(bsz, ch, ha, wa, self.len_h, self.len_w)

        return ct_pruned

    def forward(self, x):
        if self.stride[2:][-1] > 1:
            out1 = self.prune(x)
        else:
            out1 = x
        bsz, inch, ha, wa, hb, wb = out1.size()
        out1 = out1.permute(0, 4, 5, 1, 2, 3).contiguous().view(-1, inch, ha, wa)
        out1 = self.conv1(out1)
        outch, o_ha, o_wa = out1.size(-3), out1.size(-2), out1.size(-1)
        out1 = out1.view(bsz, hb, wb, outch, o_ha, o_wa).permute(0, 3, 4, 5, 1, 2).contiguous()

        bsz, inch, ha, wa, hb, wb = x.size()
        out2 = x.permute(0, 2, 3, 1, 4, 5).contiguous().view(-1, inch, hb, wb)
        out2 = self.conv2(out2)
        outch, o_hb, o_wb = out2.size(-3), out2.size(-2), out2.size(-1)
        out2 = out2.view(bsz, ha, wa, outch, o_hb, o_wb).permute(0, 3, 1, 2, 4, 5).contiguous()

        if out1.size()[-2:] != out2.size()[-2:] and self.padding[-2:] == (0, 0):
            out1 = out1.view(bsz, outch, o_ha, o_wa, -1).sum(dim=-1)
            out2 = out2.squeeze()

        y = out1 + out2
        return y

    pass


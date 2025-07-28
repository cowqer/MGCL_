import torch
import torch.nn as nn
import torch.nn.functional as F



class SelfGatingFusion(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.gate_conv = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, feat, prototype):
        # feat, prototype: [B, C, H, W]
        fused_input = torch.cat([feat, prototype], dim=1)  # [B, 2C, H, W]
        gate = self.gate_conv(fused_input)                # [B, C, H, W]
        out = gate * prototype + (1 - gate) * feat
        return out
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

def compute_query_prior(query_feats, prototype_fg, prototype_bg):

    sim_fg = compute_cosine_similarity(query_feats, prototype_fg)  # [B, 1, H, W]
    sim_bg = compute_cosine_similarity(query_feats, prototype_bg)  # [B, 1, H, W]
    sim = torch.cat([sim_fg, sim_bg], dim=1)  # [B, 2, H, W]
    prior = F.softmax(sim, dim=1)  # [B, 2, H, W]
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



class HyperGraphBuilder:
    def __init__(self, epsilon=0.5, device='cpu'):
        """
        初始化超图构建器
        :param epsilon: 特征空间中的ϵ-ball阈值
        :param device: 计算设备（'cpu'或'cuda'）
        """
        self.epsilon = epsilon
        self.device = device

    def fuse_multiscale_features(self, feature_list):
        """
        融合来自多个层次的特征图（通道维拼接）
        :param feature_list: list of tensors [B, C_i, H_i, W_i]
        :return: fused feature map [B, C_total, H, W]
        """
        # 对所有特征图上采样至第一个特征图的空间尺寸
        target_size = feature_list[0].shape[2:]
        resized = [F.interpolate(f, size=target_size, mode='bilinear', align_corners=False)
                   for f in feature_list]
        fused = torch.cat(resized, dim=1)  # 通道维拼接
        return fused

    def build_hypergraph(self, fused_feats):
        """
        构建超图结构（每个位置为顶点，通过ϵ-ball方式连边）
        :param fused_feats: [B, C, H, W] 融合后的特征图
        :return: list of H (B 个样本的超图关联矩阵)
        """
        B, C, H, W = fused_feats.shape
        N = H * W
        fused_feats = fused_feats.view(B, C, N).permute(0, 2, 1)  # [B, N, C]
        
        hypergraphs = []

        for b in range(B):
            feat = fused_feats[b]  # [N, C]
            dist = torch.cdist(feat, feat, p=2)  # [N, N] 欧式距离
            hyperedges = []

            for i in range(N):
                neighbors = torch.where(dist[i] < self.epsilon)[0]
                if len(neighbors) > 0:
                    hyperedges.append(neighbors)

            # 构建关联矩阵 H ∈ [N, E]，E 为超边数
            E = len(hyperedges)
            H = torch.zeros(N, E, device=self.device)

            for e_idx, node_indices in enumerate(hyperedges):
                H[node_indices, e_idx] = 1.0

            hypergraphs.append(H)

        return hypergraphs

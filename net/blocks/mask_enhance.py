import torch
import torch.nn.functional as F
import torch.nn as nn
import torch

def priority_decay_masks(masks, decay=0.5):
    """
    优先级衰减策略处理重叠 mask
    Args:
        masks: [B, M, H, W]  输入 mask (0/1 二值化 或 [0,1] 概率值都行)
        decay: float, 衰减系数 (0~1)，越小对大 mask 压制越强
    Returns:
        new_masks: [B, M, H, W]  衰减后的 mask
    """
    masks = masks.float()  # 
    B, M, H, W = masks.shape
    new_masks = torch.zeros_like(masks)

    for b in range(B):
        # 计算每个 mask 面积
        areas = masks[b].sum(dim=(1, 2))  # [M]
        # 按面积从大到小排序
        sorted_idx = torch.argsort(areas, descending=True)

        # 初始化一个已占用区域
        occupied = torch.zeros((H, W), device=masks.device)

        for rank, idx in enumerate(sorted_idx):
            cur_mask = masks[b, idx]

            # 找到和之前 mask 重叠的区域
            overlap = (cur_mask > 0) & (occupied > 0)

            # 在重叠区域衰减
            adjusted_mask = cur_mask.clone()
            adjusted_mask[overlap] = adjusted_mask[overlap] * decay  

            # 累加到新 mask
            new_masks[b, idx] = adjusted_mask

            # 更新占用区域（这里不做硬覆盖，保留 soft 信息）
            occupied = torch.maximum(occupied, adjusted_mask)

    return new_masks
def priority_decay_masks(masks, decay=0.5):
    """
    优先级衰减策略处理重叠 mask（向量化版本）
    Args:
        masks: [B, M, H, W] 输入 mask (0/1 二值化 或 [0,1] 概率值都行)
        decay: float, 衰减系数 (0~1)，越小对大 mask 压制越强
    Returns:
        new_masks: [B, M, H, W] 衰减后的 mask
    """
    masks = masks.float()
    B, M, H, W = masks.shape

    # 计算面积并排序（大→小）
    areas = masks.sum(dim=(2, 3))  # [B, M]
    sorted_idx = torch.argsort(areas, descending=True, dim=1)  # [B, M]

    # 按排序重排 mask
    sorted_masks = torch.gather(
        masks, 1, sorted_idx[:, :, None, None].expand(B, M, H, W)
    )  # [B, M, H, W]

    # 依次衰减（用累积最大值代替显式 overlap）
    occupied = torch.zeros((B, H, W), device=masks.device)
    new_sorted_masks = torch.zeros_like(sorted_masks)

    for m in range(M):
        cur_mask = sorted_masks[:, m]  # [B, H, W]
        overlap = (cur_mask > 0) & (occupied > 0)
        adjusted = cur_mask.clone()
        adjusted[overlap] = adjusted[overlap] * decay
        new_sorted_masks[:, m] = adjusted
        occupied = torch.maximum(occupied, adjusted)

    # 按原顺序还原
    new_masks = torch.zeros_like(new_sorted_masks)
    new_masks.scatter_(1, sorted_idx[:, :, None, None].expand(B, M, H, W), new_sorted_masks)

    return new_masks
def batch_cos_sim(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # calculate cosine similarity between a and b
    # a: [batch,num_a,channel]
    # b: [batch,num_b,channel]
    # return: [batch,num_a,num_b]
    assert a.shape[0] == b.shape[0], 'batch size of a and b must be equal'
    assert a.shape[2] == b.shape[2], 'channel of a and b must be equal'
    cos_esp = 1e-8
    a_norm = a.norm(dim=2, keepdim=True)
    b_norm = b.norm(dim=2, keepdim=True)
    cos_sim = torch.bmm(a, b.permute(0, 2, 1))
    cos_sim = cos_sim / (torch.bmm(a_norm, b_norm.permute(0, 2, 1)) + cos_esp)
    return cos_sim


class GraphAttention(nn.Module):
    def __init__(self, h_dim=None):
        super(GraphAttention, self).__init__()
        self.with_projection = h_dim is not None
        if self.with_projection:
            self.linear = nn.Linear(h_dim, h_dim)
            # self.linear_k = nn.Linear(h_dim, h_dim)
            # self.linear_v = nn.Linear(h_dim, h_dim)

    def forward(self, q_node, k_node, v_node):
        assert q_node.shape[0] == k_node.shape[0] and q_node.shape[
            0] == v_node.shape[0]
        assert k_node.shape[1] == v_node.shape[1]
        assert q_node.shape[2] == k_node.shape[2]

        if self.with_projection:
            q_node = self.linear(q_node)
            k_node = self.linear(k_node)
            v_node = self.linear(v_node)

        cos_sim = batch_cos_sim(q_node, k_node)
        sum_sim = cos_sim.sum(dim=2, keepdim=True)
        edge_weight = cos_sim / (sum_sim + 1e-8)
        edge_feature = torch.bmm(edge_weight, v_node)
        return edge_feature
    
    

class MaskEnhancer(nn.Module):
    def __init__(self, feature_dims, init_weight=0.1):
        """
        feature_dims: list[int], 各层通道数
        init_weight: 门控参数初始值，越小越保守
        """
        super(MaskEnhancer, self).__init__()
        self.graph_attentions = nn.ModuleList([
            GraphAttention(h_dim=dim) for dim in feature_dims
        ])
        self.gates = nn.ParameterList([
            nn.Parameter(torch.tensor(init_weight)) for _ in feature_dims
        ])

    def forward(self, feats, sam_masks):
        """
        feats: List[[B, C, H, W]]  backbone多层特征
        sam_masks: [B, M, Hm, Wm]  SAM输出mask
        """
        b, m, h_mask, w_mask = sam_masks.shape
        sam_masks = sam_masks[:, 1:, :, :]  # 去掉 dataset 特殊通道
        m = sam_masks.shape[1]

        # 合并重叠mask
        index_mask = torch.zeros_like(sam_masks[:, 0]).long() + m
        for i in range(m):
            index_mask[sam_masks[:, i] == 1] = i
        masks = torch.nn.functional.one_hot(index_mask)[:, :, :, :m].permute(0, 3, 1, 2).float()

        enhanced_feats = []
        for i, (feat, gnn) in enumerate(zip(feats, self.graph_attentions)):
            b, c, h, w = feat.shape

            # mask resize到对应分辨率
            target_masks = F.interpolate(masks, size=(h, w), mode='nearest')

            # masked average pooling -> BxMxC
            map_features = self.masked_average_pooling(feat, target_masks)

            # normalize 避免数值爆炸
            map_features = F.normalize(map_features, p=2, dim=-1)

            # GAT增强
            graph_prompt = gnn(map_features, map_features, map_features)
            graph_prompt = F.normalize(graph_prompt, p=2, dim=-1)

            # 残差+门控
            gate = torch.sigmoid(self.gates[i])
            map_features = map_features + gate * graph_prompt

            # 只增强最后一层，其它层保持原始特征
            if i < len(feats) - 1:
                enhanced_feats.append(feat)
                continue

            # 映射回空间域
            b, m, h_feat, w_feat = target_masks.shape
            _, m, c = map_features.shape
            _map_features = map_features.permute(0, 2, 1).contiguous()  # BxCxM
            feature_sum = _map_features @ target_masks.view(b, m, -1)
            feature_sum = feature_sum.view(b, c, h_feat, w_feat)

            enhanced_feats.append(feat + feature_sum)  # 残差融合

        return enhanced_feats

    def masked_average_pooling(self, feature, mask):
        b, c, h, w = feature.shape
        _, m, _, _ = mask.shape
        _mask = mask.view(b, m, -1)
        _feature = feature.view(b, c, -1).permute(0, 2, 1)  # BxHWxC
        feature_sum = _mask @ _feature  # BxMxC
        masked_sum = torch.sum(_mask, dim=2, keepdim=True)
        return feature_sum / (masked_sum + 1e-5)
    
import torch
import torch.nn.functional as F
import torch.nn as nn

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
    
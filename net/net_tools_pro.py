import torch
import torch.nn as nn
import torch.nn.functional as F
from .net_tools import MGCDModule, MGFEModule

class MGFEModuleV2(object):
    """
    改进版 MGFE 模块，支持选择性池化：
    - mode='avg'  : 原来的 masked average pooling
    - mode='max'  : masked max pooling
    - mode='topk' : masked top-k pooling
    """

    pool_mode = 'avg'  # 默认池化模式
    topk = 5           # top-k 参数

    @classmethod
    def update_feature_one(cls, query_feat, query_mask):
        """
        给一个特征和 mask，输出增强后的特征
        """
        return cls.enabled_feature([query_feat], query_mask)[0]

    @classmethod
    def update_feature(cls, query_feats, support_feats, query_mask, support_masks):
        """
        给一组特征和 mask，输出增强后的特征
        """
        query_feats = cls.enabled_feature(query_feats, query_mask)
        support_feats = cls.enabled_feature(support_feats, support_masks)
        return query_feats, support_feats

    @classmethod
    def enabled_feature(cls, feats, masks):
        """
        给定特征图列表 feats 和它们的掩膜 masks，返回根据掩膜增强的特征图
        """
        b, m, w, h = masks.shape
        index_mask = torch.zeros_like(masks[:, 0]).long() + m
        for i in range(m):
            index_mask[masks[:, i] == 1] = i
        masks = torch.nn.functional.one_hot(index_mask)[:, :, :, :m].permute(0, 3, 1, 2)

        enabled_feats = []
        for feat in feats:
            target_masks = F.interpolate(masks.float(), feat.shape[-2:], mode='nearest')
            
            # 根据选择的模式计算每个区域的特征
            if cls.pool_mode == 'avg':
                map_features = cls.my_masked_average_pooling(feat, target_masks)
            elif cls.pool_mode == 'max':
                map_features = cls.my_masked_max_pooling(feat, target_masks)
            elif cls.pool_mode == 'topk':
                map_features = cls.my_masked_topk_pooling(feat, target_masks, k=cls.topk)
            else:
                raise ValueError(f"Unsupported pool_mode: {cls.pool_mode}")

            b, m, w_feat, h_feat = target_masks.shape
            _, _, c = map_features.shape
            _map_features = map_features.permute(0, 2, 1).contiguous()
            feature_sum = _map_features @ target_masks.view(b, m, -1)
            feature_sum = feature_sum.view(b, c, w_feat, h_feat)

            sum_mask = target_masks.sum(dim=1, keepdim=True)
            enabled_feat = torch.div(feature_sum, sum_mask + 1e-8)
            enabled_feats.append(enabled_feat)

        return enabled_feats

    @staticmethod
    def my_masked_average_pooling(feature, mask):
        b, c, w, h = feature.shape
        _, m, _, _ = mask.shape
        _mask = mask.view(b, m, -1)
        _feature = feature.view(b, c, -1).permute(0, 2, 1).contiguous()
        feature_sum = _mask @ _feature
        masked_sum = torch.sum(_mask, dim=2, keepdim=True)
        return torch.div(feature_sum, masked_sum + 1e-8)

    @staticmethod
    def my_masked_max_pooling(feature, mask):
        b, c, w, h = feature.shape
        _, m, _, _ = mask.shape
        _feature = feature.view(b, c, -1).unsqueeze(1).expand(-1, m, -1, -1)  # [b, m, c, w*h]
        _mask = mask.view(b, m, 1, -1)
        _feature = _feature * _mask.float() + (_mask == 0).float() * -1e9
        masked_max = _feature.max(dim=-1)[0]  # [b, m, c]
        return masked_max

    @staticmethod
    def my_masked_topk_pooling(feature, mask, k=5):
        b, c, w, h = feature.shape
        _, m, _, _ = mask.shape
        _feature = feature.view(b, c, -1).permute(0, 2, 1)  # [b, w*h, c]
        _mask = mask.view(b, m, -1)
        pooled = []
        for i in range(m):
            masked_feat = _feature * _mask[:, i:i+1].permute(0, 2, 1).contiguous()
            topk_val, _ = masked_feat.topk(min(k, masked_feat.shape[1]), dim=1)
            pooled.append(topk_val.mean(dim=1))
        pooled = torch.stack(pooled, dim=1)
        return pooled

class CDModule(MGCDModule):
    """
    改进版 MGCD 模块，支持选择性池化
    """

    def __init__(self, inch):
        super().__init__(inch)
        self.decoder1 = nn.Sequential(
            nn.Conv2d(128, 64, (3, 3), padding=(1, 1), bias=True), nn.ReLU(),
            nn.Conv2d(64, 32, (3, 3), padding=(1, 1), bias=True), nn.ReLU())
        self.decoder1_my = nn.Sequential(
            nn.Conv2d(64, 32, (3, 3), padding=(1, 1), bias=True), nn.ReLU(),
            nn.Conv2d(32, 32, (3, 3), padding=(1, 1), bias=True), nn.ReLU())
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

        if query_mask is not None:
            # MGFE
            # _hypercorr_encoded = MGFEModule.update_feature_one(hypercorr_encoded, query_mask)
            # hypercorr_encoded = torch.concat([hypercorr_encoded, _hypercorr_encoded], dim=1)
            hypercorr_encoded = torch.concat([hypercorr_encoded, hypercorr_encoded], dim=1)
        else:
            hypercorr_encoded = torch.concat([hypercorr_encoded, hypercorr_encoded], dim=1)
            pass

        # Decode the encoded 4D-tensor
        hypercorr_decoded = self.decoder1(hypercorr_encoded)
        # hypercorr_decoded = self.decoder1_my(hypercorr_decoded)
        upsample_size = (hypercorr_decoded.size(-1) * 2,) * 2
        hypercorr_decoded = F.interpolate(hypercorr_decoded, upsample_size,
                                          mode='bilinear', align_corners=True)
        logit = self.decoder2(hypercorr_decoded)
        return logit

    pass

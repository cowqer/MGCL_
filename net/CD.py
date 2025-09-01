import torch
import torch.nn as nn
import torch.nn.functional as F
from .net_tools_pro import CDModule
from .modules import CenterPivotConv4d
from .blocks.select_sobel import SSblock
from .blocks.maskdecoder import MaskPriorDecoder


class SSCDModule(nn.Module):
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
        # self.decoder1_my = nn.Sequential(
        #     nn.Conv2d(64, 32, (3, 3), padding=(1, 1), bias=True), nn.ReLU(),
        #     nn.Conv2d(32, 32, (3, 3), padding=(1, 1), bias=True), nn.ReLU())
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

        if query_mask is not None:
            hypercorr_encoded = torch.concat([hypercorr_encoded, hypercorr_encoded], dim=1)
        else:
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

class MCDModule(nn.Module):
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
        # self.decoder1_my = nn.Sequential(
        #     nn.Conv2d(64, 32, (3, 3), padding=(1, 1), bias=True), nn.ReLU(),
        #     nn.Conv2d(32, 32, (3, 3), padding=(1, 1), bias=True), nn.ReLU())
        self.decoder2 = nn.Sequential(
            nn.Conv2d(outch2, outch1, (3, 3), padding=(1, 1), bias=True), nn.ReLU(),
            nn.Conv2d(outch1, 2, (3, 3), padding=(1, 1), bias=True))
        self.mask_decoder = MaskPriorDecoder(in_channels=outch4, mask_channels=128)
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

        if query_mask is not None:
            hypercorr_encoded = torch.concat([hypercorr_encoded, hypercorr_encoded], dim=1)
        else:
            hypercorr_encoded = torch.concat([hypercorr_encoded, hypercorr_encoded], dim=1)
            pass

        hypercorr_decoded = self.mask_decoder(hypercorr_encoded, query_mask) if query_mask is not None else hypercorr_encoded
       
        hypercorr_decoded = self.decoder1(hypercorr_encoded)
        upsample_size = (hypercorr_decoded.size(-1) * 2,) * 2
        hypercorr_decoded = F.interpolate(hypercorr_decoded, upsample_size,
                                          mode='bilinear', align_corners=True)
        logit = self.decoder2(hypercorr_decoded)
        return logit

    pass

import torch
import torch.nn as nn
import torch.nn.functional as F


class Sobel3x3(nn.Module):
    """3x3 Sobel 卷积"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        sobel_kernel_x = torch.tensor([[-1., 0., 1.],
                                       [-2., 0., 2.],
                                       [-1., 0., 1.]])
        sobel_kernel_y = torch.tensor([[-1., -2., -1.],
                                       [0.,  0.,  0.],
                                       [1.,  2.,  1.]])
        kernel = torch.stack([sobel_kernel_x, sobel_kernel_y])  # [2, 3, 3]
        kernel = kernel.unsqueeze(1)  # [2,1,3,3]
        kernel = kernel.repeat(in_channels, 1, 1, 1)  # [2*in_ch,1,3,3]
        self.weight = nn.Parameter(kernel, requires_grad=False)
        self.groups = in_channels

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                              stride=1, padding=1, bias=False, groups=in_channels)
        self.conv.weight.data = self.weight[:out_channels]

    def forward(self, x):
        return self.conv(x)


class Sobel5x5(nn.Module):
    """5x5 Sobel 卷积"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        sobel_kernel_x = torch.tensor([[-5., -4., 0., 4., 5.],
                                       [-8., -10., 0., 10., 8.],
                                       [-10., -20., 0., 20., 10.],
                                       [-8., -10., 0., 10., 8.],
                                       [-5., -4., 0., 4., 5.]])
        sobel_kernel_y = sobel_kernel_x.T
        kernel = torch.stack([sobel_kernel_x, sobel_kernel_y])  # [2,5,5]
        kernel = kernel.unsqueeze(1)
        kernel = kernel.repeat(in_channels, 1, 1, 1)  # [2*in_ch,1,5,5]
        self.weight = nn.Parameter(kernel, requires_grad=False)
        self.groups = in_channels

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=5,
                              stride=1, padding=2, bias=False, groups=in_channels)
        self.conv.weight.data = self.weight[:out_channels]

    def forward(self, x):
        return self.conv(x)


class SSblock(nn.Module):
    """Sobel + Attention 模块"""
    def __init__(self, dim):
        super().__init__()
        self.sobel1 = Sobel3x3(dim, dim)
        self.sobel2 = Sobel5x5(dim, dim)

        self.conv1 = nn.Conv2d(dim, dim // 2, 1)
        self.conv2 = nn.Conv2d(dim, dim // 2, 1)

        self.conv_squeeze = nn.Conv2d(2, 2, 7, padding=3)
        self.conv = nn.Conv2d(dim // 2, dim, 1)

    def forward(self, x):
        # Sobel 提取
        attn1 = self.sobel1(x)
        attn2 = self.sobel2(attn1)

        # 通道压缩
        attn1 = self.conv1(attn1)
        attn2 = self.conv2(attn2)

        # 拼接后求 avg & max
        attn = torch.cat([attn1, attn2], dim=1)
        avg_attn = torch.mean(attn, dim=1, keepdim=True)
        max_attn, _ = torch.max(attn, dim=1, keepdim=True)
        agg = torch.cat([avg_attn, max_attn], dim=1)

        # 生成权重
        sig = self.conv_squeeze(agg).sigmoid()
        attn = attn1 * sig[:, 0:1, :, :] + attn2 * sig[:, 1:2, :, :]

        # 融合回原始维度
        attn = self.conv(attn)
        return x * (1 + attn)   # 这样保证不会掉点（恒等映射时退化为原始 x）


if __name__ == "__main__":
    x = torch.randn(2, 64, 128, 128)  # [B, C, H, W]
    block = SSblock(64)
    y = block(x)
    print(y.shape)  # torch.Size([2, 64, 128, 128])

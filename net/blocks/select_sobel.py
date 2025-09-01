import torch
import torch.nn as nn
import torch.nn.functional as F

class Sobel5x5(nn.Module):
    def __init__(self, channels=3, padding=2):
        """
        5x5 Sobel 卷积，输出为 sqrt(Gx^2 + Gy^2)，通道数保持不变
        """
        super(Sobel5x5, self).__init__()
        
        # 5x5 Sobel 核
        sobel_kernel_x = torch.tensor(
            [[-2., -1., 0., 1., 2.],
             [-2., -1., 0., 1., 2.],
             [-4., -2., 0., 2., 4.],
             [-2., -1., 0., 1., 2.],
             [-2., -1., 0., 1., 2.]]
        )

        sobel_kernel_y = sobel_kernel_x.t()
        ks = 5
        # conv_x
        self.conv_x = nn.Conv2d(channels, channels, kernel_size=ks, 
                                stride=1, padding=ks//2, bias=False, groups=channels)
        self.conv_y = nn.Conv2d(channels, channels, kernel_size=ks, 
                                stride=1, padding=ks//2, bias=False, groups=channels)


        # 初始化
        kernel_x = sobel_kernel_x.unsqueeze(0).unsqueeze(0).repeat(channels, 1, 1, 1)
        kernel_y = sobel_kernel_y.unsqueeze(0).unsqueeze(0).repeat(channels, 1, 1, 1)

        with torch.no_grad():
            self.conv_x.weight.copy_(kernel_x)
            self.conv_y.weight.copy_(kernel_y)

    def forward(self, x):
        gx = self.conv_x(x)
        gy = self.conv_y(x)
        return torch.sqrt(gx ** 2 + gy ** 2 + 1e-6)  # 避免 sqrt(0)


class Sobel3x3(nn.Module):
    def __init__(self, channels=3, padding=1):
        """
        3x3 Sobel 卷积，输出为 sqrt(Gx^2 + Gy^2)，通道数保持不变
        """
        super(Sobel3x3, self).__init__()
        
        sobel_kernel_x = torch.tensor([[-1., 0., 1.],
                                       [-2., 0., 2.],
                                       [-1., 0., 1.]])
        
        sobel_kernel_y = torch.tensor([[-1., -2., -1.],
                                       [ 0.,  0.,  0.],
                                       [ 1.,  2.,  1.]])
        ks = 3
        self.conv_x = nn.Conv2d(channels, channels, kernel_size=ks, 
                                stride=1, padding=ks//2, bias=False, groups=channels)
        self.conv_y = nn.Conv2d(channels, channels, kernel_size=ks, 
                                stride=1, padding=ks//2, bias=False, groups=channels)


        kernel_x = sobel_kernel_x.unsqueeze(0).unsqueeze(0).repeat(channels, 1, 1, 1)
        kernel_y = sobel_kernel_y.unsqueeze(0).unsqueeze(0).repeat(channels, 1, 1, 1)

        with torch.no_grad():
            self.conv_x.weight.copy_(kernel_x)
            self.conv_y.weight.copy_(kernel_y)

    def forward(self, x):
        gx = self.conv_x(x)
        gy = self.conv_y(x)
        return torch.sqrt(gx ** 2 + gy ** 2 + 1e-6)
    
    
    
class SSblock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.sobel1 = Sobel3x3(dim, dim)
        self.sobel2 = Sobel5x5(dim, dim)
        
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv_spatial = nn.Conv2d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
        
        self.conv1 = nn.Conv2d(dim, dim//2, 1)
        self.conv2 = nn.Conv2d(dim, dim//2, 1)
        self.conv_squeeze = nn.Conv2d(2, 2, 7, padding=3)
        self.conv = nn.Conv2d(dim//2, dim, 1)

    def forward(self, x):   
        attn1 = self.sobel1(x)
        attn2 = self.sobel2(attn1)
        # attn1 = self.conv0(x)
        # attn2 = self.conv_spatial(attn1)
        print(attn1.shape, attn2.shape)
        attn1 = self.conv1(attn1)
        attn2 = self.conv2(attn2)
        
        attn = torch.cat([attn1, attn2], dim=1)
        avg_attn = torch.mean(attn, dim=1, keepdim=True)
        max_attn, _ = torch.max(attn, dim=1, keepdim=True)
        agg = torch.cat([avg_attn, max_attn], dim=1)
        sig = self.conv_squeeze(agg).sigmoid()
        attn = attn1 * sig[:,0,:,:].unsqueeze(1) + attn2 * sig[:,1,:,:].unsqueeze(1)
        attn = self.conv(attn)
        return x * attn



class Attention(nn.Module):
    def __init__(self, d_model):
        super().__init__()

        self.proj_1 = nn.Conv2d(d_model, d_model, 1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = SSblock(d_model)
        self.proj_2 = nn.Conv2d(d_model, d_model, 1)

    def forward(self, x):
        shorcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        x = x + shorcut
        return x

# 测试
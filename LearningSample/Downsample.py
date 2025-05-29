import torch
import torch.nn as nn
import torch.nn.functional as F

class Downsample(nn.Module):
    """ConvNeXt的下采样层
    
    作用：
    1. 空间维度缩小：H×W → H/2×W/2
    2. 通道维度增加：通常 C → 2C
    
    实现方式：LayerNorm + 2×2卷积(stride=2)
    """
    
    def __init__(self, input_dim, output_dim):
        """
        Args:
            input_dim: 输入通道数
            output_dim: 输出通道数（通常是input_dim的2倍）
        """
        super().__init__()
        
        # LayerNorm（channels_first格式）
        self.norm = LayerNorm(input_dim, eps=1e-6, data_format="channels_first")
        
        # 2×2卷积，stride=2实现下采样
        self.reduction = nn.Conv2d(input_dim, output_dim, kernel_size=2, stride=2)

    def forward(self, x):
        """
        Args:
            x: 输入特征图 [B, input_dim, H, W]
        Returns:
            下采样后的特征图 [B, output_dim, H/2, W/2]
        """
        x = self.norm(x)
        x = self.reduction(x)
        return x


class LayerNorm(nn.Module):
    """为2D特征图设计的LayerNorm（从之前的代码复制）"""
    
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


# 测试下采样层
if __name__ == "__main__":
    print("=== ConvNeXt 下采样层测试 ===")
    
    # 测试1：基本功能测试
    print("\n1. 基本功能测试：")
    downsample = Downsample(input_dim=96, output_dim=192)
    
    # 输入：[batch=2, channels=96, height=32, width=32]
    x = torch.randn(2, 96, 32, 32)
    output = downsample(x)
    
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    print(f"空间维度是否缩小一半: {output.shape[2] == x.shape[2]//2 and output.shape[3] == x.shape[3]//2}")
    print(f"通道维度是否正确: {output.shape[1] == 192}")
    
    # 测试2：不同尺寸测试
    print("\n2. 不同输入尺寸测试：")
    test_sizes = [(64, 64), (128, 128), (256, 256)]
    
    for h, w in test_sizes:
        x_test = torch.randn(1, 96, h, w)
        output_test = downsample(x_test)
        print(f"输入 [1, 96, {h}, {w}] → 输出 {list(output_test.shape)}")
    
    # 测试3：参数量统计
    print(f"\n3. 下采样层参数量: {sum(p.numel() for p in downsample.parameters()):,}")
    
    # 测试4：模拟完整的stage转换
    print("\n4. 模拟ConvNeXt的stage转换：")
    stage_configs = [
        (96, 192),    # Stage 1 → Stage 2
        (192, 384),   # Stage 2 → Stage 3  
        (384, 768),   # Stage 3 → Stage 4
    ]
    
    x = torch.randn(1, 96, 56, 56)  # 模拟stage 1的输出
    print(f"初始特征图: {list(x.shape)}")
    
    for i, (in_dim, out_dim) in enumerate(stage_configs):
        downsample_layer = Downsample(in_dim, out_dim)
        x = downsample_layer(x)
        print(f"经过下采样层 {i+1}: {list(x.shape)}")


# 额外：对比不同下采样方法的实现
class AlternativeDownsample(nn.Module):
    """对比：其他常见的下采样方法"""
    
    def __init__(self, input_dim, output_dim, method="conv"):
        super().__init__()
        self.method = method
        
        if method == "conv":
            # 方法1：2×2卷积 stride=2（ConvNeXt使用的方法）
            self.downsample = nn.Conv2d(input_dim, output_dim, kernel_size=2, stride=2)
            
        elif method == "maxpool":
            # 方法2：最大池化 + 1×1卷积调整通道
            self.downsample = nn.Sequential(
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(input_dim, output_dim, kernel_size=1)
            )
            
        elif method == "avgpool":
            # 方法3：平均池化 + 1×1卷积调整通道
            self.downsample = nn.Sequential(
                nn.AvgPool2d(kernel_size=2, stride=2),
                nn.Conv2d(input_dim, output_dim, kernel_size=1)
            )
    
    def forward(self, x):
        return self.downsample(x)


# 比较不同下采样方法
if __name__ == "__main__":
    print("\n=== 不同下采样方法对比 ===")
    
    x = torch.randn(1, 96, 32, 32)
    methods = ["conv", "maxpool", "avgpool"]
    
    for method in methods:
        downsampler = AlternativeDownsample(96, 192, method=method)
        output = downsampler(x)
        params = sum(p.numel() for p in downsampler.parameters())
        print(f"{method:8s} 方法: 输出形状 {list(output.shape)}, 参数量 {params:,}")
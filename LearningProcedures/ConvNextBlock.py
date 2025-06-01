import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvNeXtBlock(nn.Module):
    """ConvNeXt Block 基础实现
    
    ConvNeXt Block的结构：
    输入 → 7x7 DWConv → LayerNorm → 1x1 Conv(扩展4倍) → GELU → 1x1 Conv(还原) → DropPath → 残差连接 → 输出
    """
    
    def __init__(self, dim, drop_path=0.):
        """
        Args:
            dim: 输入和输出的通道数
            drop_path: DropPath的概率，用于正则化
        """
        super().__init__()
        
        # 第一到第三步：升级版深度可分离卷积 (深度卷积 + 倒置瓶颈)

        # 第一步：7x7深度卷积 (groups=dim表示深度卷积)
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        
        # 第二步：LayerNorm (注意：这里需要特殊处理，因为LayerNorm通常用于1D)
        self.norm = LayerNorm(dim, eps=1e-6)
        
        # 第三步：两个1x1卷积构成的倒置瓶颈 (这里同时也是深度可分离卷积中的逐点卷积升级版)
        # 当kernel_size=1时，Linear层等价于1×1卷积，但Linear层在[B, H, W, C]格式下更高效
        # 先升维到4倍通道数
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        # 再降维回原来的通道数
        self.pwconv2 = nn.Linear(4 * dim, dim)
        
        # 第四步：GELU激活函数
        self.act = nn.GELU()
        
        # 第五步：DropPath (随机丢弃路径，用于正则化)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        """
        Args:
            x: 输入特征图，形状为 [B, C, H, W] (B: batch_size, C: channels, H: height, W: width)
        Returns:
            输出特征图，形状为 [B, C, H, W]
        """
        # 保存输入，用于残差连接
        input = x
        
        # 步骤1: 7x7深度卷积
        x = self.dwconv(x)
        
        # 步骤2: LayerNorm
        # 需要调整维度：[B, C, H, W] → [B, H, W, C]
        x = x.permute(0, 2, 3, 1)  # (B, C, H, W) -> (B, H, W, C)
        x = self.norm(x)
        
        # 步骤3: 第一个1x1卷积 (升维)
        x = self.pwconv1(x)
        
        # 步骤4: GELU激活
        x = self.act(x)
        
        # 步骤5: 第二个1x1卷积 (降维)
        x = self.pwconv2(x)
        
        # 调整维度回原来的格式：[B, H, W, C] → [B, C, H, W]
        x = x.permute(0, 3, 1, 2)  # (B, H, W, C) -> (B, C, H, W)
        
        # 步骤6: DropPath + 残差连接
        x = input + self.drop_path(x)
        
        return x


class LayerNorm(nn.Module):
    """为2D特征图设计的LayerNorm
    
    标准的LayerNorm用于1D序列，这里我们需要适配2D特征图
    """
    
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
            # x的形状: [B, H, W, C]
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            # x的形状: [B, C, H, W]
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    """逐样本的路径丢弃 (随机深度) (当应用于残差块的主路径时)"""
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """逐样本的路径丢弃 (随机深度) (当应用于残差块的主路径时)"""
    
    def __init__(self, drop_prob=None, scale_by_keep=True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)


# 测试代码
if __name__ == "__main__":
    # 创建一个ConvNeXt Block
    block = ConvNeXtBlock(dim=96)
    
    # 创建测试输入 [batch_size=2, channels=96, height=32, width=32]
    x = torch.randn(2, 96, 32, 32)
    
    # 前向传播
    output = block(x)
    
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    print(f"输入输出形状是否一致: {x.shape == output.shape}")
    
    # 计算参数量
    total_params = sum(p.numel() for p in block.parameters())
    print(f"ConvNeXt Block 参数量: {total_params:,}")
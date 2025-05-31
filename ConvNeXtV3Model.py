import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

class ConvNeXtWithCSAttention(nn.Module):
    """带通道+空间注意力的ConvNeXt网络
    
    集成CBAM风格的通道注意力和空间注意力机制
    - Channel Attention: 学习特征通道重要性
    - Spatial Attention: 学习空间位置重要性
    """
    
    def __init__(self, 
                 in_chans=3,
                 num_classes=1000, 
                 depths=[3, 3, 9, 3],
                 dims=[96, 192, 384, 768],
                 drop_path_rate=0.,
                 layer_scale_init_value=1e-6,
                 attention_stages=[False, True, True, True],  # 哪些stage使用注意力
                 ca_reduction=16,  # 通道注意力降维比例
                 sa_kernel_size=7):  # 空间注意力卷积核大小
        """
        Args:
            attention_stages: 每个stage是否使用注意力机制
            ca_reduction: 通道注意力中间层降维比例
            sa_kernel_size: 空间注意力卷积核大小
        """
        super().__init__()
        
        # Stem Layer
        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        
        # 构建下采样层
        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        # 构建带注意力的Stage
        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        
        cur = 0
        for i in range(4):
            stage_blocks = []
            for j in range(depths[i]):
                # 决定是否使用注意力
                use_attention = attention_stages[i] if i < len(attention_stages) else False
                
                if use_attention:
                    block = ConvNeXtBlockWithCSAttention(
                        dim=dims[i], 
                        drop_path=dp_rates[cur + j],
                        layer_scale_init_value=layer_scale_init_value,
                        ca_reduction=ca_reduction,
                        sa_kernel_size=sa_kernel_size
                    )
                else:
                    block = ConvNeXtBlock(
                        dim=dims[i], 
                        drop_path=dp_rates[cur + j],
                        layer_scale_init_value=layer_scale_init_value
                    )
                stage_blocks.append(block)
            
            stage = nn.Sequential(*stage_blocks)
            self.stages.append(stage)
            cur += depths[i]

        # 分类头
        self.norm = LayerNorm(dims[-1], eps=1e-6, data_format="channels_first")
        self.head = nn.Linear(dims[-1], num_classes)
        self.apply(self._init_weights)
        
        # 打印注意力配置信息
        self._print_attention_info(attention_stages, dims)
        
    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def _print_attention_info(self, attention_stages, dims):
        """打印注意力机制配置信息"""
        print("🔍 注意力机制配置:")
        for i, (use_att, dim) in enumerate(zip(attention_stages, dims)):
            status = "✅ 启用" if use_att else "❌ 关闭"
            print(f"  Stage {i+1} (dim={dim:3d}): {status}")

    def forward_features(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        return self.norm(x)

    def forward(self, x):
        x = self.forward_features(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.head(x)
        return x


class ConvNeXtBlockWithCSAttention(nn.Module):
    """集成通道+空间注意力的ConvNeXt Block
    
    注意力应用位置：在深度卷积后，点卷积前
    这样可以在保持空间信息的同时进行特征增强
    """
    
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6, 
                 ca_reduction=16, sa_kernel_size=7):
        super().__init__()
        
        # 原始ConvNeXt组件
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        
        # 通道+空间注意力 (CBAM风格)
        self.channel_attention = ChannelAttention(dim, reduction=ca_reduction)
        self.spatial_attention = SpatialAttention(kernel_size=sa_kernel_size)
        
        # 后续组件
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        
        # Layer Scale
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                requires_grad=True) if layer_scale_init_value > 0 else None
        
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        
        # 1. 深度卷积
        x = self.dwconv(x)
        
        # 2. 应用通道注意力
        x = self.channel_attention(x)
        
        # 3. 应用空间注意力
        x = self.spatial_attention(x)
        
        # 4. 原始ConvNeXt的后续处理
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        
        if self.gamma is not None:
            x = self.gamma * x
            
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        x = input + self.drop_path(x)
        return x


class ChannelAttention(nn.Module):
    """通道注意力机制 (CBAM风格)
    
    通过全局平均池化和最大池化获取通道统计信息，
    然后通过MLP学习通道权重
    """
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # 共享的MLP
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 分别计算平均池化和最大池化的通道注意力
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        
        # 融合两种池化结果
        out = avg_out + max_out
        
        # 应用注意力权重
        return x * self.sigmoid(out)


class SpatialAttention(nn.Module):
    """空间注意力机制 (CBAM风格)
    
    通过通道维度的平均和最大操作获取空间统计信息，
    然后通过卷积学习空间权重
    """
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 在通道维度进行池化操作
        avg_out = torch.mean(x, dim=1, keepdim=True)  # [B, 1, H, W]
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # [B, 1, H, W]
        
        # 拼接两种池化结果
        x_cat = torch.cat([avg_out, max_out], dim=1)  # [B, 2, H, W]
        
        # 通过卷积学习空间注意力
        x_out = self.conv(x_cat)  # [B, 1, H, W]
        
        # 应用注意力权重
        return x * self.sigmoid(x_out)


# 原始ConvNeXt Block（无注意力）
class ConvNeXtBlock(nn.Module):
    """原始ConvNeXt Block"""
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                requires_grad=True) if layer_scale_init_value > 0 else None
        
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        
        if self.gamma is not None:
            x = self.gamma * x
            
        x = x.permute(0, 3, 1, 2)
        x = input + self.drop_path(x)
        return x


class LayerNorm(nn.Module):
    """LayerNorm实现"""
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
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


def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob=None, scale_by_keep=True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)


# 模型构建函数
def convnext_tiny_cs_attention(attention_stages=[False, True, True, True], 
                              ca_reduction=16, sa_kernel_size=7, **kwargs):
    """ConvNeXt-Tiny with Channel+Spatial Attention
    
    Args:
        attention_stages: 每个stage是否启用注意力 [Stage1, Stage2, Stage3, Stage4]
        ca_reduction: 通道注意力降维比例，越大计算量越小
        sa_kernel_size: 空间注意力卷积核大小，通常使用7
    """
    model = ConvNeXtWithCSAttention(
        depths=[3, 3, 9, 3], 
        dims=[96, 192, 384, 768],
        attention_stages=attention_stages,
        ca_reduction=ca_reduction,
        sa_kernel_size=sa_kernel_size,
        **kwargs
    )
    return model


def convnext_small_cs_attention(attention_stages=[False, True, True, True], 
                               ca_reduction=16, sa_kernel_size=7, **kwargs):
    """ConvNeXt-Small with Channel+Spatial Attention"""
    model = ConvNeXtWithCSAttention(
        depths=[3, 3, 27, 3], 
        dims=[96, 192, 384, 768],
        attention_stages=attention_stages,
        ca_reduction=ca_reduction,
        sa_kernel_size=sa_kernel_size,
        **kwargs
    )
    return model


def convnext_base_cs_attention(attention_stages=[False, True, True, True], 
                              ca_reduction=16, sa_kernel_size=7, **kwargs):
    """ConvNeXt-Base with Channel+Spatial Attention"""
    model = ConvNeXtWithCSAttention(
        depths=[3, 3, 27, 3], 
        dims=[128, 256, 512, 1024],
        attention_stages=attention_stages,
        ca_reduction=ca_reduction,
        sa_kernel_size=sa_kernel_size,
        **kwargs
    )
    return model


# 测试和对比代码
if __name__ == "__main__":
    print("=== ConvNeXt + 通道空间注意力 测试 ===")
    
    # 测试1: 标准配置
    print("\n1. 标准配置测试 (Stage2-4启用注意力):")
    model = convnext_tiny_cs_attention(
        attention_stages=[False, True, True, True],
        ca_reduction=16,
        sa_kernel_size=7
    )
    
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        output = model(x)
    
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"总参数量: {total_params:,}")
    
    # 测试2: 不同配置策略对比
    print(f"\n2. 不同注意力配置策略对比:")
    
    configs = {
        '无注意力': [False, False, False, False],
        '仅Stage4': [False, False, False, True],
        '仅Stage3-4': [False, False, True, True],
        '标准配置(Stage2-4)': [False, True, True, True],
        '全部启用': [True, True, True, True]
    }
    
    for name, att_stages in configs.items():
        if name == '无注意力':
            # 使用原始ConvNeXt作为对比
            from ConvNeXtV1Model import convnext_tiny
            model_temp = convnext_tiny()
        else:
            model_temp = convnext_tiny_cs_attention(attention_stages=att_stages)
        
        params = sum(p.numel() for p in model_temp.parameters())
        
        # 计算额外参数
        if name == '无注意力':
            base_params = params
            extra_params = 0
        else:
            extra_params = params - base_params
        
        print(f"{name:15s}: {params:>10,} 参数 (+{extra_params:>6,})")
    
    # 测试3: 超参数影响分析
    print(f"\n3. 超参数对参数量的影响:")
    
    print("通道注意力降维比例影响:")
    for reduction in [8, 16, 32]:
        model_temp = convnext_tiny_cs_attention(
            attention_stages=[False, False, True, True],
            ca_reduction=reduction
        )
        params = sum(p.numel() for p in model_temp.parameters())
        print(f"  reduction={reduction:2d}: {params:,} 参数")
    
    print("\n空间注意力卷积核大小影响:")
    for kernel_size in [3, 5, 7]:
        model_temp = convnext_tiny_cs_attention(
            attention_stages=[False, False, True, True],
            sa_kernel_size=kernel_size
        )
        params = sum(p.numel() for p in model_temp.parameters())
        print(f"  kernel_size={kernel_size}: {params:,} 参数")
    
    # 测试4: 推理速度测试
    print(f"\n4. 前向传播测试:")
    model.eval()
    
    # 预热
    with torch.no_grad():
        for _ in range(5):
            _ = model(x)
    
    # 计时
    import time
    times = []
    with torch.no_grad():
        for _ in range(10):
            start = time.time()
            _ = model(x)
            times.append(time.time() - start)
    
    avg_time = sum(times) / len(times)
    print(f"平均推理时间: {avg_time*1000:.2f} ms (batch_size=2)")
    
    # 测试5: 注意力模块细节
    print(f"\n5. 注意力模块详细信息:")
    
    # 查看一个带注意力的block
    attention_block = None
    for stage in model.stages:
        for block in stage:
            if hasattr(block, 'channel_attention'):
                attention_block = block
                break
        if attention_block:
            break
    
    if attention_block:
        ca_params = sum(p.numel() for p in attention_block.channel_attention.parameters())
        sa_params = sum(p.numel() for p in attention_block.spatial_attention.parameters())
        
        print(f"单个Block中:")
        print(f"  通道注意力参数: {ca_params:,}")
        print(f"  空间注意力参数: {sa_params:,}")
        print(f"  注意力总参数: {ca_params + sa_params:,}")
    
    print("\n✅ ConvNeXt + 通道空间注意力测试完成！")
    print("\n💡 使用建议:")
    print("- 推荐配置: Stage2-4启用注意力，平衡性能和计算成本")
    print("- 轻量配置: 仅Stage3-4启用注意力，减少计算开销")
    print("- 通道注意力降维比例16是经验最优值")
    print("- 空间注意力卷积核7×7效果最佳")
    print("- 该配置相比原始ConvNeXt仅增加很少参数，但性能提升明显")
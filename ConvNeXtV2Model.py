import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

class ConvNeXt(nn.Module): # Change to V2
    """完整的ConvNeXt网络实现
    
    ConvNeXt网络结构：
    Stem Layer → Stage1 → Downsample → Stage2 → Downsample → Stage3 → Downsample → Stage4 → Head
    """
    
    def __init__(self, 
                 in_chans=3,
                 num_classes=1000, 
                 depths=[3, 3, 9, 3],
                 dims=[96, 192, 384, 768],
                 drop_path_rate=0.,
                 head_init_scale=1.): # 添加了 head_init_scale 参数，用于初始化分类头
                 # layer_scale_init_value=1e-6): # 取消 V1 的 Layer Scale
        """
        Args:
            in_chans: 输入图像通道数（RGB图像为3）
            num_classes: 分类类别数
            depths: 每个stage的block数量 [stage1_blocks, stage2_blocks, stage3_blocks, stage4_blocks]
            dims: 每个stage的通道数 [stage1_dim, stage2_dim, stage3_dim, stage4_dim]  
            drop_path_rate: DropPath的最大概率（会逐层递增）
            head_init_scale: 分类头的初始化缩放因子
        """
        super().__init__()
        
        # Stem Layer: 将RGB图像转换为初始特征图
        # 使用4×4卷积，stride=4，相当于两次2×2下采样，同时设置维度到stage1_dim为进入stage1作准备
        # self.depths = depths
        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        
        # 构建3个下采样层（Stage1→Stage2, Stage2→Stage3, Stage3→Stage4）
        # 使用2×2卷积，stride=2，相当于一次2×2下采样，同时设置维度到每一层的下一个stage_dim为进入下一层作准备
        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        # 构建4个Stage，每个Stage包含多个ConvNeXt Block
        self.stages = nn.ModuleList()
        
        # 计算每层的drop_path_rate（逐层递增）
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        
        cur = 0  # 当前block的全局索引，用于从dp_rates中获取drop_path_rate
        for i in range(4):  # 4个stage
            # 创建当前stage的所有blocks
            stage_blocks = []
            for j in range(depths[i]):
                block = ConvNeXtBlock(
                    dim=dims[i], 
                    drop_path=dp_rates[cur + j]
                    # 取消 V1 的 Layer Scale
                    # layer_scale_init_value=layer_scale_init_value
                )
                stage_blocks.append(block)
            
            stage = nn.Sequential(*stage_blocks)
            self.stages.append(stage)
            cur += depths[i]

        # 分类头
        self.norm = LayerNorm(dims[-1], eps=1e-6, data_format="channels_first")  # 最后一层归一化
        self.head = nn.Linear(dims[-1], num_classes)

        # 权重初始化
        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)
        
    def _init_weights(self, m):
        """权重初始化"""
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        """特征提取部分（不包括分类头）"""
        for i in range(4):
            x = self.downsample_layers[i](x)  # 下采样（包括stem）
            x = self.stages[i](x)             # 对应stage的blocks
        x = self.norm(x)  # 先进行归一化
        x = x.mean([-2, -1])  # 然后进行全局平均池化
        return x

    def forward(self, x):
        """完整前向传播"""
        x = self.forward_features(x)         # 特征提取
        # x = F.adaptive_avg_pool2d(x, (1, 1)) # 全局平均池化
        # x = torch.flatten(x, 1)              # 展平
        x = self.head(x)                     # 分类头
        return x


class ConvNeXtBlock(nn.Module): # Change to V2
    """ConvNeXt Block V2"""
    def __init__(self, dim, drop_path=0.):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()

        # V2 添加了 GRN 模块取代 Layer Scale
        self.grn = GRN(4 * dim)

        self.pwconv2 = nn.Linear(4 * dim, dim)

        # # 取消了 V1 的 Layer Scale
        # self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
        #                         requires_grad=True) if layer_scale_init_value > 0 else None

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)

        # V2 应用 GRN 模块
        x = self.grn(x)

        x = self.pwconv2(x)
        
        # # 取消 Layer Scale
        # if self.gamma is not None:
        #     x = self.gamma * x
            
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        x = input + self.drop_path(x)
        return x

# LayerNorm 参考实现
class LayerNorm(nn.Module): # Change to V2
    """LayerNorm实现（复用之前的代码）"""
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(normalized_shape), requires_grad=True)
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

# DropPath 参考实现
def drop_path(x, drop_prob: float = 0., training: bool = False): # Change to V2
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module): # Change to V2
    def __init__(self, drop_prob=None, scale_by_keep=True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        # self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training) # 这里取消了原本的scale_by_keep
    
class GRN(nn.Module): # Change to V2
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(1,2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x


# 定义不同规模的ConvNeXt模型
def convnext_tiny(num_classes: int, pretrained: bool = False):
    model = ConvNeXt(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], num_classes=num_classes)

    if pretrained:
        # 加载预训练权重
        try:
            # 直接下载预训练权重文件
            url = "https://dl.fbaipublicfiles.com/convnext/convnextv2/im1k/convnextv2_tiny_1k_224_ema.pt"
            state_dict = torch.hub.load_state_dict_from_url(url)
            if 'model' in state_dict:
                state_dict = state_dict['model']
            if num_classes != 1000:
                del state_dict['head.weight']
                del state_dict['head.bias']
            print(model.load_state_dict(state_dict, strict=False))
            print("✅ 成功加载预训练权重")
        except Exception as e:
            print(f"❌ 加载预训练权重失败: {e}")
            print("将使用随机初始化的权重继续训练")

    return model

def convnext_base(num_classes: int):
    """ConvNeXt-Base: depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024]"""
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], num_classes=num_classes)
    return model

def convnext_large(num_classes: int):
    """ConvNeXt-Large: depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536]"""
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536], num_classes=num_classes)
    return model


# 测试代码
if __name__ == "__main__":
    print("=== ConvNeXt完整网络测试 ===")
    
    # 测试1: ConvNeXt-Tiny
    print("\n1. ConvNeXt-Tiny 测试:")
    model = convnext_tiny(num_classes=1000)
    
    # 模拟ImageNet输入
    x = torch.randn(2, 3, 224, 224)
    
    # 前向传播
    with torch.no_grad():
        output = model(x)
        features = model.forward_features(x)
    
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    print(f"特征图形状: {features.shape}")
    
    # 测试2: 参数量统计
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n2. 模型参数统计:")
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")
    
    # 测试3: 不同规模模型对比
    print(f"\n3. 不同规模ConvNeXt对比:")
    models = {
        'Tiny': convnext_tiny(num_classes=1000),
        'Base': convnext_base(num_classes=1000),
        'Large': convnext_large(num_classes=1000)
    }
    
    for name, model in models.items():
        params = sum(p.numel() for p in model.parameters())
        print(f"ConvNeXt-{name:5s}: {params:>10,} 参数")
    
    # 测试4: 逐层特征图大小变化
    print(f"\n4. 特征图尺寸变化追踪:")
    model = convnext_tiny(num_classes=1000)
    x = torch.randn(1, 3, 224, 224)
    
    print(f"输入: {list(x.shape)}")
    
    # 逐层跟踪
    with torch.no_grad():
        for i in range(4):
            x = model.downsample_layers[i](x)
            print(f"经过下采样层{i}: {list(x.shape)}")
            x = model.stages[i](x)
            print(f"经过Stage{i+1}: {list(x.shape)}")
    
    print("\n✅ ConvNeXt网络构建完成！")
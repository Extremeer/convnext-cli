# ConvNeXt 项目

这是一个基于 ConvNeXt 架构的深度学习项目实现。本项目包含了 ConvNeXt-V1 和 ConvNeXt-V2 的完整实现，以及现代化的训练框架。

## 项目特点

- 🚀 同时支持 ConvNeXt-V1 和 ConvNeXt-V2 架构
- 📊 集成现代训练技巧（混合精度、MixUp、CutMix等）
- 🛠 支持多种规模的模型（Tiny、Base、Large）
- 🔧 完整的训练和推理流程
- 📈 详细的训练过程可视化
- 🎯 支持迁移学习和自定义数据集

## 项目结构

```
ConvNeXt/
├── ConvNeXtModelV1.py     # ConvNeXt-V1 模型实现
├── ConvNeXtModelV2.py     # ConvNeXt-V2 模型实现
├── train.py               # 训练主程序
├── train_flowers.py       # 花卉数据集训练示例
├── predict_flowers.py     # 花卉图像预测示例
├── LearningSample/        # 学习示例代码
│   └── ConvNeXtBlock.py  # ConvNeXt基础模块实现
└── data/                 # 数据集目录
```

## 环境要求

- Python 3.8+
- PyTorch 2.0+
- CUDA（推荐用于GPU训练）

## 主要功能

### 1. 模型架构

- **ConvNeXt-V1**：原始ConvNeXt实现，包含Layer Scale优化
- **ConvNeXt-V2**：改进版实现，移除Layer Scale，添加更多现代化设计
- 支持的模型规模：
  - Tiny: 28M参数
  - Base: 89M参数
  - Large: 198M参数

### 2. 训练框架特性

- 混合精度训练 (AMP)
- MixUp 和 CutMix 数据增强
- 标签平滑正则化
- 自适应学习率调度（Cosine/Step）
- 权重衰减优化
- 训练过程可视化

### 3. 数据增强策略

- RandomCrop
- RandomHorizontalFlip
- ColorJitter
- RandomRotation
- RandomAffine
- RandomErasing
- AutoAugment（可选）

## 使用方法

### 1. 基础训练

```python
from ConvNeXtModelV2 import convnext_tiny
from train import ConvNeXtTrainer

# 创建模型
model = convnext_tiny(num_classes=1000, pretrained=True)

# 初始化训练器
trainer = ConvNeXtTrainer(
    model,
    device='cuda',
    use_amp=True,
    use_mixup=True,
    use_cutmix=True,
    label_smoothing=0.1
)

# 设置优化器和调度器
trainer.setup_optimizer(learning_rate=1e-4, weight_decay=0.05)
trainer.setup_scheduler(scheduler_type='cosine', epochs=100)

# 开始训练
trainer.train(train_loader, val_loader, epochs=100)
```

### 2. 使用预训练模型

```python
from ConvNeXtModelV2 import convnext_tiny

# 加载预训练模型
model = convnext_tiny(num_classes=your_classes, pretrained=True)
```

### 3. 推理示例

```python
model.eval()
with torch.no_grad():
    predictions = model(images)
```

## 训练可视化

项目会自动生成训练过程的可视化图表，包括：
- 训练和验证损失曲线
- 训练和验证准确率曲线
- 学习率变化曲线

## 预训练模型

提供以下预训练模型：
- ConvNeXt-V2-Tiny (ImageNet-1K)
- ConvNeXt-V2-Base (ImageNet-1K)
- ConvNeXt-V2-Large (ImageNet-1K)

## 注意事项

- 训练时建议使用GPU加速
- 大规模模型训练需要较大显存
- 建议使用混合精度训练以提高效率
- 定期保存检查点避免训练中断
- 使用预训练模型可显著加快收敛

## 参考资料

- [ConvNeXt论文](https://arxiv.org/abs/2201.03545)
- [ConvNeXt-V2论文](https://arxiv.org/abs/2301.00808)
- [官方实现](https://github.com/facebookresearch/ConvNeXt)

## 许可证

本项目采用 MIT 许可证。 
 # ConvNeXt 项目

这是一个基于 ConvNeXt 架构的深度学习项目实现。本项目包含了 ConvNeXt-V1、ConvNeXt-V2 和 引入注意力机制的 V1 的完整实现，以及现代化的训练框架。

## 项目特点

- 🚀 支持 ConvNeXt-V1、V2 和 引入注意力机制的 V1 架构
- 📊 集成现代训练技巧（混合精度、MixUp、CutMix等）
- 🛠 支持多种规模的模型（Tiny、Base、Large）
- 🔧 完整的训练和推理流程
- 📈 详细的训练过程可视化
- 🎯 支持迁移学习和自定义数据集

## 项目结构

```
ConvNeXt/
├── Models/                # 模型实现目录
│   ├── ConvNeXtV1Model.py  # ConvNeXt-V1 实现
│   ├── ConvNeXtV2Model.py  # ConvNeXt-V2 实现
│   └── ConvNeXtV3Model.py  # ConvNeXt-V3 实现
├── Utils/                 # 工具函数目录
│   └── Train.py            # 训练核心实现
├── Scripts/               # 示例脚本目录
│   ├── TrainFlowers.py     # 花卉数据集训练示例
│   └── PredictFlowers.py   # 花卉图像预测示例
├── Data/                  # 数据集目录
├── Run/                   # 运行时文件目录
└── LearningProcedures/   # 学习过程记录目录
```

## 环境要求

- Python 3.8+
- PyTorch 2.0+
- CUDA（推荐用于GPU训练）

## 主要功能

### 1. 模型架构

- **ConvNeXt-V1**：原始ConvNeXt实现
- **ConvNeXt-V2**：改进版实现，优化了Layer Scale设计
- **引入注意力机制的 V1**：最新版本，包含更多现代化改进
- 支持的模型规模：
  - Tiny: 28M参数
  - Base: 89M参数
  - Large: 198M参数

### 2. 训练框架特性

- 混合精度训练 (AMP)
- 多种数据增强策略
  - MixUp
  - CutMix
  - RandomAugment
- 标签平滑正则化
- 自适应学习率调度
- 权重衰减优化
- 训练过程可视化

## 使用方法

### 1. 基础训练

```python
from Models.ConvNeXtV2Model import convnext_tiny
from Utils.Train import ConvNeXtTrainer

# 创建模型
model = convnext_tiny(num_classes=1000)

# 初始化训练器
trainer = ConvNeXtTrainer(
    model,
    device='cuda',
    use_amp=True,
    use_mixup=True,
    use_cutmix=True
)

# 开始训练
trainer.train(train_loader, val_loader, epochs=100)
```

### 2. 花卉数据集训练示例

```python
python Scripts/TrainFlowers.py --model v2 --size tiny --epochs 50
```

### 3. 预测示例

```python
python Scripts/PredictFlowers.py --model v2 --size tiny --image path/to/image.jpg
```

## 训练可视化

训练过程中会自动记录以下指标：
- 训练和验证损失
- 准确率曲线
- 学习率变化
- 资源使用情况

可视化数据保存在 `Run` 目录下。

## 预训练模型

提供以下预训练模型：
- ConvNeXt-V1/V2/V3-Tiny (ImageNet-1K)
- ConvNeXt-V1/V2/V3-Base (ImageNet-1K)
- ConvNeXt-V1/V2/V3-Large (ImageNet-1K)

## 注意事项

- 推荐使用GPU进行训练
- 大模型训练需要较大显存
- 建议使用混合精度训练
- 定期保存检查点
- 可利用预训练模型加速收敛

## 参考资料

- [ConvNeXt论文](https://arxiv.org/abs/2201.03545)
- [ConvNeXt-V2论文](https://arxiv.org/abs/2301.00808)

## 许可证

本项目采用 MIT 许可证。
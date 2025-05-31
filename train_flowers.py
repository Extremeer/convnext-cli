import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision import datasets, transforms

# 导入模型和训练器
# from ConvNeXtV1Model import convnext_tiny
# from ConvNeXtV2Model import convnext_tiny
from ConvNeXtV3Model import convnext_tiny_cs_attention as convnext_tiny
from train import ConvNeXtTransforms, ConvNeXtTrainer

# 定义可序列化的数据集包装类
class TransformedSubset(Dataset):
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform
        
    def __getitem__(self, idx):
        x, y = self.subset[idx]
        if self.transform:
            x = self.transform(x)
        return x, y
        
    def __len__(self):
        return len(self.subset)

# 设置随机种子保证可重复性
def set_seed(seed=42):
    import random
    import numpy as np
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def main():
    # 设置随机种子
    set_seed(42)
    
    # 配置参数
    data_root = "./data/flower_photos"  # 数据集根目录
    img_size = 224                      # 输入图像大小
    batch_size = 16#32                     # 批次大小
    num_epochs = 200                     # 训练轮数
    learning_rate = 1e-3#1e-4                # 学习率
    weight_decay = 0.05                 # 权重衰减
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 设备
    model_size = "tiny"                 # 模型大小：tiny 或 small
    pretrained = True                     # 是否使用预训练权重
    save_path = "convnext_flowers_v3.pth"  # 保存路径
    val_split = 0.15                    # 验证集比例
    test_split = 0.15                   # 测试集比例
    
    # 数据增强和预处理
    train_transform = ConvNeXtTransforms.get_train_transforms(img_size=img_size)
    val_transform = ConvNeXtTransforms.get_val_transforms(img_size=img_size)
    
    # 加载完整数据集
    full_dataset = datasets.ImageFolder(
        root=data_root,
        transform=None  # 先不应用变换，后面会根据子集应用
    )
    
    # 获取类别信息
    class_names = full_dataset.classes
    class_to_idx = full_dataset.class_to_idx
    num_classes = len(class_names)
    
    # 打印数据集信息
    print(f"数据集总大小: {len(full_dataset)}")
    print(f"类别数: {num_classes}")
    print("类别列表:")
    for i, class_name in enumerate(class_names):
        class_size = len(os.listdir(os.path.join(data_root, class_name)))
        print(f"  {i}: {class_name} ({class_size}张图片)")
    
    # 划分数据集
    total_size = len(full_dataset)
    test_size = int(total_size * test_split)
    val_size = int(total_size * val_split)
    train_size = total_size - val_size - test_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, 
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)  # 保证可重复性
    )
    
    # 应用变换
    train_dataset = TransformedSubset(train_dataset, train_transform)
    val_dataset = TransformedSubset(val_dataset, val_transform)
    test_dataset = TransformedSubset(test_dataset, val_transform)  # 测试集使用验证集的转换
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # 打印数据集划分信息
    print(f"训练集大小: {len(train_dataset)}")
    print(f"验证集大小: {len(val_dataset)}")
    print(f"测试集大小: {len(test_dataset)}")
    
    # 创建模型
    if model_size.lower() == "tiny":
        model = convnext_tiny(num_classes=num_classes)#, pretrained=pretrained)
        print("使用 ConvNeXt-Tiny 模型")
    else:
        raise ValueError(f"不支持的模型大小: {model_size}")
    
    # 创建训练器
    trainer = ConvNeXtTrainer(
        model=model,
        device=device,
        num_classes=num_classes,
        use_amp=torch.cuda.is_available(),  # 使用混合精度训练
        use_mixup=True,                     # 使用MixUp
        use_cutmix=True,                    # 使用CutMix
        label_smoothing=0.1                 # 标签平滑
    )
    
    # 设置优化器和学习率调度器
    trainer.setup_optimizer(
        learning_rate=learning_rate, 
        weight_decay=weight_decay,
        optimizer_type="adamw"
    )
    
    trainer.setup_scheduler(
        scheduler_type="cosine", 
        epochs=num_epochs,
        min_lr=1e-6
    )
    
    # 开始训练
    print(f"\n开始训练花卉分类模型...")
    best_val_acc = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=num_epochs,
        save_path=save_path,
        class_names=class_names,  # 传递类别名称
        class_to_idx=class_to_idx  # 传递类别到索引的映射
    )
    
    # 可视化训练过程
    trainer.plot_training_curves(save_path="training_curves.png")
    
    # 在测试集上评估
    print("\n在测试集上评估模型...")
    test_loss, test_acc = trainer.validate(test_loader)
    print(f"测试集结果 - Loss: {test_loss:.4f}, Accuracy: {test_acc:.2f}%")
    
    # 打印最终结果
    print(f"\n训练完成! 最佳验证准确率: {best_val_acc:.2f}%")
    print(f"测试集准确率: {test_acc:.2f}%")
    print(f"模型已保存至: {save_path}")
    print(f"类别名称已保存在模型文件中")

if __name__ == "__main__":
    main() 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
import numpy as np
import random
import time
from typing import Dict, List
import matplotlib.pyplot as plt

# ===== 1. 数据预处理和增强 =====

class ConvNeXtTransforms:
    """ConvNeXt训练和测试的数据预处理"""
    
    @staticmethod
    def get_train_transforms(img_size=224, auto_augment=True):
        """训练时的数据增强
        
        现代训练技巧：
        - RandAugment: 随机数据增强策略
        - MixUp/CutMix: 在训练循环中实现
        - RandomErasing: 随机擦除部分区域
        """
        transforms_list = [
            transforms.Resize((256, 256)),  # 稍大一点，为随机裁剪做准备
            transforms.RandomCrop(img_size, padding=4),  # 随机裁剪
            transforms.RandomHorizontalFlip(p=0.5),      # 随机水平翻转
        ]
        
        if auto_augment:
            # 添加更强的数据增强
            transforms_list.extend([
                transforms.ColorJitter(
                    brightness=0.4,    # 亮度变化
                    contrast=0.4,      # 对比度变化  
                    saturation=0.4,    # 饱和度变化
                    hue=0.1           # 色调变化
                ),
                transforms.RandomRotation(15),              # 随机旋转
                transforms.RandomAffine(0, shear=0.2),     # 随机仿射变换
            ])
        
        transforms_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet标准化
                std=[0.229, 0.224, 0.225]
            ),
            transforms.RandomErasing(p=0.25)  # 随机擦除
        ])
        
        return transforms.Compose(transforms_list)
    
    @staticmethod
    def get_val_transforms(img_size=224):
        """验证时的数据预处理（只做必要的变换）"""
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])


# ===== 2. 现代训练技巧 =====

class MixUp:
    """MixUp数据增强实现"""
    
    def __init__(self, alpha=0.2):
        self.alpha = alpha
    
    def __call__(self, x, y):
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1
        
        batch_size = x.size(0)
        index = torch.randperm(batch_size).to(x.device)
        
        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        
        return mixed_x, y_a, y_b, lam


class CutMix:
    """CutMix数据增强实现"""
    
    def __init__(self, alpha=1.0):
        self.alpha = alpha
    
    def __call__(self, x, y):
        lam = np.random.beta(self.alpha, self.alpha)
        
        batch_size = x.size(0)
        index = torch.randperm(batch_size).to(x.device)
        
        _, _, H, W = x.shape
        cut_rat = np.sqrt(1. - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)
        
        # 随机选择裁剪区域
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
        
        x[:, :, bby1:bby2, bbx1:bbx2] = x[index, :, bby1:bby2, bbx1:bbx2]
        
        # 调整lambda
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
        y_a, y_b = y, y[index]
        
        return x, y_a, y_b, lam


class LabelSmoothing(nn.Module):
    """标签平滑正则化"""
    
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
    
    def forward(self, pred, target):
        log_prob = F.log_softmax(pred, dim=-1)
        nll_loss = -log_prob.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -log_prob.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


# ===== 3. 训练器主类 =====

class ConvNeXtTrainer:
    """ConvNeXt训练器 - 集成现代训练技巧"""
    
    def __init__(self, 
                 model, 
                 device='cuda',
                 num_classes=1000,
                 use_amp=True,           # 混合精度训练
                 use_mixup=True,         # MixUp增强
                 use_cutmix=True,        # CutMix增强  
                 label_smoothing=0.1):   # 标签平滑
        
        self.model = model.to(device)
        self.device = device
        self.num_classes = num_classes
        self.use_amp = use_amp
        
        # 现代训练技巧
        self.mixup = MixUp(alpha=0.2) if use_mixup else None
        self.cutmix = CutMix(alpha=1.0) if use_cutmix else None
        self.use_augment_prob = 0.5  # 使用数据增强的概率
        
        # 损失函数
        if label_smoothing > 0:
            self.criterion = LabelSmoothing(smoothing=label_smoothing)
        else:
            self.criterion = nn.CrossEntropyLoss()
        
        # 混合精度训练
        if self.use_amp:
            self.scaler = torch.amp.GradScaler('cuda')
        
        # 训练记录
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        self.learning_rates = []
    
    def setup_optimizer(self, learning_rate=1e-4, weight_decay=0.05, optimizer_type='adamw'):
        """设置优化器 - ConvNeXt官方推荐AdamW"""
        
        # 参数分组：对不同类型参数应用不同的weight_decay
        param_groups = self._get_param_groups(weight_decay)
        
        if optimizer_type.lower() == 'adamw':
            self.optimizer = optim.AdamW(
                param_groups,
                lr=learning_rate,
                betas=(0.9, 0.999),
                eps=1e-8
            )
        elif optimizer_type.lower() == 'sgd':
            self.optimizer = optim.SGD(
                param_groups,
                lr=learning_rate,
                momentum=0.9,
                nesterov=True
            )
        else:
            raise ValueError(f"不支持的优化器类型: {optimizer_type}")
    
    def _get_param_groups(self, weight_decay):
        """参数分组 - 不对bias和norm层应用weight_decay"""
        decay_params = []
        no_decay_params = []
        
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            
            # 不对bias和归一化层参数应用weight decay
            if len(param.shape) == 1 or name.endswith('.bias') or 'norm' in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)
        
        return [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.}
        ]
    
    def setup_scheduler(self, scheduler_type='cosine', epochs=100, min_lr=1e-6):
        """设置学习率调度器"""
        
        if scheduler_type == 'cosine':
            self.scheduler = CosineAnnealingLR(
                self.optimizer, 
                T_max=epochs,
                eta_min=min_lr
            )
        elif scheduler_type == 'step':
            self.scheduler = StepLR(
                self.optimizer,
                step_size=30,
                gamma=0.1
            )
        else:
            raise ValueError(f"不支持的调度器类型: {scheduler_type}")
    
    def train_epoch(self, train_loader, epoch):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            # 应用MixUp或CutMix
            if self.mixup is not None or self.cutmix is not None:
                use_augment = random.random() < self.use_augment_prob
                if use_augment:
                    if random.random() < 0.5 and self.mixup is not None:
                        data, target_a, target_b, lam = self.mixup(data, target)
                    elif self.cutmix is not None:
                        data, target_a, target_b, lam = self.cutmix(data, target)
                    mixed_target = True
                else:
                    mixed_target = False
            else:
                mixed_target = False
            
            self.optimizer.zero_grad()
            
            # 前向传播（使用混合精度）
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    output = self.model(data)
                    if mixed_target:
                        loss = lam * self.criterion(output, target_a) + (1 - lam) * self.criterion(output, target_b)
                    else:
                        loss = self.criterion(output, target)
                
                # 反向传播
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                output = self.model(data)
                if mixed_target:
                    loss = lam * self.criterion(output, target_a) + (1 - lam) * self.criterion(output, target_b)
                else:
                    loss = self.criterion(output, target)
                
                loss.backward()
                self.optimizer.step()
            
            # 统计
            total_loss += loss.item()
            if not mixed_target:  # 只在非混合标签时计算准确率
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)
            
            # 打印进度
            if batch_idx % 100 == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}/{len(train_loader)}, '
                      f'Loss: {loss.item():.4f}, LR: {self.optimizer.param_groups[0]["lr"]:.6f}')
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100. * correct / total if total > 0 else 0
        
        self.train_losses.append(avg_loss)
        self.train_accs.append(accuracy)
        self.learning_rates.append(self.optimizer.param_groups[0]['lr'])
        
        return avg_loss, accuracy
    
    def validate(self, val_loader):
        """验证模式"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                if self.use_amp:
                    with torch.cuda.amp.autocast():
                        output = self.model(data)
                        loss = F.cross_entropy(output, target)  # 验证时不用标签平滑
                else:
                    output = self.model(data)
                    loss = F.cross_entropy(output, target)
                
                total_loss += loss.item()
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)
        
        avg_loss = total_loss / len(val_loader)
        accuracy = 100. * correct / total
        
        self.val_losses.append(avg_loss)
        self.val_accs.append(accuracy)
        
        return avg_loss, accuracy
    
    def train(self, train_loader, val_loader, epochs, save_path='convnext_best.pth', class_names=None, class_to_idx=None):
        """完整训练流程
        
        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            epochs: 训练轮数
            save_path: 模型保存路径
            class_names: 类别名称列表（可选）
            class_to_idx: 类别到索引的映射（可选）
        """
        best_val_acc = 0
        
        print(f"开始训练ConvNeXt，共{epochs}个epoch")
        print(f"模型参数量: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"使用设备: {self.device}")
        print(f"混合精度: {self.use_amp}")
        print("-" * 60)
        
        for epoch in range(epochs):
            start_time = time.time()
            
            # 训练
            train_loss, train_acc = self.train_epoch(train_loader, epoch)
            
            # 验证
            val_loss, val_acc = self.validate(val_loader)
            
            # 更新学习率
            self.scheduler.step()
            
            epoch_time = time.time() - start_time
            
            # 打印结果
            print(f"Epoch {epoch+1}/{epochs} ({epoch_time:.1f}s)")
            print(f"Train: Loss={train_loss:.4f}, Acc={train_acc:.2f}%")
            print(f"Val:   Loss={val_loss:.4f}, Acc={val_acc:.2f}%")
            print(f"LR: {self.optimizer.param_groups[0]['lr']:.6f}")
            print("-" * 60)
            
            # 保存最佳模型
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                save_dict = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'best_val_acc': best_val_acc,
                    'train_losses': self.train_losses,
                    'val_losses': self.val_losses,
                    'train_accs': self.train_accs,
                    'val_accs': self.val_accs
                }
                
                # 添加类别信息（如果提供）
                if class_names is not None:
                    save_dict['class_names'] = class_names
                if class_to_idx is not None:
                    save_dict['class_to_idx'] = class_to_idx
                    
                torch.save(save_dict, save_path)
                print(f"✅ 新的最佳模型已保存! Val Acc: {val_acc:.2f}%")
        
        print(f"\n🎉 训练完成! 最佳验证准确率: {best_val_acc:.2f}%")
        return best_val_acc
    
    def plot_training_curves(self, save_path='training_curves.png'):
        """绘制训练曲线并保存到文件，不展示"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        # 损失曲线
        axes[0, 0].plot(self.train_losses, label='Train Loss')
        axes[0, 0].plot(self.val_losses, label='Val Loss')
        axes[0, 0].set_title('Loss Curves')
        axes[0, 0].legend()
        # 准确率曲线
        axes[0, 1].plot(self.train_accs, label='Train Acc')
        axes[0, 1].plot(self.val_accs, label='Val Acc')
        axes[0, 1].set_title('Accuracy Curves')
        axes[0, 1].legend()
        # 学习率曲线
        axes[1, 0].plot(self.learning_rates)
        axes[1, 0].set_title('Learning Rate Schedule')
        # 验证准确率放大
        axes[1, 1].plot(self.val_accs, 'g-', linewidth=2)
        axes[1, 1].set_title('Validation Accuracy Detail')
        axes[1, 1].grid(True)
        plt.tight_layout()
        fig.savefig(save_path)
        plt.close(fig)


# ===== 4. 快速测试框架 =====

def create_dummy_dataset(num_samples=1000, num_classes=10, img_size=224):
    """创建虚拟数据集用于快速测试"""
    
    class DummyDataset(Dataset):
        def __init__(self, num_samples, num_classes, transform=None):
            self.num_samples = num_samples
            self.num_classes = num_classes
            self.transform = transform
        
        def __len__(self):
            return self.num_samples
        
        def __getitem__(self, idx):
            # 随机生成图像和标签
            image = torch.randn(3, img_size, img_size)
            label = torch.randint(0, self.num_classes, (1,)).item()
            
            if self.transform:
                image = transforms.ToPILImage()(image)
                image = self.transform(image)
            
            return image, label
    
    return DummyDataset(num_samples, num_classes)


def quick_training_test():
    """快速训练测试"""
    print("=== ConvNeXt训练框架快速测试 ===\n")
    
    # 导入之前实现的ConvNeXt
    from ConvNeXtModel import convnext_tiny  # 假设之前的模型在这个文件中
    
    # 设置参数
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_classes = 10
    batch_size = 8
    img_size = 224
    
    # 创建模型
    model = convnext_tiny(num_classes=num_classes)
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 创建数据
    train_transform = ConvNeXtTransforms.get_train_transforms(img_size)
    val_transform = ConvNeXtTransforms.get_val_transforms(img_size)
    
    train_dataset = create_dummy_dataset(100, num_classes)
    val_dataset = create_dummy_dataset(50, num_classes)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # 创建训练器
    trainer = ConvNeXtTrainer(
        model=model,
        device=device,
        num_classes=num_classes,
        use_amp=torch.cuda.is_available(),
        use_mixup=True,
        use_cutmix=True,
        label_smoothing=0.1
    )
    
    # 设置优化器和调度器
    trainer.setup_optimizer(learning_rate=1e-4, weight_decay=0.05)
    trainer.setup_scheduler(scheduler_type='cosine', epochs=5)
    
    # 训练几个epoch测试
    print("开始快速训练测试...")
    trainer.train(train_loader, val_loader, epochs=100)
    
    print("\n✅ 训练框架测试完成!")


if __name__ == "__main__":
    quick_training_test()
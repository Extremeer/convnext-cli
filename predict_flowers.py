import os
import argparse
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import platform
from torchvision import transforms
from datetime import datetime

# 设置matplotlib中文字体
def set_matplotlib_chinese_font():
    """设置matplotlib中文字体"""
    system = platform.system()
    if system == "Windows":
        plt.rcParams['font.sans-serif'] = ['PingFang SC']  # 中文黑体
    elif system == "Linux":
        plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei']  # 文泉驿微米黑
    elif system == "Darwin":
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # macOS系统字体
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 在程序开始时设置字体
set_matplotlib_chinese_font()

# 导入模型
from ConvNeXtV1Model import convnext_tiny as convnext_tinyv1
from ConvNeXtV2Model import convnext_tiny as convnext_tinyv2
from ConvNeXtV3Model import convnext_tiny_cs_attention as convnext_tinyv3
from ConvNeXtV1Model import convnext_tiny as convnext_tinyv4
from train import ConvNeXtTransforms

def load_model(model_path, device='cuda', model_type='v2'):
    """加载训练好的模型"""
    # 加载检查点
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    
    # 获取类别信息
    class_names = checkpoint.get('class_names', ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips'])
    num_classes = len(class_names)
    
    # 创建模型
    if model_type == 'v1':
        model = convnext_tinyv1(num_classes=num_classes)
    elif model_type == 'v2':
        model = convnext_tinyv2(num_classes=num_classes)
    elif model_type == 'v3':
        model = convnext_tinyv3(num_classes=num_classes)
    elif model_type == 'v4':
        model = convnext_tinyv4(num_classes=num_classes)
    else:
        raise ValueError(f"不支持的模型版本: {model_type}")
    
    # 加载权重
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 移至指定设备
    model = model.to(device)
    model.eval()
    
    return model, checkpoint.get('best_val_acc', 0), class_names

def predict_image(image_path, model, class_names, device='cuda'):
    """对单张图片进行预测"""
    # 图像预处理
    transform = ConvNeXtTransforms.get_val_transforms(img_size=224)
    
    # 读取图像
    image = Image.open(image_path).convert('RGB')
    original_image = image.copy()
    
    # 应用预处理
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # 预测
    with torch.no_grad():
        outputs = model(image_tensor)
        probs = F.softmax(outputs, dim=1)
        
        # 获取所有预测结果
        probs = probs[0]
        predictions = []
        for i, prob in enumerate(probs):
            class_name = class_names[i]
            predictions.append((class_name, prob.item() * 100))
        
        # 按概率降序排序
        predictions.sort(key=lambda x: x[1], reverse=True)
    
    return original_image, predictions

def display_prediction(image, predictions, save_path=None):
    """显示预测结果并可选保存"""
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title("输入图像")
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    classes = [p[0] for p in predictions]
    scores = [p[1] for p in predictions]
    
    # 水平条形图显示概率
    bars = plt.barh(range(len(predictions)), scores, color='skyblue')
    plt.yticks(range(len(predictions)), classes)
    plt.xlabel('置信度 (%)')
    plt.title('预测结果')
    plt.xlim(0, 100)
    
    # 添加具体数值标签
    for i, bar in enumerate(bars):
        plt.text(min(bar.get_width() + 1, 95), 
                 bar.get_y() + bar.get_height()/2, 
                 f"{scores[i]:.1f}%", 
                 va='center')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    plt.show()

def save_prediction_text(predictions, image_path, save_path):
    """保存预测结果到文本文件"""
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(f"预测时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"输入图像: {image_path}\n\n")
        f.write("预测结果:\n")
        for i, (class_name, prob) in enumerate(predictions):
            f.write(f"{i+1}. {class_name}: {prob:.2f}%\n")

def main():
    parser = argparse.ArgumentParser(description='花卉分类预测')
    parser.add_argument('--image', type=str, required=True, help='图像路径')
    parser.add_argument('--model', type=str, help='模型路径')
    parser.add_argument('--cpu', action='store_true', help='使用CPU进行推理')
    parser.add_argument('--model_type', type=str, default='v2', choices=['v1', 'v2','v3','v4'], help='模型版本')
    args = parser.parse_args()
    
    # 如果没有指定模型路径，根据模型版本自动选择
    if args.model is None:
        args.model = f'convnext_flowers_{args.model_type}.pth'
    
    # 创建run文件夹
    run_dir = "Run"
    os.makedirs(run_dir, exist_ok=True)
    
    # 生成时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 设置设备
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    
    # 加载模型
    try:
        model, best_val_acc, class_names = load_model(args.model, device, args.model_type)
        print(f"加载 {args.model_type} 模型成功，使用权重文件: {args.model}")
        print(f"最佳验证准确率: {best_val_acc:.2f}%")
        print(f"类别: {', '.join(class_names)}")
    except (RuntimeError, FileNotFoundError) as e:
        print(f"错误：加载模型失败。请确保使用的权重文件 ({args.model}) 与模型版本 ({args.model_type}) 匹配。")
        print(f"建议：")
        print(f"- 对于 v1 模型，使用 convnext_flowers_v1.pth")
        print(f"- 对于 v2 模型，使用 convnext_flowers_v2.pth")
        return
    
    # 预测图像
    image, predictions = predict_image(args.image, model, class_names, device)
    
    # 设置保存路径
    result_image_path = os.path.join(run_dir, f"prediction_{timestamp}.png")
    result_text_path = os.path.join(run_dir, f"prediction_{timestamp}.txt")
    
    # 打印结果
    print("\n预测结果:")
    for i, (class_name, prob) in enumerate(predictions):
        print(f"{i+1}. {class_name}: {prob:.2f}%")
    
    # 保存结果
    display_prediction(image, predictions, save_path=result_image_path)
    save_prediction_text(predictions, args.image, result_text_path)
    
    print(f"\n结果已保存到:")
    print(f"图像结果: {result_image_path}")
    print(f"文本结果: {result_text_path}")

if __name__ == "__main__":
    main()
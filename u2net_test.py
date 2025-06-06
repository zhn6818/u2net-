import os
import glob
import torch
import numpy as np
from PIL import Image
from skimage import io
import torchvision.transforms as transforms
from torch.autograd import Variable

from model import U2NET
from model import U2NETP
from model import U2NET_GRAIN
from model import U2NETP_GRAIN
from data_loader import RescaleT
from data_loader import ToTensorLab
from data_loader import MultiChannelToTensorLab

pred_size = 512

def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)
    dn = (d-mi)/(ma-mi)
    return dn

def get_device():
    """获取可用的设备类型：CUDA、MPS 或 CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')

def read_image(image_path):
    """读取图片并进行基础处理"""
    image = io.imread(image_path)
    if len(image.shape) == 2:
        image = image[:,:,np.newaxis]
    return image

def generate_color_map(num_channels):
    """
    为多通道分割生成颜色映射
    Args:
        num_channels: 通道数量
    Returns:
        color_map: 颜色映射字典，通道索引到RGB颜色值的映射
    """
    color_map = {}
    # 通道0（通常是背景）设为黑色
    color_map[0] = (0, 0, 0)
    
    # 为其他通道生成随机颜色
    np.random.seed(2)  # 使用固定种子以确保颜色一致性
    for i in range(1, num_channels):
        # 生成明亮的颜色，避免太暗的颜色
        r = np.random.randint(50, 255)
        g = np.random.randint(50, 255)
        b = np.random.randint(50, 255)
        color_map[i] = (r, g, b)
    
    return color_map

def visualize_multi_channel(pred_tensor, color_map):
    """
    将多通道预测结果可视化为彩色图像
    Args:
        pred_tensor: 预测的多通道张量 [C, H, W]
        color_map: 通道到颜色的映射
    Returns:
        vis_image: RGB可视化图像
    """
    # 获取形状
    num_channels, height, width = pred_tensor.shape
    
    # 创建RGB输出图像
    vis_image = np.zeros((height, width, 3), dtype=np.uint8)
    
    # 对每个通道应用颜色映射
    for c in range(num_channels):
        # 获取通道预测
        channel_pred = pred_tensor[c].cpu().data.numpy()
        # 获取通道颜色
        color = color_map[c]
        
        # 对每个颜色通道应用预测结果
        for i in range(3):  # RGB三个通道
            vis_image[:, :, i] = vis_image[:, :, i] + (channel_pred * color[i]).astype(np.uint8)
    
    return vis_image

def inference_folder(image_dir, model_path, output_dir, model_type='u2net', num_channels=1):
    """
    对文件夹中的所有图片进行推理
    Args:
        image_dir: 输入图片文件夹路径
        model_path: 模型权重文件路径
        output_dir: 输出目录
        model_type: 使用的模型类型，'u2net', 'u2netp', 'u2net_grain', 'u2netp_grain'
        num_channels: 分割输出的通道数，默认为1（单通道分割）
    """
    # 设置设备
    device = get_device()
    print(f"Using device: {device}")
    print(f"Number of output channels: {num_channels}")

    # 加载模型
    if model_type == 'u2net':
        print("...load U2NET---173.6 MB")
        net = U2NET(3, num_channels)
    elif model_type == 'u2netp':
        print("...load U2NETP---4.7 MB")
        net = U2NETP(3, num_channels)
    elif model_type == 'u2net_grain':
        print("...load U2NET_GRAIN---optimized for grain boundary segmentation")
        net = U2NET_GRAIN(3, num_channels)
    elif model_type == 'u2netp_grain':
        print("...load U2NETP_GRAIN---lightweight grain boundary segmentation")
        net = U2NETP_GRAIN(3, num_channels)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # 加载模型权重
    net.load_state_dict(torch.load(model_path, map_location=device))
    net.to(device)
    net.eval()

    # 根据通道数选择合适的预处理方法
    if num_channels == 1:
        transform = transforms.Compose([
            RescaleT(pred_size),
            ToTensorLab(flag=0)
        ])
    else:
        transform = transforms.Compose([
            RescaleT(pred_size),
            MultiChannelToTensorLab(flag=0, num_channels=num_channels)
        ])

    # 生成颜色映射（用于多通道可视化）
    color_map = generate_color_map(num_channels)

    # 获取所有图片文件
    img_name_list = glob.glob(os.path.join(image_dir, '*.*'))
    # 支持的图片格式
    supported_formats = ['.jpg', '.jpeg', '.png', '.bmp']
    img_name_list = [f for f in img_name_list if os.path.splitext(f)[1].lower() in supported_formats]
    
    print(f"Found {len(img_name_list)} images in {image_dir}")

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 处理每张图片
    for image_path in img_name_list:
        print(f"Processing {image_path}")
        try:
            # 读取和处理图像
            image = read_image(image_path)
            
            # 准备标签（虚拟标签，推理时不会使用）
            if num_channels == 1:
                label = np.zeros(image.shape[0:2])
                label = label[:,:,np.newaxis]
            else:
                label = np.zeros((image.shape[0], image.shape[1], num_channels))
            
            # 准备输入数据
            sample = {'imidx': np.array([0]), 'image': image, 'label': label}
            sample = transform(sample)
            inputs_test = sample['image']
            inputs_test = inputs_test.unsqueeze(0)
            inputs_test = inputs_test.type(torch.FloatTensor)
            inputs_test = Variable(inputs_test).to(device)

            # 推理 - 根据模型类型处理不同的输出
            with torch.no_grad():
                outputs = net(inputs_test)
                
                if model_type in ['u2net', 'u2netp']:
                    # 原始U2NET模型有7个输出
                    d1, d2, d3, d4, d5, d6, d7 = outputs
                    pred = d1  # 不再只取第一个通道
                    # 清理内存
                    del d2, d3, d4, d5, d6, d7
                else:
                    # U2NET_GRAIN模型有6个输出 (d0-d5)
                    d1, d2, d3, d4, d5, d6 = outputs
                    pred = d1  # 不再只取第一个通道
                    # 清理内存
                    del d2, d3, d4, d5, d6
                
                # 对每个通道分别归一化
                for c in range(num_channels):
                    pred[:,c,:,:] = normPRED(pred[:,c,:,:])

                # 处理预测结果
                predict = pred.squeeze()  # 移除批次维度，现在形状是 [C, H, W]
                
                # 处理单通道和多通道的情况
                if num_channels == 1:
                    # 单通道情况（与原代码相同）
                    predict_np = predict.cpu().data.numpy()
                    im = Image.fromarray(predict_np*255).convert('RGB')
                else:
                    # 多通道情况（将多通道预测可视化为彩色图像）
                    vis_image = visualize_multi_channel(predict, color_map)
                    im = Image.fromarray(vis_image)
                
                # 调整回原始图像大小
                image = io.imread(image_path)
                imo = im.resize((image.shape[1], image.shape[0]), resample=Image.BILINEAR)

                # 保存结果
                img_name = os.path.splitext(os.path.basename(image_path))[0]
                imo.save(os.path.join(output_dir, f"{img_name}.png"))

                # 为每个通道单独保存结果（可选）
                if num_channels > 1:
                    # 创建通道特定的输出目录
                    channels_dir = os.path.join(output_dir, "channels")
                    os.makedirs(channels_dir, exist_ok=True)
                    
                    for c in range(num_channels):
                        channel_pred = predict[c].cpu().data.numpy()
                        channel_im = Image.fromarray((channel_pred*255).astype(np.uint8))
                        channel_im = channel_im.resize((image.shape[1], image.shape[0]), resample=Image.BILINEAR)
                        channel_im.save(os.path.join(channels_dir, f"{img_name}_channel_{c}.png"))

        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
            continue

    print(f"Processing completed. Results saved in {output_dir}")

def inference_single_image(image_path, model_path, output_path, model_type='u2net', num_channels=1):
    """
    对单张图片进行推理
    Args:
        image_path: 输入图片路径
        model_path: 模型权重文件路径
        output_path: 输出图片路径
        model_type: 使用的模型类型
        num_channels: 分割输出的通道数，默认为1（单通道分割）
    """
    # 设置设备
    device = get_device()
    print(f"Using device: {device}")
    print(f"Number of output channels: {num_channels}")

    # 加载模型
    if model_type == 'u2net':
        print("...load U2NET---173.6 MB")
        net = U2NET(3, num_channels)
    elif model_type == 'u2netp':
        print("...load U2NETP---4.7 MB")
        net = U2NETP(3, num_channels)
    elif model_type == 'u2net_grain':
        print("...load U2NET_GRAIN---optimized for grain boundary segmentation")
        net = U2NET_GRAIN(3, num_channels)
    elif model_type == 'u2netp_grain':
        print("...load U2NETP_GRAIN---lightweight grain boundary segmentation")
        net = U2NETP_GRAIN(3, num_channels)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # 加载模型权重
    net.load_state_dict(torch.load(model_path, map_location=device))
    net.to(device)
    net.eval()

    # 根据通道数选择合适的预处理方法
    if num_channels == 1:
        transform = transforms.Compose([
            RescaleT(pred_size),
            ToTensorLab(flag=0)
        ])
    else:
        transform = transforms.Compose([
            RescaleT(pred_size),
            MultiChannelToTensorLab(flag=0, num_channels=num_channels)
        ])

    # 生成颜色映射（用于多通道可视化）
    color_map = generate_color_map(num_channels)

    print(f"Processing {image_path}")
    
    # 读取和处理图像
    image = read_image(image_path)
    
    # 准备标签（虚拟标签，推理时不会使用）
    if num_channels == 1:
        label = np.zeros(image.shape[0:2])
        label = label[:,:,np.newaxis]
    else:
        label = np.zeros((image.shape[0], image.shape[1], num_channels))
    
    # 准备输入数据
    sample = {'imidx': np.array([0]), 'image': image, 'label': label}
    sample = transform(sample)
    inputs_test = sample['image']
    inputs_test = inputs_test.unsqueeze(0)
    inputs_test = inputs_test.type(torch.FloatTensor)
    inputs_test = Variable(inputs_test).to(device)

    # 推理
    with torch.no_grad():
        outputs = net(inputs_test)
        
        if model_type in ['u2net', 'u2netp']:
            # 原始U2NET模型有7个输出
            d1, d2, d3, d4, d5, d6, d7 = outputs
            pred = d1  # 不再只取第一个通道
            del d2, d3, d4, d5, d6, d7
        else:
            # U2NET_GRAIN模型有6个输出 (d0-d5)
            d1, d2, d3, d4, d5, d6 = outputs
            pred = d1  # 不再只取第一个通道
            del d2, d3, d4, d5, d6
        
        # 对每个通道分别归一化
        for c in range(num_channels):
            pred[:,c,:,:] = normPRED(pred[:,c,:,:])

        # 处理预测结果
        predict = pred.squeeze()  # 移除批次维度，现在形状是 [C, H, W] 或 [H, W]
        
        # 处理单通道和多通道的情况
        if num_channels == 1:
            # 单通道情况（与原代码相同）
            predict_np = predict.cpu().data.numpy()
            im = Image.fromarray(predict_np*255).convert('RGB')
        else:
            # 多通道情况（将多通道预测可视化为彩色图像）
            vis_image = visualize_multi_channel(predict, color_map)
            im = Image.fromarray(vis_image)
        
        # 调整回原始图像大小
        original_image = io.imread(image_path)
        imo = im.resize((original_image.shape[1], original_image.shape[0]), resample=Image.BILINEAR)

        # 保存结果
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        imo.save(output_path)
        print(f"Result saved to {output_path}")
        
        # 为每个通道单独保存结果（可选）
        if num_channels > 1:
            # 创建通道特定的输出目录
            output_dir = os.path.dirname(output_path)
            channels_dir = os.path.join(output_dir, "channels")
            os.makedirs(channels_dir, exist_ok=True)
            
            # 获取输出文件名（不含扩展名）
            base_name = os.path.splitext(os.path.basename(output_path))[0]
            
            for c in range(num_channels):
                channel_pred = predict[c].cpu().data.numpy()
                channel_im = Image.fromarray((channel_pred*255).astype(np.uint8))
                channel_im = channel_im.resize((original_image.shape[1], original_image.shape[0]), resample=Image.BILINEAR)
                channel_im.save(os.path.join(channels_dir, f"{base_name}_channel_{c}.png"))

if __name__ == "__main__":
    # 示例使用多通道模型
    model_type = 'u2net_grain'  # 可选: 'u2net', 'u2netp', 'u2net_grain', 'u2netp_grain'
    image_dir = 'datasetv2/testimg/'
    # 修改为对应的多通道模型路径
    model_path = 'saved_models/u2net_grain_3ch/u2net_grain_best_acc_0.9856_epoch_12.pth'
    output_dir = 'test_results_multi_3ch/'
    num_channels = 3  # 设置为多通道模型的通道数
    
    # 示例使用
    # model_type = 'u2net'  # 可选: 'u2net', 'u2netp', 'u2net_grain', 'u2netp_grain'
    # image_dir = '/Volumes/data1/JH/projects/JLD_imgprocess/dataset/img/'
    # model_path = 'saved_models/u2net/u2net_best_acc_0.9056_epoch_35.pth'
    # output_dir = 'test_results/'
    # num_channels = 1  # 单通道分割
    
    # 批量推理
    inference_folder(
        image_dir=image_dir,
        model_path=model_path,
        output_dir=output_dir,
        model_type=model_type,
        num_channels=num_channels
    )
    
    # 或者单张图片推理
    # inference_single_image(
    #     image_path='path/to/single/image.jpg',
    #     model_path=model_path,
    #     output_path='output/result.png',
    #     model_type=model_type,
    #     num_channels=num_channels
    # ) 
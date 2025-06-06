# U2NET+ 图像分割项目

这是一个基于U2NET架构的图像分割项目，支持单通道和多通道分割任务。通过简单的配置修改，可以灵活地切换不同分割任务。

## 特性

- 支持单通道分割（例如前景/背景分割）
- 支持多通道分割（例如语义分割，每个通道代表一个类别）
- 多种数据增强方法（色彩抖动、随机划痕、最大值滤波等）
- 多种U2NET网络结构变体
- 自动保存最佳模型和阶段性检查点

## 使用方法

### 1. 准备数据

训练数据应按以下格式组织：
- 创建 `train.txt` 文件，每行包含一对图像和标签的路径，用空格分隔
- 对于单通道分割，标签应为单通道灰度图像
- 对于多通道分割，标签应包含N个通道，每个通道对应一个类别

示例 `train.txt`:
```
/path/to/image1.jpg /path/to/label1.png
/path/to/image2.jpg /path/to/label2.png
...
```

### 2. 配置训练参数

在 `u2net_train.py` 文件中，修改以下参数以适应您的需求：

```python
# -------- 配置参数 ---------
model_name = 'u2net_grain'  # 可选: 'u2net', 'u2netp', 'u2net_grain', 'u2netp_grain'
num_channels = 1  # 分割输出通道数 (1: 单通道分割, >1: 多通道分割)
batch_size_train = 2  # 训练批次大小
learning_rate = 0.001  # 学习率
epoch_num = 100000  # 最大训练轮次
train_txt_path = "./dataset/train.txt"  # 训练数据文件列表
save_freq = 2000  # 模型保存频率(迭代次数)
```

### 3. 开始训练

```bash
python u2net_train.py
```

### 多通道分割设置

要使用多通道分割，只需修改 `num_channels` 参数即可：

```python
num_channels = 3  # 设置为所需的通道数
```

例如，如果您有一个3类分割任务（背景、前景1、前景2），设置 `num_channels = 3`。

确保您的标签数据格式正确：
- 对于单通道分割：标签是单通道图像，像素值表示目标区域（通常为255或1）
- 对于多通道分割：标签是多通道图像，每个通道对应一个类别（像素值为1表示该位置属于该类别）

## 模型说明

本项目支持以下模型：

- `u2net`: 原始U2NET模型，参数量较大
- `u2netp`: 轻量级U2NET模型，适合资源受限场景
- `u2net_grain`: 针对纹理分割优化的U2NET模型
- `u2netp_grain`: 轻量级纹理分割U2NET模型

## 保存的模型

训练过程中，模型将按以下规则保存：

1. 每 `save_freq` 次迭代保存一次检查点
2. 每5个epoch保存一次阶段性模型
3. 当达到新的最佳准确率时，保存最佳模型

所有模型保存在 `saved_models/{model_name}_{num_channels}ch/` 目录下。例如：
- 单通道U2NET模型会保存在 `saved_models/u2net_1ch/`
- 3通道U2NET_GRAIN模型会保存在 `saved_models/u2net_grain_3ch/`

这种命名方式可以帮助您清晰区分不同模型和不同通道数的训练结果。

## 环境要求

- Python 3.6+
- PyTorch 1.7+
- NumPy
- scikit-image
- OpenCV
- Matplotlib（可选，用于可视化） 
markdown

<div align="center">
# 基于改进 YOLOv5 的红外小目标检测系统 🚀

本项目针对红外图像中小目标检测的难点（尺寸小、对比度低、背景复杂），对 YOLOv5 模型进行了**多策略融合改进**，显著提升了检测精度和鲁棒性。同时，项目提供了一个**基于 Tkinter 的可视化交互界面**，方便用户进行模型推理和结果分析。

</div>

## 📚 项目亮点

- **🎯 改进模型**：在 YOLOv5s 基线模型上集成了三种有效改进策略：
  - **NWD (Normalized Wasserstein Distance) 损失函数**：优化小目标边界框回归，缓解传统 IoU 对小目标不敏感的问题。
  - **CoordConv (Coordinate Convolution) 卷积层**：增强模型对空间位置的感知能力，提升小目标定位精度。
  - **SimAM (Similarity-based Attention Module) 注意力机制**：无参数注意力模块，增强目标特征，有效抑制复杂背景干扰。
- **📊 性能提升**：改进后的模型在红外小目标检测任务上取得了显著效果：
  - **mAP@0.5 提升 8.8%** (从 0.863 到 0.939)
  - **精确率 (Precision) 提升 11.8%** (从 0.838 到 0.937)
  - **召回率 (Recall) 提升 6.0%** (从 0.853 到 0.904)
- **🖥️ 用户友好界面**：基于 `Tkinter` 开发了可视化工具，支持图片/视频推理、参数调节和结果展示，无需编写代码即可体验改进模型的强大功能。
- **⚙️ 数据处理自动化**：提供 `DataLoader.py` 脚本，自动将红外图像掩码转换为 YOLOv5 格式的标签，并完成数据集划分。

## 🔧 安装与使用

### 环境要求

- Python >= 3.8.0
- PyTorch >= 1.8.0
- 其他依赖项请见 `requirements.txt`

```bash
# 克隆项目
https://github.com/BrainStormings/yolov5_Demo.git
cd infrared-small-target-detection
```

### 安装依赖

```bash
pip install -r requirements.txt
```

### 数据准备与预处理

本项目提供了 DataLoader.py 脚本，用于将包含图像和二值掩码的数据集自动转换为 YOLOv5 格式。

```bash
python DataLoader.py \
  --images_dir /path/to/images \
  --masks_dir /path/to/masks \
  --output_dir /path/to/dataset \
  --train_ratio 0.7 \
  --val_ratio 0.2
```

### 模型训练

使用改进后的模型配置文件 models/yolov5s_improved.yaml 进行训练。

```bash
python train.py \
  --data /path/to/dataset/data.yaml \
  --cfg models/yolov5s_improved.yaml \
  --weights yolov5s.pt \
  --epochs 100 \
  --batch-size 16 \
  --device 0
```

### 模型推理

你可以通过命令行或使用我们提供的 GUI 界面进行推理。

命令行推理

```bash
# 对单张图片进行检测
python detect.py --weights runs/train/exp/weights/best.pt --source img.jpg
```

# 对视频文件进行检测

python detect.py --weights runs/train/exp/weights/best.pt --source video.mp4

### GUI 界面推理

启动可视化工具，轻松进行模型推理。

python gui.py
🧩 改进模块详解

1. NWD 损失函数
   传统 IoU 对小目标的位置偏移非常敏感。NWD 将边界框建模为二维高斯分布，并计算其 Wasserstein 距离，从而更准确地衡量预测框与真实框的相似度，有效提升小目标定位精度。

2. CoordConv 卷积层
   标准卷积是平移等变的，这导致其丢失了绝对位置信息。CoordConv 通过在输入特征图上添加两个坐标通道 (x, y)，显式地提供了位置信息，使模型能够更好地学习目标的坐标相关性。

3. SimAM 注意力机制
   SimAM 是一种无参数的注意力模块，通过定义能量函数来为每个神经元计算一个3D注意力权重（同时作用于通道和空间）。它能够有效增强目标区域的响应，并抑制背景干扰，且不增加额外参数。

### 📊 实验结果

```
主要性能对比
模型	mAP@0.5	Precision	Recall	训练轮数
基线 YOLOv5s	0.863	0.838	0.853	200
改进模型	0.939	0.937	0.904	100
```

🖥️ 可视化界面
我们提供了一个基于 Tkinter 的友好界面，方便进行模型验证和结果展示。

```
📁 项目结构
text
infrared-small-target-detection/
├── data/                   # 存放数据集和配置文件
│   └── IRSTD-1K.yaml       # 数据集配置文件
├── models/                 # 模型定义
│   ├── common.py           # 包含 CoordConv 等自定义模块
│   ├── SimAM.py            # SimAM 注意力模块实现
│   └── yolov5s_improved.yaml # 改进后的模型配置文件
├── utils/                  # 工具函数
│   └── loss.py             # 包含 NWD 损失函数的实现
├── DataLoader.py           # 数据集预处理脚本
├── gui.py                  # Tkinter 可视化界面主程序
├── train.py                # 模型训练脚本
├── detect.py               # 模型推理脚本
├── requirements.txt        # 项目依赖
└── README.md               # 项目说明文档
```

🤝 贡献与反馈
我们欢迎任何形式的贡献！如果你有任何问题、建议或发现了 Bug，欢迎提交 Issue 或 Pull Request。

📜 许可证
本项目基于 AGPL-3.0 许可证开源，详情请见 LICENSE 文件。

📧 联系
项目作者：小诚

邮箱：fu.bocheng@outlook.com

<div align="center"> <sub>如果这个项目对您有帮助，请给我们一个 ⭐️ Star！</sub> </div> ```

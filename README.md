# RKNN ROS2  - OrangePi 5 Pro 

本仓库提供了在 OrangePi 5 Pro (RK3588S) 上运行 RKNN (Rockchip 神经网络) 推理的 ROS2 节点。该节点订阅摄像头图像话题，并使用预训练的 RKNN 模型进行目标检测。

## 📋 前提条件

### 硬件要求
- OrangePi 5 Pro (RK3588S 芯片)
- 兼容的摄像头模块

### 软件要求
- Ubuntu 22.04 (使用官方方法烧录)
- 已配置 RK3588S NPU 环境
- ROS2 (推荐 Humble 版本)

## 🚀 安装步骤

### 1. 烧录 Ubuntu 22.04
按照 OrangePi 5 Pro 用户手册的官方说明烧录 Ubuntu 22.04 系统。

### 2. 配置 NPU 环境
根据 RK3588S 用户手册配置 NPU 运行环境。

### 3. 克隆仓库
git clone https://github.com/Ikunio/rknn_ros2.git

### 4. 编译
在工作空间终端输入colcon build

### 5. source
source install/setup.bash

### 6. 运行
ros2 run rknn_ros2 rknn_ros_test



# RKNN ROS2 节点配置指南

## 📷 图像话题配置

### 如何修改图像话题
在 `rknn_ros_test.py` 文件中找到以下代码行，然后修改成自己的相机话题：

```python
frame = get_image_frame('/up_camera_image')
```

## 🤖 RKNN 模型配置

### 如何修改模型路径
在 `rknn_ros_test.py` 文件中找到以下代码行，修改成自己的rknn模型路径,  ###!!!要使用绝对路径###：


```python
RKNN_MODEL = '/home/orangepi/YOLO_TEST/src/rknn_ros2/rknn_ros2/apple.rknn'
```

## 🖥️ 模型类别配置

### 如何修改模型类别
在 `rknn_ros_test.py` 文件中找到以下代码行，修改成自己的模型类别 ：


```python
 CLASSES = ("apple")
```




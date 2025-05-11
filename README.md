1.根据香橙派5Pro用户手册进入官网烧入Ubuntu22.04镜像
2.首先要根据rk3588s用户手册，将调用NPU的环境配置好
3.git clone https://github.com/Ikunio/rknn_ros2.git
4.在工作空间colcon build
5.source install/setup.bash
6.ros2 run rknn_ros2 rknn_ros_test

根据自己的相机图像话题修改rknn_ros_test.py中  frame = get_image_frame('/up_camera_image')，替换成自己的图像话题
根据自己的rknn模型修改rknn_ros_test.py中  RKNN_MODEL = '/home/orangepi/YOLO_TEST/src/rknn_ros2/rknn_ros2/apple.rknn'
根据模型类别修改rknn_ros_test.py中   CLASSES = ("apple")

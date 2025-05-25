import time
import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from rknnlite.api import RKNNLite
import threading


RKNN_MODEL = '/home/orangepi/YOLO_TEST/src/rknn_ros2/rknn_ros2/apple.rknn'
IMG_PATH = './1_1.jpg'
OBJ_THRESH = 0.7

NMS_THRESH = 0.1
IMG_SIZE = 640
#CLASSES = ("0","1","2","3","4","5","6","7","8","9")
#CLASSES = ("else waste","hazardous","kitchen","recyclable")
CLASSES = ("apple")
# def sigmoid(x):
#     return 1 / (1 + np.exp(-x))

def xywh2xyxy(x):
    # Convert [x, y, w, h] to [x1, y1, x2, y2]
    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def process(input, mask, anchors):

    anchors = [anchors[i] for i in mask]
    grid_h, grid_w = map(int, input.shape[0:2])

    # box_confidence = sigmoid(input[..., 4])
    box_confidence = input[..., 4]
    box_confidence = np.expand_dims(box_confidence, axis=-1)

    # box_class_probs = sigmoid(input[..., 5:])
    box_class_probs = input[..., 5:]
    # box_xy = sigmoid(input[..., :2])*2 - 0.5
    box_xy = input[..., :2] * 2 - 0.5
    col = np.tile(np.arange(0, grid_w), grid_w).reshape(-1, grid_w)
    row = np.tile(np.arange(0, grid_h).reshape(-1, 1), grid_h)
    col = col.reshape(grid_h, grid_w, 1, 1).repeat(3, axis=-2)
    row = row.reshape(grid_h, grid_w, 1, 1).repeat(3, axis=-2)
    grid = np.concatenate((col, row), axis=-1)
    box_xy += grid
    box_xy *= int(IMG_SIZE/grid_h)

    #box_wh = pow(sigmoid(input[..., 2:4])*2, 2)
    box_wh = pow(input[..., 2:4] * 2, 2)
    box_wh = box_wh * anchors

    box = np.concatenate((box_xy, box_wh), axis=-1)

    return box, box_confidence, box_class_probs


def filter_boxes(boxes, box_confidences, box_class_probs):
    boxes = boxes.reshape(-1, 4)
    box_confidences = box_confidences.reshape(-1)
    box_class_probs = box_class_probs.reshape(-1, box_class_probs.shape[-1])

    _box_pos = np.where(box_confidences >= OBJ_THRESH)
    boxes = boxes[_box_pos]
    box_confidences = box_confidences[_box_pos]
    box_class_probs = box_class_probs[_box_pos]

    class_max_score = np.max(box_class_probs, axis=-1)
    classes = np.argmax(box_class_probs, axis=-1)
    _class_pos = np.where(class_max_score >= OBJ_THRESH)

    boxes = boxes[_class_pos]
    classes = classes[_class_pos]
    scores = (class_max_score* box_confidences)[_class_pos]

    return boxes, classes, scores


def nms_boxes(boxes, scores):
    x = boxes[:, 0]
    y = boxes[:, 1]
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]

    areas = w * h
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x[i], x[order[1:]])
        yy1 = np.maximum(y[i], y[order[1:]])
        xx2 = np.minimum(x[i] + w[i], x[order[1:]] + w[order[1:]])
        yy2 = np.minimum(y[i] + h[i], y[order[1:]] + h[order[1:]])

        w1 = np.maximum(0.0, xx2 - xx1 + 0.00001)
        h1 = np.maximum(0.0, yy2 - yy1 + 0.00001)
        inter = w1 * h1

        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= NMS_THRESH)[0]
        order = order[inds + 1]
    keep = np.array(keep)
    return keep


def yolov5_post_process(input_data):
    masks = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    # anchors = [[46, 115], [48, 104], [51, 117], [53, 107], [56, 116], [59, 99],
    #            [61, 124], [63, 109], [71, 100]]
    anchors = [[10, 13], [16, 30], [33, 23], [30, 61], [62, 45],
               [59, 119], [116, 90], [156, 198], [373, 326]]

    boxes, classes, scores = [], [], []
    for input, mask in zip(input_data, masks):
        b, c, s = process(input, mask, anchors)
        b, c, s = filter_boxes(b, c, s)
        boxes.append(b)
        classes.append(c)
        scores.append(s)

    boxes = np.concatenate(boxes)
    boxes = xywh2xyxy(boxes)
    classes = np.concatenate(classes)
    scores = np.concatenate(scores)

    nboxes, nclasses, nscores = [], [], []
    for c in set(classes):
        inds = np.where(classes == c)
        b = boxes[inds]
        c = classes[inds]
        s = scores[inds]

        keep = nms_boxes(b, s)

        nboxes.append(b[keep])
        nclasses.append(c[keep])
        nscores.append(s[keep])

    if not nclasses and not nscores:
        return None, None, None

    boxes = np.concatenate(nboxes)
    classes = np.concatenate(nclasses)
    scores = np.concatenate(nscores)

    return boxes, classes, scores


def draw1(image, boxes, scores, classes):
    for box, score, cl in zip(boxes, scores, classes):
        top, left, right, bottom = box

        print('class: {}, score: {}'.format(CLASSES[cl], score))
        print('box coordinate left,top,right,down: [{}, {}, {}, {}]'.format(top, left, right, bottom))
        top = int(top)
        left = int(left)
        right = int(right)
        bottom = int(bottom)

        
        cv2.rectangle(image, (top, left), (right, bottom), (255, 0, 0), 2)
        cv2.putText(image, '{0} {1:.2f}'.format(CLASSES[cl], score),
                    (top, left - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 255), 2)


def letterbox(im, new_shape=(640, 640), color=(0, 0, 0)):
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)

class ImageSubscriber(Node):
    def __init__(self, topic_name):
        super().__init__('image_subscriber')
        self.bridge = CvBridge()
        self.latest_frame = None
        self.lock = threading.Lock()
        
        self.subscription = self.create_subscription(
            Image,
            topic_name,
            self.image_callback,
            10)
        self.subscription  # prevent unused variable warning

    def image_callback(self, msg):
        try:
            # Convert ROS Image message to OpenCV format (BGR8)
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            
            with self.lock:
                self.latest_frame = cv_image.copy()
        except Exception as e:
            self.get_logger().error(f'Failed to convert image: {str(e)}')

    def get_latest_frame(self):
        with self.lock:
            return self.latest_frame.copy() if self.latest_frame is not None else None

# 修改get_image_frame函数，避免重复初始化ROS2
def get_image_frame(topic_name, timeout=5.0):
    """
    订阅ROS2图像话题并返回BGR格式的frame
    
    参数:
        topic_name (str): 要订阅的图像话题名称
        timeout (float): 等待图像的最大时间(秒)
    
    返回:
        numpy.ndarray: BGR格式的图像帧，如果超时或出错则返回None
    """
    # 检查ROS2是否已经初始化，避免重复初始化
    ros_initialized = rclpy.ok()
    if not ros_initialized:
        rclpy.init()
    
    # 创建订阅者节点
    image_subscriber = ImageSubscriber(topic_name)
    
    # 设置超时
    start_time = image_subscriber.get_clock().now().seconds_nanoseconds()[0]
    
    # 等待接收图像
    while rclpy.ok():
        rclpy.spin_once(image_subscriber, timeout_sec=0.1)
        
        # 检查是否接收到图像
        frame = image_subscriber.get_latest_frame()
        if frame is not None:
            image_subscriber.destroy_node()
            # 只有当我们自己初始化ROS2时才关闭它
            if not ros_initialized:
                rclpy.shutdown()
            return frame
        
        # 检查是否超时
        current_time = image_subscriber.get_clock().now().seconds_nanoseconds()[0]
        if current_time - start_time > timeout:
            image_subscriber.get_logger().warn(f'Timeout while waiting for image on topic {topic_name}')
            image_subscriber.destroy_node()
            # 只有当我们自己初始化ROS2时才关闭它
            if not ros_initialized:
                rclpy.shutdown()
            return None
    
    # 如果ROS2关闭，清理并返回None
    image_subscriber.destroy_node()
    # 只有当我们自己初始化ROS2时才关闭它
    if not ros_initialized:
        rclpy.shutdown()
    return None


def main(args=None):
    rclpy.init(args=args)
    rknn = RKNNLite()

    print('--> Load RKNN model')
    ret = rknn.load_rknn(RKNN_MODEL)
    if ret != 0:
        print('Load RKNN model failed')
        exit(ret)
    print('done')
    ret = rknn.init_runtime()
    if ret != 0:
        print('Init runtime environment failed!')
        exit(ret)
    print('done')
    
    # capture = cv2.VideoCapture(0)

    # ref, old_frame = capture.read()
    # if not ref:
    #     raise ValueError("error reading")
 
    fps = 0.0
    while True:
        t1 = time.time()

        # ret, frame = capture.read()
        # if not ret:
        #     break

        # 可以省略初始的BGR到RGB转换
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = get_image_frame('/up_camera_image')
        img = frame
        img, ratio, (dw, dh) = letterbox(img, new_shape=(IMG_SIZE, IMG_SIZE))

        # 确保img是BGR，因为你之前已经将img转为RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.expand_dims(img, 0)

        # Inference
        print('--> Running model')
        outputs = rknn.inference(inputs=[img])

        input0_data = outputs[0]
        input1_data = outputs[1]
        input2_data = outputs[2]

        input0_data = input0_data.reshape([3, -1] + list(input0_data.shape[-2:]))
        input1_data = input1_data.reshape([3, -1] + list(input1_data.shape[-2:]))
        input2_data = input2_data.reshape([3, -1] + list(input2_data.shape[-2:]))

        input_data = [np.transpose(input0_data, (2, 3, 0, 1)),
                      np.transpose(input1_data, (2, 3, 0, 1)),
                      np.transpose(input2_data, (2, 3, 0, 1))]

        boxes, classes, scores = yolov5_post_process(input_data)

        # 转换回BGR图像用于显示
        img_1 = cv2.cvtColor(img[0], cv2.COLOR_RGB2BGR)  # 注意这里需要[0]来降维
        if boxes is not None:
            draw1(img_1, boxes, scores, classes)

        fps = (fps + (1. / (time.time() - t1))) / 2
        print("fps= %.2f" % (fps))

        # 检查图像的形状和类型
        print(f'Image shape: {img_1.shape}, dtype: {img_1.dtype}')

        cv2.imshow("video", img_1)
        c= cv2.waitKey(1) & 0xff
        if c==27:
            capture.release()
            break
    print("Video Detection Done!")
    capture.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

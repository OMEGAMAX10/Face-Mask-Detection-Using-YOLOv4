import cv2
import threading
import numpy as np

LABELS = ["Without Mask", "Mask"]
COLORS = [[0, 0, 255], [0, 255, 0]]
weightsPath = "yolo_utils/yolov4_face_mask.weights"
configPath = "yolo_utils/yolov4-mask.cfg"


def create_detection_net(config_path, weights_path):
    net = cv2.dnn_DetectionModel(config_path, weights_path)
    net.setInputSize(608, 608)
    net.setInputScale(1.0 / 255)
    net.setInputSwapRB(True)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    return net


class CamThread(threading.Thread):
    def __init__(self, camName, camID, confThreshold=0.5, nmsThreshold=0.5):
        # camID can be either a number (0, 1, ...) or an IP address ('https://192.168.43.1:8080/video' for example)
        threading.Thread.__init__(self)
        self.previewName = camName
        self.camID = camID
        self.confThreshold = confThreshold
        self.nmsThreshold = nmsThreshold

    def run(self):
        print("Starting " + self.previewName)
        mask_detection_camera(self.previewName, self.camID, self.confThreshold, self.nmsThreshold)


def get_processed_image(img, net, confThreshold, nmsThreshold):
    mask_count = 0
    nomask_count = 0
    border_size = 50
    border_text_color = [255, 255, 255]
    img = cv2.copyMakeBorder(img, border_size, 0, 0, 0, cv2.BORDER_CONSTANT)
    classes, confidences, boxes = net.detect(img, confThreshold, nmsThreshold)
    for cl, score, (left, top, width, height) in zip(classes, confidences, boxes):
        mask_count += cl[0]
        nomask_count += (1 - cl[0])
        start_point = (int(left), int(top))
        end_point = (int(left + width), int(top + height))
        color = COLORS[cl[0]]
        img = cv2.rectangle(img, start_point, end_point, color, 2)  # draw class box
        text = f'{LABELS[cl[0]]}: {score[0]:0.2f}'
        (test_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_ITALIC, 0.6, 1)
        end_point = (int(left + test_width + 2), int(top - text_height - 2))
        img = cv2.rectangle(img, start_point, end_point, color, -1)
        cv2.putText(img, text, start_point, cv2.FONT_ITALIC, 0.6, COLORS[1 - cl[0]], 1)  # print class type with score
    text = f'   Mask Count: {mask_count}        No Mask Count: {nomask_count}'
    cv2.putText(img, text, (0, int(border_size - 17)), cv2.FONT_ITALIC, 0.8, border_text_color, 2)
    cv2.putText(img, "Status:", (img.shape[1] - 230, int(border_size - 15)), cv2.FONT_ITALIC, 0.8, border_text_color, 2)
    ratio = nomask_count / (mask_count + nomask_count + 0.000001)
    if ratio >= 0.1 and nomask_count >= 3:
        cv2.putText(img, "Danger !", (img.shape[1] - 130, int(border_size - 17)), cv2.FONT_ITALIC, 0.8, [26, 13, 247], 2)
    elif ratio != 0 and np.isnan(ratio) is not True:
        cv2.putText(img, "Warning !", (img.shape[1] - 130, int(border_size - 17)), cv2.FONT_ITALIC, 0.8, [0, 255, 255], 2)
    else:
        cv2.putText(img, "Safe", (img.shape[1] - 130, int(border_size - 17)), cv2.FONT_ITALIC, 0.8, [0, 255, 0], 2)
    return img


def mask_detection_camera(camName, camID, confThreshold=0.5, nmsThreshold=0.5):
    net = create_detection_net(configPath, weightsPath)
    cam = cv2.VideoCapture(camID)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    while cam.isOpened():
        ret, frame = cam.read()
        if ret is False:
            break
        frame = get_processed_image(frame, net, confThreshold, nmsThreshold)
        cv2.imshow(camName + ' - Face Mask Detection (Press ESC for close camera)', frame)
        if cv2.waitKey(int(1000 // cam.get(cv2.CAP_PROP_FPS))) & 0xFF == 27:  # 27 = ESC ASCII code
            break
    cam.release()
    cv2.destroyAllWindows()


cam_threads = []
cam_threads.append(CamThread("Camera 1", 0))
cam_threads.append(CamThread("Camera 2", 1))
# cam_threads.append(CamThread("IP Camera", 'https://192.168.43.1:8080/video'))

for thread in cam_threads:
    thread.start()

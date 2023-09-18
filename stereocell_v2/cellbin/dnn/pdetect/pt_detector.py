import numpy as np
import onnxruntime

from cellbin.dnn.onnx_net import OnnxNet
from cellbin.dnn.common import letterbox
from cellbin.dnn.common import scale_polys
from cellbin.dnn.pdetect.util import rbox2poly
from cellbin.dnn.pdetect.util import non_max_suppression_obb_np


def init_session(model_path):
    EP_list = ['CPUExecutionProvider']
    sess = onnxruntime.InferenceSession(model_path, providers=EP_list)
    return sess


class PickableInferenceSession:  # This is a wrapper to make the current InferenceSession class pickable.
    def __init__(self, model_path):
        self.model_path = model_path
        self.sess = init_session(self.model_path)

    def run(self, input_name, data):
        return self.sess.run(None, {f"{input_name}": data})

    def __getstate__(self):
        return {'model_path': self.model_path}

    def __setstate__(self, values):
        self.model_path = values['model_path']
        self.sess = init_session(self.model_path)


class PointDetector(object):
    def __init__(
            self,
            conf_thresh=0.25,
            iou_thresh=0.1,
            gpu='-1',
            num_threads=0,
    ):
        self.gpu = gpu
        self.num_threads = num_threads
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh

        self.img_func = None
        self.img_size = (1024, 1024)
        self.onnx_net = None
        self._input_name = 'images'
        self.model = None

    def load_model(self, weight_path):
        self.model = PickableInferenceSession(weight_path)

    def set_func(self, fun):
        self.img_func = fun

    def preprocess(self, img):
        enhance_img = self.img_func(img)  # returned image in BGR format
        ori_shape = enhance_img.shape
        padded_img = letterbox(enhance_img, self.img_size)
        # Convert
        adjust_img = padded_img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        adjust_img = np.ascontiguousarray(adjust_img)  # 变成C连续数组(内存连续), 处理速度更快
        adjust_img = np.float32(adjust_img)  # must do this for onnx
        adjust_img /= 255
        return ori_shape, adjust_img

    @staticmethod
    def postprocess(det, w, h, tol=15):
        angles = np.median(np.degrees(det[:, -3]))
        if angles >= 0:
            img_angle = 90 - (180 - 45 + angles)
            img_angle = -img_angle
        else:
            img_angle = 90 - (180 - (-angles + 45))
        mid_1 = (det[:, 4: 6] + det[:, 0: 2]) / 2
        mid_2 = (det[:, 6: 8] + det[:, 2: 4]) / 2
        mid_final = (mid_1 + mid_2) / 2.0
        xy = np.concatenate((mid_final, det[:, -2].reshape(-1, 1),), axis=1)
        x, y = xy[:, 0], xy[:, 1]
        remain = ~np.logical_or.reduce(np.concatenate(((x - tol < 0).reshape(-1, 1),
                                                       (x + tol > w).reshape(-1, 1),
                                                       (y - tol < 0).reshape(-1, 1),
                                                       (y + tol > h).reshape(-1, 1)), axis=1), axis=1)
        xy = xy[remain]
        return xy.tolist(), img_angle

    def predict(self, img):
        # if not isinstance(img, np.ndarray):
        #     raise Exception(f"Only accept numpy array as input")
        #
        # if not isinstance(ori_shape, tuple):
        #     raise Exception(f"Original shape should be in tuple format")
        ori_shape, img = self.preprocess(img)
        cp, angle = list(), None
        h, w = ori_shape[:2]
        if len(img.shape) == 3:
            img = img[None]

        # Inference
        pred = self.model.run(input_name=self._input_name, data=img)[0]  # list*(n, [cx, cy, l, s, θ, conf, cls]) θ ∈ [-pi/2, pi/2)
        pred = non_max_suppression_obb_np(
            prediction=pred,
            conf_thres=self.conf_thresh,
            iou_thres=self.iou_thresh,
            multi_label=True,
            max_det=200,
        )

        det = pred[0]
        # for i, det in enumerate(pred):  # per image
        pred_poly = rbox2poly(det[:, :5])  # (n, [x1 y1 x2 y2 x3 y3 x4 y4])
        pred_poly = scale_polys(img.shape[2:], pred_poly, ori_shape)
        det = np.concatenate((pred_poly, pred[0][:, -3:]), axis=1)  # pred[0][:, -3:] -> [θ, conf, cls]
        if len(det):
            cp, angle = self.postprocess(det, w, h)

        return cp, angle


if __name__ == '__main__':
    import cv2

    model_path = r"D:\PycharmProjects\imageqc\ImageQC\weights\ST_TP_V2_05_13_Nano.onnx"
    ci = PointDetector(model_path)
    ci.load_model()
    ci.set_func(test_enhance)
    img_path = r"D:\Data\tmp\Y00035MD\Y00035MD\Y00035MD_0008_0005_2023-01-30_15-51-18-483.tif"
    img = cv2.imread(img_path, -1)
    ori_shape, adjust_img = ci.preprocess(img)
    cp, angle = ci.predict(adjust_img, ori_shape)
    print("asd")

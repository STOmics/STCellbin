from cellbin.dnn.tseg import TissueSegmentation
from cellbin.dnn.tseg.yolo.processing import f_process_mask, f_preformat, f_scale_image, f_img_process
from cellbin.dnn.tseg.yolo.nms import f_non_max_suppression
from cellbin.dnn.onnx_net import OnnxNet

import numpy as np
import cv2


class TissueSegmentationYolo(TissueSegmentation):
    def __init__(self,
                 gpu=-1,
                 num_threads=0,
                 classes=None,  # filter by class: --class 0, or --class 0 2 3
                 conf_thres=0.25,  # confidence threshold
                 iou_thres=0.45,  # NMS IOU threshold
                 max_det=1000,  # maximum detections per image
                 agnostic_nms=False,  # class-agnostic NMS
                 half=False,  # use FP16 half-precision inference
                 ):
        self._INPUT_SIZE = (640, 640)
        self._model_path = None
        self._gpu = gpu
        self._num_threads = num_threads
        self._conf_thres = conf_thres
        self._iou_thres = iou_thres
        self._max_det = max_det
        self._classes = classes
        self._agnostic_nms = agnostic_nms
        self._half = half
        self._model = None
        self.mask_num = None
        # self._f_init_model()

    def f_init_model(self, model_path):
        """
        init model
        """
        self._model = OnnxNet(model_path, self._gpu, self._num_threads)

    def f_predict(self, img):
        source_shape = img.shape[:2]
        img = f_img_process(img)
        img = f_preformat(img)
        pred, proto = self._model.f_predict(img)
        pred = f_non_max_suppression(pred, self._conf_thres, self._iou_thres, self._classes, self._agnostic_nms,
                                     self._max_det, nm=32)
        for i, det in enumerate(pred):  # per image
            if len(det):
                masks = f_process_mask(proto[i], det[:, 6:], det[:, :4], img.shape[2:], upsample=True)  # HWC
                mask = np.zeros(self._INPUT_SIZE, dtype=np.uint8)
                for m in masks:
                    a = np.asarray(m, dtype=np.float16)
                    a[a > self._conf_thres] = 1.0
                    a = np.asarray(a, dtype=np.uint8)
                    mask = cv2.bitwise_or(mask, a)
                # mask = f_scale_image((self._INPUT_SIZE[0], self._INPUT_SIZE[1], 3), mask,
                #                      (source_shape[0], source_shape[1], 3))
                mask = cv2.resize(mask, (source_shape[1], source_shape[0]), interpolation=cv2.INTER_NEAREST_EXACT)
                self.mask_num = np.sum(mask)

        return mask


def main():
    import os
    import tifffile
    seg = TissueSegmentationYolo()
    seg.f_init_model(r"D:\code\envs\tissuecut_yolo\tissueseg_yolo_SH_20230131.onnx",)
    img = tifffile.imread(r"D:\stock\dataset\test\fov_stitched.tif")
    mask = seg.f_predict(img, type="matrix")
    tifffile.imwrite(r"D:\stock\dataset\test\1\fov_stitched.tif", mask)


if __name__ == '__main__':
    import sys

    main()
    sys.exit()


import os
import json
import numpy as np

from cellbin.utils.file_manager import rc_key
from cellbin.modules import CellBinElement, StainType
from cellbin.modules.iqc.classify_fov import ClassifyFOV
from cellbin.dnn.tseg.yolo.detector import TissueSegmentationYolo
from cellbin.image.augmentation import pt_enhance_method, line_enhance_method, clarity_enhance_method
from cellbin.modules.iqc.regist_qc import RegistQC
from cellbin.modules.iqc.clarity_qc import ClarityQC
from cellbin.image.wsi_split import SplitWSI
from cellbin.modules.iqc.stitch_qc import StitchQC
from cellbin.image.wsi_stitch import StitchingWSI
from cellbin.contrib.track_roi_picker import TrackROIPicker
from cellbin.image import Image
from cellbin.image.mask import iou
from cellbin.utils import clog


class ImageQualityControl(CellBinElement):
    def __init__(self, ):
        super(ImageQualityControl, self).__init__()
        self._tissue_detector = TissueSegmentationYolo()
        self._rqc = RegistQC()
        self._cqc = ClarityQC()
        self._cl_fov = ClassifyFOV()

        self.debug = False

        self.json_path = None
        self.image_root = None
        self.image_map = None
        self._stain_type = None
        self._stereo_chip = None
        self._fov_rows = self._fov_cols = None
        self._is_stitched = False
        self._fov_size = None
        self._magnification = None

        self._tissue_mask = None
        self._mosaic = None
        self._fovs_loc = None
        self._roi_fovs = None

        # running config
        self.pt_running_process = None
        self.stitch_running_process = None

        # detect channel
        self.marker_detect_channel = None

        # threshold
        self.track_point_score_threshold = None
        self.track_line_score_threshold = None
        self.clarity_threshold = None
        self.cluster_area_threshold = None
        self.cluster_width_threshold = None
        self.cluster_height_threshold = None

        #
        self.track_point_first_level_threshold = None
        self.track_point_second_level_threshold = None
        self.track_point_good_threshold = None
        self.track_line_topk = None

        # template threshold
        self.template_pair_points_threshold = None
        self.template_pair_points_qc_threshold = None
        self.template_range_image_size_threshold = None
        self.template_pair_correct_threshold = None
        self.template_cluster_num_threshold = None

        # flag
        self.pt_qc_flag = None
        self.line_qc_flag = None
        self.regist_qc_flag = None  # pt_qc pass + line_qc pass = regist_qc pass
        self.clarity_flag = None  # clarity pass or failed
        self.global_template_flag = None  # global template eval success or fail
        self.microscope_stitch = None

        # output
        # Image info
        self.total_fov_count = None

        # tissue qc output
        self._box_mosaic = None  #
        self._fov_tissue_mask = None
        self.stitch_roi = ()
        self.tissue_mask_score = None
        self.tissue_area = -1

        # regist qc output
        self.track_score = -1.0
        self.track_line_score = -1.0
        self.track_pts = np.array([])
        self.rotation = -1.0
        self.scale_x = -1.0
        self.scale_y = -1.0
        self.good_fov_count = -1

        self.track_pt_fov_mask = -1
        self.track_line_best_match = []
        self._src_fov_info = None  # template fov for stitch

        # clarity qc output
        self.clarity_score = -1.0  # float，清晰度得分
        self.clarity_preds_arr = -1
        self.clarity_heatmap = np.array([])  # numpy array，清晰度结果呈现在原图上
        self.clarity_cluster = None  # plt plot object，清晰度成团聚类结果

        self.clarity_topk_result = []
        self.clarity_counts = {}

        # stitch qc + global template qc output
        self.stitch_diff = -1  # np ndarray
        self.jitter_diff = -1  # np ndrray
        self.stitch_diff_max = -1.0  # float
        self.jitter_diff_max = -1.0  # float
        self.template_max = -1.0  # float
        self.template_mean = -1.0  # float
        self.template_std = -1.0  # float
        self.template_qc_precision = -1.0  # float
        self.template_qc_recall = -1.0  # float
        self.global_template = -1  # list
        self.jitter = [-1, -1]
        self.stitch_fov_loc = -1
        self.stitch_template_global = -1  # global heatmap, r x c x 1

    def initialize(self, tissue_detector_file: str, pt_detector_file: str, clarity_file: str):
        """
        Initialize dnn model

        Args:
            tissue_detector_file (): tissue cut weight path
            pt_detector_file (): track detect weight path
            clarity_file (): clarity eval weight path

        Returns:

        """
        # 导入模型权重
        try:
            self._tissue_detector.f_init_model(tissue_detector_file)
            self._rqc.track_pt_qc.ci.load_model(pt_detector_file)
            self._cqc.load_model(clarity_file)
        except Exception as e:
            clog.error(f"{e}")
            clog.error(f"dnn model weights load failed")
            return 1
        return 0

    def initialize_json(self, json_path: str):
        """
        Initialize config file, will initialize all the thresholds and

        Args:
            json_path (): config json path

        Returns:

        """
        with open(json_path, 'r') as f:
            th_dict = json.load(f)
        for key, val in th_dict.items():  # auto assign dict
            self.__dict__.update(val)
            clog.info(f"Assigning {val} to {key}")

    @staticmethod
    def iou_calculation(clarity_mask, tissue_mask):
        iou_result = iou(clarity_mask, tissue_mask)
        return iou_result

    def _fov_box(self, b):
        w, h = self._fov_size
        y_begin, y_end, x_begin, x_end = b
        row = y_begin // h
        col = x_begin // w
        self._fovs_loc[row, col] = [x_begin, y_begin]
        self.image_map[rc_key(row, col)] = b

    def _get_jitter(self, ):
        if not self._is_stitched:
            from cellbin.contrib.fov_aligner import FOVAligner
            r, c = self._fovs_loc.shape[:2]
            clog.info(f"Fov aligner using channel: {self.marker_detect_channel[self._stain_type]['fft']}")
            fa = FOVAligner(
                images_path=self.image_map,
                rows=r,
                cols=c,
                channel=self.marker_detect_channel[self._stain_type]['fft']
            )
            fa.set_process(self.stitch_running_process)
            fa.create_jitter()
            self.jitter = [fa.horizontal_jitter, fa.vertical_jitter]

    def _prepare(self, ):
        """
        This func is mainly used to deal with
        - split large tif to fovs

        Returns:

        """
        if self._is_stitched:
            image_reader = Image()
            img_path = os.path.join(self.image_root, self.image_map)
            image_reader.read(img_path)
            self._mosaic = image_reader.image
            h_, w_ = image_reader.height, image_reader.width
            w, h = self._fov_size
            self._fov_rows, self._fov_cols = [(h_ // h) + 1, (w_ // w) + 1]
            self._fovs_loc = np.zeros((self._fov_rows, self._fov_cols, 2), dtype=int)
            self.image_map = dict()

            wsi = SplitWSI(img=self._mosaic, win_shape=(h, w),
                           overlap=0, batch_size=1, need_fun_ret=False, need_combine_ret=False)
            _box_lst, _fun_ret, _dst = wsi.f_split2run()
            for b in _box_lst:
                self._fov_box(b)
        else:
            wsi = StitchingWSI()
            self._fov_rows, self._fov_cols = self._fovs_loc.shape[:2]
            wsi.mosaic(src_image=self.image_map, loc=self._fovs_loc, downsample=1)
            self._mosaic = wsi.buffer
        self.total_fov_count = len(self.image_map)  # 大图需要从这里获取total fov count

    def _classify_fov(self, ):
        """
        This func will do:
        - tissue cut based on "stitched" image
        - classify fovs based on tissue cut result
        - classified result will contain
            - tissue fov
            - non tissue fov

        Returns:
            self._fov_tissue_mask: classified fov result mask
            self._tissue_mask : tissue mask
            self._box_mosaic: tissue bounding box on stitched image
            self.stitch_roi: stitched roi. col, row, col, row
            self.tissue_area: sum of tissue mask

        """
        # fov_classify = ClassifyFOV(self._tissue_detector)
        self._cl_fov.set_detector(self._tissue_detector)
        self._cl_fov.classify(
            mosaic=self._mosaic,
            fov_loc=self._fovs_loc,
            fov_size=self._fov_size,
            expand=1,
            ch=self.marker_detect_channel[self._stain_type]['fft']
        )
        self._fov_tissue_mask = self._cl_fov.tissue_fov_map
        self._tissue_mask = self._cl_fov.tissue_mask
        self._box_mosaic = self._cl_fov.tissue_bbox_in_mosaic()
        self.stitch_roi = self._cl_fov.tissue_fov_roi
        self.tissue_area = self._cl_fov.tissue_detector.mask_num

    def _registration_qc(self, ):
        """
        This func will do
        - track pt detect
        - track line detect
        - track line result match

        Returns:

        """
        # set threshold
        self._rqc.set_chip_template(self._stereo_chip)
        self._rqc.set_track_pt_thresh(
            th=self.track_point_first_level_threshold,
            th2=self.track_point_second_level_threshold,
            good_thresh=self.track_point_good_threshold,
        )
        self._rqc.set_topk(self.track_line_topk)
        self._rqc.set_track_pt_process(self.pt_running_process)

        # start
        buffer = None
        if self._is_stitched:
            buffer = self._mosaic

        # Track点检测
        self._rqc.run_pt_qc(
            fovs=self.image_map,
            enhance_func=pt_enhance_method.get(self._stain_type, None),
            detect_channel=self.marker_detect_channel[self._stain_type]['pt_detect'],
            buffer=buffer
        )
        self.track_score = self._rqc.pt_score
        if self.track_score >= self.track_point_score_threshold:
            self.pt_qc_flag = 1
        else:
            self.pt_qc_flag = 0

        self.track_pts = self._rqc.pt_result  # 未检测到为空

        # Track线检测
        if self._magnification <= 15:
            line_fovs = None
        else:
            trp = TrackROIPicker(
                images_path=self.image_map, jitter_list=self.jitter,
                tissue_mask=self._fov_tissue_mask,
                track_points=self._rqc.track_pt_qc.track_result()
            )
            line_fovs = trp.getRoiImages()

        self._rqc.run_line_qc(
            line_fovs=line_fovs,
            detect_channel=self.marker_detect_channel[self._stain_type]['line_detect'],
            magnification=self._magnification,
            buffer=buffer,
            enhance_func=line_enhance_method.get(self._stain_type, None),
        )
        self.track_line_score = self._rqc.line_score
        if self.track_line_score > self.track_line_score_threshold:
            self.line_qc_flag = 1
            self._src_fov_info = self._rqc.best_match
            self.rotation = self._src_fov_info[-1]
            self.scale_x = self._src_fov_info[-3]
            self.scale_y = self._src_fov_info[-2]
        else:
            self.line_qc_flag = 0

        # Track点 + Track线通过 = regist qc通过
        if self.pt_qc_flag and self.line_qc_flag:
            self.regist_qc_flag = 1
        else:
            self.regist_qc_flag = 0

        self.track_pt_fov_mask = self._rqc.track_pt_qc.fov_mask
        self.track_line_best_match = self._rqc.best_match
        self.good_fov_count = self._rqc.good_fov_count
        # TODO: Track line qc pass + regist qc pass -> 推模板qc，不过就没法做推模板qc

    def _stitching_qc(self, ):
        """
        拼接总体QC模块 主要调用 StitchQC 实现拼接坐标计算以及模板推导计算
        并包含各类评估指标信息
        """
        clog.info(f"Stitch qc using channel: {self.marker_detect_channel[self._stain_type]['fft']}")
        if self._rqc.line_score == 0:
            sth_qc = StitchQC(
                is_stitched=self._is_stitched,
                src_fovs=self.image_map,
                fft_channel=self.marker_detect_channel[self._stain_type]['fft']
            )

        else:
            src_fov, cross_pts, x_scale, y_scale, rotation = self._src_fov_info
            total_cross_pt = dict()
            for k, v in self._rqc.pt_result.items():
                total_cross_pt[k] = v[0]
            correct_points = total_cross_pt[src_fov]
            total_cross_pt[src_fov] = cross_pts
            sth_qc = StitchQC(
                is_stitched=self._is_stitched,
                src_fovs=self.image_map,
                pts=total_cross_pt,
                scale_x=x_scale,
                scale_y=y_scale,
                rotate=rotation,
                chipno=self._stereo_chip,
                index=src_fov,
                correct_points=correct_points,
                pair_thresh=self.template_pair_points_threshold,
                qc_thresh=self.template_pair_points_qc_threshold,
                range_thresh=self.template_range_image_size_threshold,
                correct_thresh=self.template_pair_correct_threshold,
                cluster_num=self.template_cluster_num_threshold,
                fft_channel=self.marker_detect_channel[self._stain_type]['fft']
            )

        if self._fovs_loc is not None:
            sth_qc.set_location(self._fovs_loc)
        sth_qc.set_size(self._fov_rows, self._fov_cols)
        dct, template_global = sth_qc.run_qc()
        self.microscope_stitch = 1
        if -1 <= dct.get('template_re_conf', -1.0) < 0.1 \
                and not self._is_stitched \
                and self._rqc.line_score != 0:
            clog.info("Microscope coordinates have significant errors, use BGI stitching algo")
            self.microscope_stitch = 0
            self._get_jitter()
            sth_qc.set_jitter(self.jitter)
            sth_qc._scale_x = x_scale
            sth_qc._scale_y = y_scale
            sth_qc._rotate = rotation
            sth_qc.fov_location = None
            dct, template_global = sth_qc.run_qc()

        # stitch module
        self.stitch_template_global = template_global
        self.stitch_fov_loc = sth_qc.fov_location
        self.stitch_diff = dct.get('stitch_diff', -1.0)
        self.jitter_diff = dct.get('jitter_diff', -1.0)
        self.stitch_diff_max = dct.get('stitch_diff_max', -1.0)
        self.jitter_diff_max = dct.get('jitter_diff_max', -1.0)

        if self.stitch_diff is None:
            self.stitch_diff = self.jitter_diff = -1
            self.stitch_diff_max = self.jitter_diff_max = -1

        # template module
        self.scale_x, self.scale_y, self.rotation = sth_qc.get_scale_and_rotation()
        self.global_template = sth_qc.template
        self.template_max = dct.get('template_max', -1.0)
        self.template_mean = dct.get('template_mean', -1.0)
        self.template_std = dct.get('template_std', -1.0)
        self.template_qc_precision = dct.get('template_qc_conf', -1.0)
        self.template_qc_recall = dct.get('template_re_conf', -1.0)
        if self.template_max == -1 or self.template_mean == -1 or self.template_std == -1:
            self.global_template_flag = 0
        else:
            self.global_template_flag = 1

    def _clarity_qc(self, ):
        """
        This func will do clarity eval on stitched image

        Returns:

        """
        x0, y0, x1, y1 = self._box_mosaic
        self._cqc.set_enhance_func(clarity_enhance_method.get(self._stain_type, None))
        self._cqc.run(
            img=self._mosaic,
            detect_channel=self.marker_detect_channel[self._stain_type]['clarity']
        )
        if self.debug:
            self._cqc.cluster()
            self.clarity_cluster = self._cqc.fig  # plt plot object，清晰度成团聚类结果
            self.clarity_topk_result = self._cqc.topk_result

        # clarity qc output
        self._cqc.post_process()
        self.clarity_cut_size = self._cqc.cl_classify.img_size
        self.clarity_overlap = self._cqc.cl_classify.overlap
        self.clarity_score = self._cqc.score  # float，清晰度得分
        self.clarity_heatmap = self._cqc.draw_img  # numpy array，清晰度结果呈现在原图上
        self.clarity_preds_arr = self._cqc.preds  # clarity predictions array

        self.clarity_counts = self._cqc.counts

    def set_is_stitched(self, stitched):
        self._is_stitched = stitched

    def set_fov_loc(self, loc):
        self._fovs_loc = loc

    def set_fov_size(self, s):
        self._fov_size = s

    def set_stereo_chip(self, c):
        self._stereo_chip = c

    def set_stain_type(self, s):
        self._stain_type = s

    def set_magnification(self, m):
        assert m in [10, 20, 40]
        self._magnification = m

    def set_debug_mode(self, d):
        self.debug = d

    def run(self, image_root: str, image_map):
        # try:
        clog.info("-------------------Start QC-------------------")
        clog.info(f"Is_stitched: {self._is_stitched}, fov size: {self._fov_size}, stain_type: {self._stain_type} \n"
                  f"Magnification: {self._magnification}, debug mode: {self.debug}, chip: {self._stereo_chip}")
        self.image_root = image_root
        self.image_map = image_map
        clog.info(f"Image root : {self.image_root}")
        self._prepare()
        self._classify_fov()
        # self._get_jitter()

        self._registration_qc()
        self._stitching_qc()
        if self.debug:
            self._clarity_qc()
            self.tissue_mask_score = self.iou_calculation(
                clarity_mask=self._cqc.black_img,
                tissue_mask=self._tissue_mask
            )
        clog.info("-------------------End QC-------------------")
        # except Exception as e:
        #     clog.error(f"{e}")
        #     clog.error("QC run bug")
        # return 0

    @property
    def src_fov_info(self):
        return self._src_fov_info

    @property
    def box_mosaic(self):
        return self._box_mosaic

    @property
    def fov_tissue_mask(self):
        return self._fov_tissue_mask

    @property
    def tissue_mask(self):
        return self._tissue_mask

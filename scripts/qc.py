import os
import time
import cv2
import h5py
import numpy as np
import datetime
import tifffile
import tqdm
import sys

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(ROOT)

from stereocell_v2.cellbin.dnn.weights import auto_download_weights
from stereocell_v2.cellbin.utils import clog
from stereocell_v2.cellbin.modules.image_qc import ImageQualityControl
from stereocell_v2.cellbin.utils.file_manager import rc_key
from stereocell_v2.cellbin.utils.file_manager import rc_key_trans
from stereocell_v2.cellbin.image import Image
from stereocell_v2.cellbin.image.wsi_stitch import StitchingWSI
from stereocell_v2.cellbin.image.wsi_split import SplitWSI

from stereocell_v2.stio import slide2ipr0d0d1
from stereocell_v2.stio.microscopy.slide_factory import MicroscopeBaseFileFactory
from stereocell_v2.stio.chip import STOmicsChip

from stereocell_v2.calibration.match_calibration import FFTRegister

# DEFAULT_ZOO_PATH = '../../cellbin/dnn/weights'
ZOO = {
    'tissue': 'tissueseg_yolo_SH_20230131_th.onnx',
    'track_pt': 'points_detect_yolov5obb_SSDNA_20220513_pytorch.onnx',
    'clarity': 'clarity_eval_mobilev3small05064_DAPI_20230202_pytorch.onnx'
}

CONFIG_PATH = "config.json"

VERSION = '0.2'


class MIFImageQualityControl(ImageQualityControl):
    def __init__(self):
        super(MIFImageQualityControl, self).__init__()
        self.init_value = 999
        self.mif_stitch_dict: dict = dict()
        self.mif_stitch_dict["BGIStitchFlag"] = False

    @staticmethod
    def save_moasic(src_image: dict, loc: np.ndarray, output_path: str = None, downsample=1, multi=False):
        '''
        save image by get location
        Args:
            src_image: dict: {"row_col": file_path}
            loc: stitch global location
            output_path: output path (including file name)
            downsample:
            multi:
        Returns:

        '''
        wsi = StitchingWSI()
        wsi.mosaic(src_image=src_image, loc=loc, downsample=downsample, multi=multi)
        if output_path is not None:
            tifffile.imwrite(output_path, wsi.buffer)
        return wsi.buffer

    @staticmethod
    def stitch_stability_score(matrix: np.ndarray, thread: int = 7):
        """
        Stitching scores were calculated based on the heat map
        :param matrix: stitch_ED_max
        :param thread: stitch offset of thread
        :return: score
        """
        # matrix[np.where((matrix<5) & (matrix>0))]=0
        values = matrix[np.where(matrix >= 0)]

        score = 1 / (1 + np.exp(values - thread))
        score = np.sum(score)
        if len(values) / matrix.size < 0.3:
            return score / matrix.size
        else:
            return score / len(values)

    def small_scope_jitter(self, image_map, fft_detect_channel=0):
        """
        calculate scope overlap stability
        Args:
            image_map: dict

        Returns:

        """
        from cellbin.contrib.fov_aligner import FOVAligner
        r, c = self._fovs_loc.shape[:2]
        fa = FOVAligner(
            images_path=image_map,
            rows=r,
            cols=c,
            channel=fft_detect_channel
        )
        fa.set_process(self.stitch_running_process)
        fa.create_jitter()
        self.jitter = [fa.horizontal_jitter, fa.vertical_jitter]
        return fa.horizontal_jitter, fa.vertical_jitter

    def _prepare(self):
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

            wsi = SplitWSI(img=self._mosaic, win_shape=(w, h),
                           overlap=0, batch_size=1, need_fun_ret=False, need_combine_ret=False)
            _box_lst, _fun_ret, _dst = wsi.f_split2run()
            for b in _box_lst:
                self._fov_box(b)
        else:
            from cellbin.contrib.fov_aligner import FOVAligner
            r, c = self._fovs_loc.shape[:2]
            self.small_scope_jitter(self.image_map, self.marker_detect_channel[self._stain_type]['fft'])

            # start stitch
            from cellbin.modules.stitching import Stitching
            cbs = Stitching()
            cbs.set_jitter(self.jitter[0], self.jitter[1])
            cbs.set_size(r, c)
            cbs.stitch(self.image_map)
            # eval stitch
            self.mif_stitch_dict = cbs.get_stitch_eval()
            if "stitch_diff" in self.mif_stitch_dict.keys():
                scope_jitter = self.mif_stitch_dict["jitter_diff"]
                self.mif_stitch_dict["ScopeStitchQCScore"] = self.stitch_stability_score(scope_jitter)
                self.mif_stitch_dict["BGIStitchFlag"] = True

            # stitch module
            self.dapi_stitch_fov_loc = cbs.fov_location
            self._fovs_loc = cbs.fov_location
            self._fov_rows, self._fov_cols = self._fovs_loc.shape[:2]

            # get stitch image by fov location
            wsi = StitchingWSI()
            wsi.mosaic(src_image=self.image_map, loc=self._fovs_loc, downsample=1)
            self._mosaic = wsi.buffer

            # get fov box
            w, h = self._fov_size
            for col in range(c):
                for row in range(r):
                    name = '{}_{}'.format(str(row).zfill(4), str(col).zfill(4))
                    x_begin, y_begin = self._fovs_loc[row, col]
                    # y_begin, y_end, x_begin, x_end = box
                    self.image_map[name] = [y_begin, y_begin + w, x_begin, x_begin + h]
            self._is_stitched = True

        global_height, global_width = self._mosaic.shape[:2]
        self.mif_stitch_dict["StitchedGlobalHeight"] = global_height
        self.mif_stitch_dict["StitchedGlobalWidth"] = global_width

    def run(self, image_root: str, image_map, save_path=None):
        self.image_root = image_root
        self.image_map = image_map
        self._prepare()

        self._classify_fov()

        self._registration_qc()
        self._stitching_qc()

        # todo temporarily closed clarity debug
        self.debug = False
        # self._clarity_qc()
        print("asd")


class QcRunner(object):
    def __init__(self):
        self.image_qc = MIFImageQualityControl()
        self.zoo_dir = ""
        self.tissue_weight_path = ""
        self.pt_detect_weight_path = ""
        self.clarity_weight_path = ""

        # qc input
        self.image_root = None  # user input
        self.is_stitched = None  # ipr read
        self.stain_type = None  # user input
        self.TrackLineChannel = "DAPI" # user input
        self.stereo_chip = None  # ipr read
        self.magnification = None  # ipr read
        self.fov_location = None  # ipr read
        self.image_dict = None  # ipr read
        self.fov_size = None  # ipr read, (w, h)

        # cio
        self.ipr_manager = None
        self.chip_name = None
        self.chip_template_getter = STOmicsChip()

        #
        self.debug = False
        self.save_dir = None
        self.ipr_save_path = None
        self.file_name = None

    def download_and_check_weights(self, save_dir):
        download_code = auto_download_weights(save_dir)
        if download_code != 0:
            return 1
        return 0

    def read_microscope(self, src_data):
        if isinstance(src_data, str):
            msp = MicroscopeBaseFileFactory().create_microscope_file(src_data)
        else:
            msp = src_data
            msp.parse_single_img_info(self.save_dir, msp.current_channel)
            src_data = self.save_dir
        self.ipr_manager = slide2ipr0d0d1(msp)
        # 芯片号合法化校验
        if not self.chip_template_getter.is_chip_number_legal(self.chip_name):
            # if not check_chip_no(self.chip_name):
            raise Exception(f"{self.chip_name} not supported")
        # 获取芯片号对应的芯片模板
        short_sn = self.chip_template_getter.get_valid_chip_no(self.chip_name)
        self.stereo_chip = self.chip_template_getter.get_chip_grids(short_sn)
        self.is_stitched = self.ipr_manager.image_info.stitched_image
        if not self.is_stitched:
            self.image_root = src_data
        else:
            self.image_root = os.path.dirname(src_data)
        self.magnification = self.ipr_manager.image_info.scan_objective
        self.fov_location = self.ipr_manager.stitch.scope_stitch.global_loc
        self.image_dict = self.ipr_manager.research.fovs_tag
        if len(self.ipr_manager.research.fovs_tag) != 0:
            self.image_dict = dict()
            for row, col in np.ndindex(self.ipr_manager.research.fovs_tag.shape):
                path = self.ipr_manager.research.fovs_tag[row, col]
                if isinstance(path, bytes):
                    try:
                        path = path.decode('utf-8')
                        if os.path.exists(os.path.join(self.image_root, path)):
                            self.image_dict[rc_key(row, col)] = os.path.join(self.image_root, path)
                    except:
                        clog.info("row:{}, col:{} fov image is not exit".format(row, col))
        self.fov_size = (self.ipr_manager.image_info.fov_width, self.ipr_manager.image_info.fov_height)

    def check_qc_input(self, ):
        for key, val in self.image_dict.items():
            if not os.path.exists(val):
                raise Exception(f"{val} does not exist")

        if len(self.stereo_chip) == 0 or len(self.stereo_chip[1]) != 9:
            raise Exception("Stereo chip error")
        if not self.is_stitched and (
                len(self.fov_location) == 0 or len(self.fov_location.shape) != 3 or self.fov_location.shape[-1] != 2):
            raise Exception("Fov location error")
        if not isinstance(self.magnification, int) and not isinstance(self.magnification, float):
            raise Exception("magnification error")
        if self.fov_size is None or self.image_root is None or self.is_stitched is None:
            return 1
        return 0

    def init_image_qc(self):
        initialize_code = self.image_qc.initialize(
            tissue_detector_file=self.tissue_weight_path,
            pt_detector_file=self.pt_detect_weight_path,
            clarity_file=self.clarity_weight_path,
        )
        if initialize_code != 0:
            return 1
        clog.info(f"image qc initialization finished")
        # qc 赋值
        self.image_qc.set_is_stitched(self.is_stitched)
        self.image_qc.set_stain_type(self.stain_type)
        self.image_qc.set_stereo_chip(self.stereo_chip)
        self.image_qc.set_fov_loc(self.fov_location)  # scope stitch location
        self.image_qc.set_fov_size(self.fov_size)
        self.image_qc.set_magnification(self.magnification)

    def run_qc(self, save_path=None):
        self.init_image_qc()
        if self.check_qc_input() != 0:
            raise Exception(f"QC input error")

        if self.is_stitched:
            self.image_dict = os.path.basename(list(self.image_dict.values())[0])

        self.image_qc.run(
            image_root=self.image_root,
            image_map=self.image_dict,
            save_path=save_path
        )

        # self.write_to_ipr()
        return 0

    def set_zoo_dir(self, path: str):
        self.zoo_dir = path
        self.update_weight_path()

    def set_stain_type(self, stain_type: str):
        self.stain_type = stain_type.upper()

    def set_chip_name(self, chip_name: str):
        self.chip_name = chip_name

    def set_w_h(self, w, h):
        # 小图模式可以直接从显微镜配置文件读取
        if self.is_stitched:
            self.fov_size = (w, h)

    def set_magnification(self, m):
        self.magnification = m

    def set_debug_mode(self, d):
        self.debug = d
        self.image_qc.set_debug_mode(d)

    def set_save_file(self, f: str):
        self.save_dir = f

    def update_weight_path(self, ):
        self.tissue_weight_path = os.path.join(self.zoo_dir, ZOO['tissue'])
        self.pt_detect_weight_path = os.path.join(self.zoo_dir, ZOO['track_pt'])
        self.clarity_weight_path = os.path.join(self.zoo_dir, ZOO['clarity'])

    def write_to_ipr(self, f: h5py, research_group=None):
        # with h5py.File(self.ipr_save_path, 'a') as f:
        # same with old version
        qc_info = f["QCInfo"]
        qc_info.attrs['ClarityScore'] = self.image_qc.clarity_score
        qc_info.attrs['GoodFOVCount'] = -1
        qc_info.attrs[
            'QCPassFlag'] = False if self.image_qc.line_qc_flag is None else self.image_qc.line_qc_flag  # 只要能推出一个模板就能做tissue bin
        qc_info.attrs['StainType'] = self.stain_type
        qc_info.attrs['TrackLineScore'] = self.image_qc.track_score
        cps = qc_info['CrossPoints']
        tps_copy = self.image_qc.track_pts
        if self.image_qc.line_qc_flag:
            del tps_copy[self.image_qc.track_line_best_match[0]]
            tps_copy[self.image_qc.track_line_best_match[0]] = self.image_qc.track_line_best_match[1]
        if isinstance(tps_copy, dict):
            for row_column, points in tps_copy.items():
                if points[0]:
                    if self.image_qc.line_qc_flag and row_column == self.image_qc.track_line_best_match[0]:
                        cps.create_dataset(row_column, data=np.array(points))
                    else:
                        cps.create_dataset(row_column, data=np.array(points[0]))

        #  scope stitch
        qc_info.attrs["TrackLineChannel"] = self.TrackLineChannel
        qc_info.attrs["ScopeStitchQCScore"] = self.image_qc.mif_stitch_dict.get("ScopeStitchQCScore", -1)
        qc_info.create_dataset("ScopeStitchQCMatrix", data=self.image_qc.mif_stitch_dict.get("jitter_diff", -1))

        # Judge the stitch effect
        ScopeStitchQCPassFlag = True
        if self.image_qc.mif_stitch_dict.get("stitch_diff_max", -1) > 10:
            if self.image_qc.mif_stitch_dict.get("ScopeStitchQCScore", -1) < 0.7:
                ScopeStitchQCPassFlag = False

        qc_info.attrs["ScopeStitchQCPassFlag"] = ScopeStitchQCPassFlag  # todo, wait modification
        qc_info.attrs["TrackCrossQCPassFlag"] = self.image_qc.template_qc_recall > 0.1
        qc_info.attrs["TemplateScore"] = self.image_qc.template_qc_recall

        regist_info = f['Register']
        regist_info.attrs['Rotation'] = self.image_qc.rotation
        regist_info.attrs['ScaleX'] = self.image_qc.scale_x
        regist_info.attrs['ScaleY'] = self.image_qc.scale_y

        # stitch group
        stitch_info = f["Stitch"]
        if self.image_qc.mif_stitch_dict["BGIStitchFlag"]:
            stitch_info["BGIStitch"].attrs["StitchedGlobalHeight"] = self.image_qc.mif_stitch_dict[
                "StitchedGlobalHeight"]
            stitch_info["BGIStitch"].attrs["StitchedGlobalWidth"] = self.image_qc.mif_stitch_dict["StitchedGlobalWidth"]
            if "StitchedGlobalLoc" in stitch_info["BGIStitch"].keys():
                del stitch_info["BGIStitch/StitchedGlobalLoc"]
            stitch_info.create_dataset('BGIStitch/StitchedGlobalLoc', data=self.image_qc.stitch_fov_loc)
        if self.image_qc.line_qc_flag:
            stitch_info.attrs['TemplateSource'] = '_'.join(map(str, rc_key_trans(self.image_qc.src_fov_info[0])))

        image_info = f["ImageInfo"]
        image_info.attrs['FOVHeight'] = self.fov_size[1]
        image_info.attrs['FOVWidth'] = self.fov_size[0]
        image_info.attrs['QCResultFile'] = self.file_name
        image_info.attrs['STOmicsChipSN'] = self.chip_name

        # research
        if isinstance(research_group, h5py.Group):
            research_info = research_group
            stitch_research = research_info.require_group('Stitch')
            stitch_research.attrs['StitchDiffMax'] = self.image_qc.mif_stitch_dict.get("stitch_diff_max", -1)
            stitch_research.attrs['JitterDiffMax'] = self.image_qc.mif_stitch_dict.get("jitter_diff_max", -1)
            stitch_research.create_dataset('StitchDiff', data=self.image_qc.mif_stitch_dict.get("stitch_diff", -1))
            stitch_research.create_dataset('JitterDiff', data=self.image_qc.mif_stitch_dict.get("jitter_diff", -1))
            stitch_research.create_dataset('HorizontalJitter', data=self.image_qc.jitter[0])
            stitch_research.create_dataset('VerticalJitter', data=self.image_qc.jitter[1])

            stitch_research.attrs['TemplateMax'] = self.image_qc.template_max
            stitch_research.attrs['TemplateMean'] = self.image_qc.template_mean
            stitch_research.attrs['TemplateStd'] = self.image_qc.template_std
            stitch_research.attrs['TemplatePrecision'] = self.image_qc.template_qc_precision
            stitch_research.attrs['TemplateRecall'] = self.image_qc.template_qc_recall
            stitch_research.create_dataset('GlobalTemplate', data=np.array(self.image_qc.global_template))
            stitch_research.attrs['StitchRoiFov'] = self.image_qc.stitch_roi
            stitch_research.attrs['MicroscopeStitch'] = True
            stitch_research.create_dataset('StitchFovLocation', data=self.image_qc.stitch_fov_loc)

            # eval tissuebin or cellbin
            if self.image_qc.template_qc_recall > 0.1:
                stitch_research.attrs["TissueBinPassFlag"] = 1
            else:
                stitch_research.attrs["TissueBinPassFlag"] = 0

            if self.image_qc.template_qc_recall > 0.5 and self.image_qc.clarity_score > 0.5:
                stitch_research.attrs["CellBinPassFlag"] = 1
            else:
                stitch_research.attrs["CellBinPassFlag"] = 0

            # tissue seg
            tissue_research = research_info.require_group('tissue')
            if self.image_qc.fov_tissue_mask is not None:
                tissue_research.create_dataset('tissue_mask', data=self.image_qc.fov_tissue_mask)
            if self.image_qc.box_mosaic is not None:
                tissue_research.attrs['clarity_box'] = self.image_qc.box_mosaic

            # regist
            if self.image_qc.line_qc_flag and self.image_qc.line_qc_flag is not None:
                regist_research = research_info.require_group('Regist')
                regist_research.attrs['track_line_score'] = self.image_qc.track_line_score
                regist_research.create_dataset('track_pt_fov_mask', data=self.image_qc.track_pt_fov_mask)
                regist_best_track_line = regist_research.require_group('Best_track_line')
                regist_best_track_line.attrs['ScaleX'] = self.image_qc.track_line_best_match[2]
                regist_best_track_line.attrs['ScaleY'] = self.image_qc.track_line_best_match[3]
                regist_best_track_line.attrs['Rotation'] = self.image_qc.track_line_best_match[4]
                regist_research.create_dataset(self.image_qc.track_line_best_match[0],
                                               data=np.array(self.image_qc.track_line_best_match[1]))

            # clarity
            clarity_research = research_info.require_group('Clarity')
            clarity_research.attrs['counts'] = str(self.image_qc.clarity_counts)
            if self.image_qc.debug and len(self.image_qc.clarity_heatmap) > 0:
                cv2.imwrite(os.path.join(self.save_dir, f"{self.file_name}.tif"), self.image_qc.clarity_heatmap)
                self.image_qc.clarity_cluster.savefig(os.path.join(self.save_dir, f"{self.file_name}.png"))
            # clarity_research.create_dataset('clarity_heatmap', data=self.image_qc.clarity_heatmap)

    def get_time_stamp(self):
        times = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        return times


class MIFQC():
    def __init__(self, stain_type, src_data, chip_name, fov_size, magnification, save_dir, debug_mode, zoo_dir,
                 config_path, track_line_channel="DAPI"):
        self.stain_type = stain_type
        self.src_data = src_data
        self.chip_name = chip_name
        self.fov_size = fov_size
        self.magnification = magnification
        self.save_dir = save_dir
        self.debug_mode = debug_mode
        self.zoo_dir = zoo_dir
        self.config_path = config_path
        self.mono_fluorescence = False

        # other
        self.file_name = None
        self.ipr_save_path = None
        self.dapi_image = None
        self.dapi_image_path = None
        self.dapi_global_location = None
        self.TrackLineChannel = track_line_channel

        # run
        self.qc_runner: QcRunner = QcRunner()
        self.calibrate = FFTRegister()
        os.makedirs(self.save_dir, exist_ok=True)
        self.qc_runner.set_save_file(save_dir)  # required
        self.qc_runner.set_zoo_dir(zoo_dir)  # required
        self.qc_runner.set_chip_name(chip_name)  # required
        self.qc_runner.set_stain_type(stain_type)  # required
        self.qc_runner.image_qc.initialize_json(config_path)  # required

    def track_line_image_qc(self, dapi_path):
        """
        start dapi qc including stitch, clarity, trackline and fune-tune
        Returns:

        """
        # init ipr
        self.file_name = "_".join([self.chip_name, self.qc_runner.get_time_stamp(), VERSION])
        dapi_output = os.path.join(self.save_dir, self.file_name, self.TrackLineChannel)


        self.qc_runner.read_microscope(dapi_path)  # required
        w, h = self.fov_size
        self.qc_runner.set_w_h(w, h)  # 要在读显微镜之后set fov size, 只有大图需要做这个
        self.qc_runner.set_magnification(self.magnification)  # motic可读到，czi不知道
        self.qc_runner.set_debug_mode(self.debug_mode)

        self.ipr_save_path = os.path.join(self.save_dir, self.file_name + ".ipr")
        self.qc_runner.ipr_manager.write(self.ipr_save_path)

        self.qc_runner.run_qc()
        self.stain_type = self.TrackLineChannel

        self.dapi_global_location = self.qc_runner.image_qc._fovs_loc
        self.dapi_image = self.qc_runner.image_qc._mosaic
        if self.debug_mode:
            os.makedirs(dapi_output, exist_ok=True)
            self.dapi_image_path = os.path.join(dapi_output,
                                                "{}_{}_IF.tif".format(self.chip_name, self.TrackLineChannel))
            tifffile.imwrite(self.dapi_image_path, self.dapi_image)
            clog.info("stitched image saved in {}".format(self.dapi_image_path))

        self.qc_runner.image_qc._mosaic = None
        # write ipr
        with h5py.File(self.ipr_save_path, 'a') as ipr:
            if self.TrackLineChannel not in ipr.keys():
                ipr.create_group(self.TrackLineChannel, track_order=True)
            if "Research" not in ipr.keys():
                ipr.create_group("Research", track_order=True)
            DAPI_group = ipr[self.TrackLineChannel]
            sub_group = ["CellSeg", "ImageInfo", "QCInfo", "Register", "Stitch", "TissueSeg", "Calibration", "Preview"]
            for sub_name in ipr.keys():
                if sub_name in sub_group:
                    ipr.move(sub_name, "{}/{}".format(self.TrackLineChannel, sub_name))

            ipr.create_group("Research/{}".format(self.TrackLineChannel), track_order=True)
            research_group = ipr["Research/{}".format(self.TrackLineChannel)]
            self.qc_runner.file_name = self.file_name
            self.qc_runner.write_to_ipr(DAPI_group, research_group)

    def eval_calibration(self, calibration: dict):
        """
        eval calibration result
        校准要求：图像相似度>1%; 尺度差异在：[0.8~1.2]之间; 角度差异：[-25, 25]之间；拼接偏差：100 pixel
        """
        CalibrationQCPassFlag = 1
        if calibration["offset"][0] > 100 or calibration["offset"][1] > 100:
            CalibrationQCPassFlag *= 0
            clog.info("offset failed: {}".format(calibration["offset"]))

        if calibration["scale"] > 1.2 or calibration["scale"] < 0.8:
            CalibrationQCPassFlag *= 0
            clog.info("scale failed: {}".format(calibration["scale"]))

        if calibration["angle"] > 25 or calibration["angle"] < -25:
            CalibrationQCPassFlag *= 0
            clog.info("angle failed: {}".format(calibration["angle"]))

        if calibration["confi"] < 0.01:
            CalibrationQCPassFlag *= 0
            clog.info("confi failed: {}".format(calibration["confi"]))
        return CalibrationQCPassFlag

    def if_calibrate_qc(self, if_path, if_type):
        """
        start if calibrate. There are three scenarios, including big image mode, single fluorescent channel and
         multifluorescent channel;
        Args:
            if_type: IF type of stain
            if_path: IF file path or msp

        Returns:

        """
        self.qc_runner.read_microscope(if_path)  # required

        w, h = self.fov_size
        self.qc_runner.set_w_h(w, h)  # 要在读显微镜之后set fov size, 只有大图需要做这个
        self.qc_runner.set_magnification(self.magnification)  # motic可读到，czi不知道
        self.qc_runner.set_debug_mode(self.debug_mode)
        self.stain_type = if_type

        if_output_path = os.path.join(self.save_dir, self.file_name, "IF", if_type)
        if self.debug_mode: os.makedirs(if_output_path, exist_ok=True)

        sub_group = ["CellSeg", "ImageInfo", "QCInfo", "Register", "Stitch", "TissueSeg", "Calibration", "Preview"]
        for sub_name in self.qc_runner.ipr_manager.group_names:
            if sub_name not in sub_group:
                self.qc_runner.ipr_manager.group_names.remove(sub_name)
        self.qc_runner.ipr_manager.write(self.ipr_save_path, mode="a")

        self.qc_runner.init_image_qc()
        self.qc_runner.image_qc.image_map = self.qc_runner.image_dict

        # self.qc_runner.image_qc._prepare(save_path=_save_path, fft_stitch=self.mono_fluorescence)
        if self.qc_runner.is_stitched:
            self.qc_runner.image_qc.image_map = os.path.basename(list(self.qc_runner.image_dict.values())[0])

        # start calibration
        if self.mono_fluorescence:
            self.qc_runner.image_qc.stitch_fov_loc = self.dapi_global_location
            calibrate_result = {"offset": [0, 0],
                                "scale": 1,
                                "angle": 0,
                                "confi": 1}
            if self.debug_mode:
                if_img_output_path = os.path.join(if_output_path, "{}_{}_IF.tif".format(self.chip_name, if_type))
                if_img = self.qc_runner.image_qc.save_moasic(self.qc_runner.image_qc.image_map,
                                                             self.dapi_global_location)
                tifffile.imwrite(if_img_output_path, if_img)
                clog.info("{} stitch image save to: {}".format(if_type, if_img_output_path))
                del if_img
        else:
            self.qc_runner.image_qc._prepare()
            if_img, calibrate_result = self.calibration(self.dapi_image, self.qc_runner.image_qc._mosaic)
            if self.debug_mode:
                if_img_output_path = os.path.join(if_output_path, "{}_{}_IF.tif".format(self.chip_name, if_type))
                tifffile.imwrite(if_img_output_path, self.qc_runner.image_qc._mosaic)
                clog.info("{} stitch image save to: {}".format(if_type, if_img_output_path))

                if_img = self.calibrate.transform_img_vips(if_img,
                                                           offset=calibrate_result["offset"],
                                                           scale=calibrate_result["scale"],
                                                           angle=calibrate_result["angle"],
                                                           dst_shape=calibrate_result["dst_shape"],
                                                           )
                register_if_path = os.path.join(if_output_path, "Register_{}_{}_IF.tif".format(self.chip_name, if_type))
                tifffile.imwrite(register_if_path, if_img)
                clog.info("{} register image save to: {}".format(if_type, register_if_path))
            del if_img

        with h5py.File(self.ipr_save_path, 'a') as ipr:
            if if_type not in ipr.keys():
                ipr.create_group(if_type, track_order=True)
            if "Research" not in ipr.keys():
                ipr.create_group("Research", track_order=True)
            IF_group = ipr[if_type]
            for sub_name in ipr.keys():
                if sub_name in sub_group:
                    ipr.move(sub_name, "{}/{}".format(if_type, sub_name))

            # calibration
            if "Calibration" not in IF_group.keys():
                IF_group.create_group("Calibration/BGI", track_order=True)
                IF_group.create_group("Calibration/Scope", track_order=True)
            IF_group_calibrate = IF_group["Calibration/BGI"]
            IF_group_calibrate.attrs["OffsetX"] = calibrate_result["offset"][0]
            IF_group_calibrate.attrs["OffsetY"] = calibrate_result["offset"][1]
            IF_group_calibrate.attrs["Scale"] = calibrate_result["scale"]
            IF_group_calibrate.attrs["Angle"] = calibrate_result["angle"]
            IF_group_calibrate.attrs["Confidence"] = calibrate_result["confi"]
            IF_group["Calibration"].attrs["CalibrationQCPassFlag"] = self.eval_calibration(calibrate_result)

            ipr.create_group("Research/{}".format(if_type), track_order=True)
            research_group = ipr["Research/{}".format(if_type)]
            self.qc_runner.file_name = self.file_name
            self.qc_runner.write_to_ipr(IF_group, research_group)

    def calibration(self, dapi_img, if_img):
        """
        start calibration based on dapi image
        Args:
            dapi_img: np.ndarray, must be 2 dimension
            if_img: np.ndarray, must be 2 dimension

        Returns: dict, which including offset, scale, angle, dst_shape and confidence
        """
        self.calibrate.debug = self.debug_mode
        dapi_img, if_img = self.calibrate.pad_same_image(dapi_img, if_img)
        result = self.calibrate.calibration(dapi_img, if_img, similarty=True)
        return if_img, result

    def parse_channel_info(self):
        self.channel_infos = {}

        src_datas = self.src_data.split(",")
        if len(src_datas) == 1 and ".tif" not in src_datas[0]:  # 多荧光通道
            msp = MicroscopeBaseFileFactory().create_microscope_file(self.src_data)
            self.channel_infos = msp.parse_all_img_info()
            self.mono_fluorescence = True
            self.channel_infos["msp"] = msp
        else:  # 单荧光通道
            # self.TrackLineChannel = "DAPI"
            for i, src_data in enumerate(src_datas):
                channel_info = {}
                file_name = os.path.basename(src_data).split(".")[0]
                if self.chip_name == file_name:
                    channel_info["Path"] = src_data
                    channel_info["Name"] = "DAPI"
                    self.channel_infos["DAPI"] = channel_info
                elif self.chip_name in file_name and "_IF" in file_name:
                    if_name = file_name.split("_")[-2]
                    channel_info["Path"] = src_data
                    channel_info["Name"] = if_name
                    self.channel_infos[if_name] = channel_info
                else:
                    raise ValueError("{} is not legal".format(file_name))

        self.channel_infos["IF_groups"] = [i for i in list(self.channel_infos.keys()) if
                                           i not in [self.TrackLineChannel, "msp", "IF_groups"]]

    def mif_qc_run(self):
        """
        1. parase channel infor
        2. track line channel eval (stitch, track pts, fov class, clarity ...)
        3. IF channel eval (calibration)
        """
        time_parse = time.time()
        self.parse_channel_info()

        if self.TrackLineChannel not in list(self.channel_infos.keys()):
            clog.info("miss trackline Image, {}".format(self.channel_infos.keys()))
            return 1
        clog.info("parse channel info cost time: {}".format(time.time() - time_parse))

        self.qc_runner.TrackLineChannel = self.TrackLineChannel
        trackline_eval_time = time.time()
        clog.info("Start trackLineChannel QC eval...")
        if self.mono_fluorescence:
            track_channel_path = self.channel_infos['msp']
            track_channel_path.current_channel = self.TrackLineChannel
        else:
            track_channel_path = self.channel_infos[self.TrackLineChannel]["Path"]
        self.track_line_image_qc(track_channel_path)
        clog.info("trackline channel eval cost time: {}".format(time.time() - trackline_eval_time))

        if_eval_time = time.time()
        for if_type in tqdm.tqdm(self.channel_infos['IF_groups'], desc="Start IF Channel QC Eval"):
            if self.mono_fluorescence:
                clog.info("multi fluorescence mode")
                if_image_path = self.channel_infos["msp"]
                if_image_path.current_channel = if_type
            else:
                clog.info("big image mode or single fluorescence mo")
                if_image_path = self.channel_infos[if_type]["Path"]
            self.if_calibrate_qc(if_image_path, if_type)
        clog.info("if channel eval cost time: {}".format(time.time() - if_eval_time))
        # summary eval
        with h5py.File(self.ipr_save_path, "a") as ipr:
            trackline_group = ipr[self.TrackLineChannel]["QCInfo"]
            trackline_QCPassFlag = trackline_group.attrs['QCPassFlag']

            if_QCPassFlag = True
            for if_type in self.channel_infos["IF_groups"]:
                if_group = ipr[if_type]
                if if_group["QCInfo"].attrs["QCPassFlag"] == 0:
                    if_QCPassFlag = False

                if if_group["Calibration"].attrs["CalibrationQCPassFlag"] == 0:
                    if_QCPassFlag = False

            if trackline_QCPassFlag and if_QCPassFlag:
                trackline_group.attrs["QCPassFlag"] = 1
            else:
                trackline_group.attrs["QCPassFlag"] = 0


def main():
    usage = ''
    import argparse
    ArgParser = argparse.ArgumentParser(usage=usage)
    ArgParser.add_argument("--version", action="version", version=VERSION)
    ArgParser.add_argument("-i", "--input", action="store", dest="input_path", type=str, required=True,
                           help="Image folder.")
    ArgParser.add_argument("-o", "--output", action="store", dest="output_path", type=str, required=True,
                           help="QC results output folder.")
    ArgParser.add_argument("-c", "--chip", action="store", dest="chip_name", type=str, required=True, help="Chip name")
    ArgParser.add_argument("-z", "--zoo", action="store", dest="zoo_dir", type=str, required=True,
                           default='', help="DNN weights dir")
    ArgParser.add_argument("-j", "--json", action="store", dest="json_path", type=str, required=True,
                           help="Config file path")
    ArgParser.add_argument("-s", "--stain", action="store", dest="stain_type", type=str, default='DAPI',
                           help="Stain type. (SSDNA, DAPI, CZI).")
    ArgParser.add_argument("-t", "--tracklinechannel", action="store", dest="track_line_channel", type=str,
                           default='DAPI',
                           help="Track line channel. (SSDNA, DAPI, AF488).")
    ArgParser.add_argument("-m", "--magnification", action="store", dest="magnification", type=int, default=10,
                           help="10X or 20X")
    ArgParser.add_argument("-f", "--fov_size", action="store", dest="fov_size", type=int, default=(2000, 2000),
                           help="Fov size used for large image")
    ArgParser.add_argument("-d", "--debug", action="store", dest="debug", type=bool, default=False, help="Debug mode")

    (para, args) = ArgParser.parse_known_args()

    clog.log2file(para.output_path, "mif_cellbin_qc.log")
    mif_qc = MIFQC(
        stain_type=para.stain_type,
        src_data=para.input_path,
        chip_name=para.chip_name,
        fov_size=para.fov_size,
        magnification=para.magnification,
        save_dir=para.output_path,
        debug_mode=para.debug,
        zoo_dir=para.zoo_dir,
        config_path=para.json_path,
        track_line_channel=para.track_line_channel
    )
    clog.info("Track line channel: {}".format(mif_qc.TrackLineChannel))
    start_time = time.time()
    mif_qc.mif_qc_run()
    end_time = time.time()
    clog.info("all cost time: {}".format(end_time - start_time))


if __name__ == '__main__':
    main()

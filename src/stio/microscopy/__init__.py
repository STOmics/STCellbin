import enum
from stio.utils.image import Image
from stio.utils import clog

import numpy as np


class SlideType(enum.Enum):
    Motic = 'motic'
    Zeiss = 'zeiss'
    Leica = 'leica'
    Olympus = 'olympus'
    ChangGuang = 'changguang'
    DaXingCheng = 'daxingcheng'
    Unknown = 'unknown'


class DeviceInfo(object):
    """
    Attributes:
        app_file_ver: A string that indicate the version of the scanner App.
        scan_time: str = ''
        background_balance = None
        color_enhancement = None
        distortion_correction = None
        contrast = None
        brightness = None
        gamma = None
        gamma_shift = None
        white_balance = None
        sharpness = None
        rgb_scale = None
        model = None
    """

    def __init__(self):
        self.scanner_app: str = ''
        self.app_file_ver: str = ''
        self.background_balance = None
        self.color_enhancement = None
        self.distortion_correction = None
        self.contrast = None
        self.brightness = None
        self.gamma = None
        self.gamma_shift = None
        self.white_balance = None
        self.sharpness = None
        self.rgb_scale = None
        self.model = None
        self.manufacturer: str = ''
        self.device_sn: str = ''
        self.gain = None
        self.illuminance = None


class MicroscopeBaseFile(object):
    def __init__(self):
        self.thumbnail = None
        self.suffix: str = ''
        self._file_path: str = ''
        self.stitched_image = None
        self.fov_path_buffer = None
        self.device = DeviceInfo()

    def get_fovs_tag(self, ):
        return self.fov_path_buffer


class MicroscopeLargeImage(MicroscopeBaseFile):
    # designed for stitched image
    def __init__(self):
        super(MicroscopeLargeImage, self).__init__()
        self.device.manufacturer = SlideType.Unknown.value
        self.is_stitched = True
        self.mosaic_height: int = 0
        self.mosaic_width: int = 0
        self.mosaic_channel: int = 0
        self.mosaic_dtype = None

    def bit_depth(self, ):
        if self.mosaic_dtype == np.uint16:
            return 16
        elif self.mosaic_dtype == np.uint8:
            return 8
        else:
            return 0

    def _check(self, ):
        import os
        if not os.path.exists(self._file_path): return 1
        return 0

    def read(self, file_path):
        self._file_path = file_path
        clog.info('Try to read file {}'.format(file_path))
        check_flag = self._check()
        check_info = ['success', 'path not exists', 'not find the exact location of the images']
        if check_flag:
            clog.error('Validation on input path failed: {}'.format(check_info[check_flag]))
            return 1
        else:
            clog.info('Validation on input path success')
        self._init()

    def _init(self, printf=True):
        import os
        img = Image()
        img.read(self._file_path)
        self.mosaic_width = img.width
        self.mosaic_height = img.height
        self.mosaic_channel = img.channel
        self.mosaic_dtype = img.dtype
        self.suffix = img.suffix

        self.fov_path_buffer = np.array([
            [os.path.basename(self._file_path)]
        ], dtype='S256')
        if printf:
            self._print_info()

    def _print_info(self, ):
        mosaic_info = 'width: {}\nheight: {}\nchannel: {}\ndtype: {}\nsuffix: {}'.format(
            self.mosaic_width, self.mosaic_height, self.mosaic_channel, self.mosaic_dtype, self.suffix)
        clog.info('[Mosaic] parameters:\n{}'.format(mosaic_info))


class ScanningInfo(object):
    def __init__(self):
        self.fov_rows: int = 0
        self.fov_cols: int = 0
        self.fov_location = None
        self.mosaic_width: int = 0
        self.mosaic_height: int = 0
        self.fov_height: int = 0
        self.fov_width: int = 0
        self.fov_channel: int = 0
        self.fov_dtype = None
        self.fov_images: list = list()
        self.fov_count: int = 0
        self.exposure_time = None
        self.scan_time: str = ''
        self.objective: int = 10
        self.scale_x: float = 1.0
        self.scale_y: float = 1.0
        self.overlap: float = 0.1


class MicroscopeSmallImage(MicroscopeBaseFile):
    def __init__(self):
        super(MicroscopeSmallImage, self).__init__()
        self.scan = ScanningInfo()

    def bit_depth(self, ):
        if self.scan.fov_dtype == np.uint16:
            return 16
        elif self.scan.fov_dtype == np.uint8:
            return 8
        else:
            return 0

    def _init(self, printf=True):
        small_image = self.scan.fov_images[0]
        img = Image()
        img.read(small_image)
        self.scan.fov_width = img.width
        self.scan.fov_height = img.height
        self.scan.fov_channel = img.channel
        self.scan.fov_dtype = img.dtype
        self.scan.suffix = img.suffix
        self.scan.fov_count = len(self.scan.fov_images)
        if printf: self._print_info()

    def _print_info(self):
        fov_info = '\nwidth: {}\nheight: {}\nchannel: {}\ndtype: {}\nfile suffix: {}'.format(
            self.scan.fov_width, self.scan.fov_height, self.scan.fov_channel, self.scan.fov_dtype, self.suffix)
        clog.info('Tile parameters: {}'.format(fov_info))

        wsi_info = 'tile row: {}\ntile col: {}\ntile count: {}\nmosaic width: {}\nmosaic height: {}'.format(
            self.scan.fov_rows, self.scan.fov_cols, self.scan.fov_count,
            self.scan.mosaic_width, self.scan.mosaic_height
        )
        clog.info('WSI(Whole Slide Image) parameters:\n{}'.format(wsi_info))

        extra_info = 'overlap: {}\nscale (XY): {}\nscan time: {}\nexposure (sec): {}'.format(self.scan.overlap,
                                                                                             (self.scan.scale_x,
                                                                                              self.scan.scale_y),
                                                                                             self.scan.scan_time,
                                                                                             self.scan.exposure_time)
        clog.info('[{}] SerialNo-{}, {} ver-{}:\n{}'.format(
            self.device.manufacturer, self.device.device_sn, self.device.scanner_app, self.device.app_file_ver,
            extra_info))

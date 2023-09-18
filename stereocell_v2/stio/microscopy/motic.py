import os
import numpy as np
from stio.microscopy import MicroscopeSmallImage, SlideType
from stio.utils.file_manager import search_files
from stio.utils import clog


class MoticMicroscopeFile(MicroscopeSmallImage):
    def __init__(self):
        super(MoticMicroscopeFile, self).__init__()
        self.device.manufacturer = SlideType.Motic.value
        self.images_path: str = ''
        self.ini_1: str = ''
        self.ini_info: str = ''
        self.jpg_3 = ''

    @staticmethod
    def file_rc_index(file_name):
        tags = os.path.splitext(file_name)[0].split('_')
        xy = list()
        for tag in tags:
            if (len(tag) == 4) and tag.isdigit(): xy.append(tag)
        c = xy[0]
        r = xy[1]
        return [int(r), int(c)]

    def _check(self, ):
        assert os.path.exists(self._file_path)
        assert os.path.isdir(self._file_path)
        need = ['1.ini', 'info.ini', '3.jpg']
        image_support = ['.jpg', '.png', '.tif']
        for it in need:
            need_file = os.path.join(self._file_path, it)
            if not os.path.exists(need_file): return 1
        cand = os.listdir(self._file_path)
        images_dir = list()
        for it in cand:
            sub = os.path.join(self._file_path, it)
            if os.path.isdir(sub):
                images = search_files(sub, exts=image_support)
                if len(images): images_dir.append(sub)
        if len(images_dir) != 1: return 2
        self.images_path = os.path.join(self._file_path, images_dir[0])
        self.scan.fov_images = search_files(self.images_path, exts=image_support)
        self.ini_1 = os.path.join(self._file_path, '1.ini')
        self.ini_info = os.path.join(self._file_path, 'info.ini')
        self.jpg_3 = os.path.join(self._file_path, '3.jpg')

        return 0

    def _set_fov_tags(self, ):
        def filename2index(file_name):
            tags = os.path.splitext(file_name)[0].split('_')
            xy = list()
            for tag in tags:
                if (len(tag) == 4) and tag.isdigit(): xy.append(tag)
            x_str = xy[0]
            y_str = xy[1]
            return [int(y_str), int(x_str)]

        self.fov_path_buffer = np.empty((self.scan.fov_rows, self.scan.fov_cols), dtype='S256')
        for img in self.scan.fov_images:
            c_, r_ = filename2index(img)
            self.fov_path_buffer[r_, c_] = os.path.join(os.path.basename(os.path.dirname(img)), os.path.basename(img))

    def _from_ini_1(self, ):
        with open(self.ini_1, 'rb') as fd:
            import struct

            struct.unpack('<4s3i', fd.read(16))
            rect = struct.unpack('<4i', fd.read(16))
            self.scan.mosaic_height = rect[3]
            self.scan.mosaic_width = rect[2]
            self.scan.fov_rows, self.scan.fov_cols = struct.unpack('<2i', fd.read(8))
            self.fov_location = np.zeros((self.scan.fov_rows, self.scan.fov_cols, 2), dtype=int)

            for i in range(self.scan.fov_rows):
                for j in range(self.scan.fov_cols):
                    v = struct.unpack('<2i4d8i4?', fd.read(76))
                    uai = UnitAssemblyInfo()
                    uai.set_from_tuple(v)
                    # uai.print_info(header=False)
                    fd.read(4)
                    self.fov_location[i, j] = [int(uai.x), int(uai.y)]
            self.scan.overlap = int.from_bytes(fd.read(4), 'little')

    @staticmethod
    def get_value(cp, sec, op):
        if cp.has_option(sec, op): return cp.get(sec, op)
        else: return None

    def _from_ini_info(self, ):
        from configparser import ConfigParser

        cp = ConfigParser()
        cp.read(self.ini_info, encoding='utf-8')
        _, self.device.app_file_ver = self.get_value(cp, 'info', 'ScanMachineInfo').split(' ')
        self.device.scanner_app = _.split('\\')[1]
        self.device.device_sn = self.get_value(cp, 'info', 'DeviceSN')
        a, b, c, d = self.device.app_file_ver.strip('b').split('.')
        if int(d) < 7:
            clog.info('The scanning software version is too low, need >= v1.0.0.7b')
            return 1

        # Assigned: AppFileVer
        self.scan.scan_time = self.get_value(cp, 'info', 'createTimeText')  # createTimeText

        red_scale = float(self.get_value(cp, 'Property', 'video.RedScale'))
        green_scale = float(self.get_value(cp, 'Property', 'video.GreenScale'))
        blue_scale = float(self.get_value(cp, 'Property', 'video.BlueScale'))
        self.device.rgb_scale = [red_scale, green_scale, blue_scale]
        self.device.brightness = int(self.get_value(cp, 'Property', 'video.Brightness'))
        self.device.color_enhancement = bool(int(self.get_value(cp, 'Property', 'video.ColorEnhancement')))
        self.device.contrast = bool(int(self.get_value(cp, 'Property', 'video.Contrast')))
        self.device.gamma = float(self.get_value(cp, 'Property', 'video.Gamma'))
        self.device.gamma_shift = bool(int(self.get_value(cp, 'Property', 'video.GammaShift')))
        self.device.sharpness = bool(int(self.get_value(cp, 'Property', 'video.Sharpness')))
        self.device.distortion_correction = bool(int(self.get_value(cp, 'Property', 'video.Distort')))
        self.device.background_balance = bool(int(self.get_value(cp, 'Property', 'video.Background')))
        self.device.model = None

        # self.white_balance = None
        # self.model = None

        self.scan.objective = float(self.get_value(cp, 'AssemblyInfo', 'lens'))
        self.scan.scale_x = self.scan.scale_y = float(self.get_value(cp, 'info', 'scale'))
        self.scan.exposure_time = float(self.get_value(cp, 'Property', 'video.Exposure'))
        self.scan.overlap = float(self.get_value(cp, 'AssemblyInfo', 'Overlap'))
        self.device.gain = None
        self.device.illuminance = None

    def read(self, file_path):
        self._file_path = file_path
        clog.info('Try to read file {}'.format(file_path))
        check_flag = self._check()
        check_info = ['success', 'path not exists', 'not find the exact location of the images']
        if check_flag:
            clog.error('Validation on input path failed: {}'.format(check_info[check_flag]))
            return 1
        else: clog.info('Validation on input path success')
        self._from_ini_1()
        self._from_ini_info()
        self._init()
        self._set_fov_tags()


class UnitAssemblyInfo(object):
    def __init__(self):
        self.row: int = 0
        self.col: int = 0
        self.x: float = 0.0
        self.y: float = 0.0
        self.z: float = 0.0
        self.focus_factor: float = 0.0
        self.left_dx: int = 0
        self.left_dy: int = 0
        self.top_dx: int = 0
        self.top_dy: int = 0
        self.right_dx: int = 0
        self.right_dy: int = 0
        self.bottom_dx: int = 0
        self.bottom_dy: int = 0
        self.link_left: bool = False
        self.link_right: bool = False
        self.link_top: bool = False
        self.link_bottom: bool = False

    def set_from_tuple(self, tpl):
        dct = self.__dict__.copy()
        ind = 0
        for attr in dct:
            setattr(self, attr, tpl[ind])
            ind += 1

    def print_info(self, header=True):
        dct = self.__dict__.copy()
        if header:
            print(dct.keys())
        else:
            print(dct.values())


def main():
    import datetime
    clog.set_level(clog.INFO)
    # t = datetime.datetime.now()
    # file_name = 'cellbin{}-{}'.format(os.getpid(), t.strftime('%Y%m%d%H%M%S.log'))
    # clog.log2file(out_dir=r'./', filename=file_name)

    mmf = MoticMicroscopeFile()
    # mmf.read(file_path=r'D:\data\StereoCellStitching\StereoCell\SS200000975BR_A6')
    mmf.read(file_path=r'D:\data\guojing\221107\SS200001153BR_D5')


if __name__ == '__main__':
    main()

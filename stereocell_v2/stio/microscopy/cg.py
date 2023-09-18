import os
import numpy as np
from stio.microscopy import MicroscopeSmallImage, SlideType
from stio.utils.file_manager import search_files
from stio.utils import clog


class ChangGuangMicroscopeFile(MicroscopeSmallImage):
    def __init__(self):
        super(ChangGuangMicroscopeFile, self).__init__()
        self.device.manufacturer = SlideType.ChangGuang.value
        self.images_path: str = ''
        self.info_json: str = ''

    def _check(self, ):
        assert os.path.exists(self._file_path)
        assert os.path.isdir(self._file_path)
        need = ['info.json']
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
        self.info_json = os.path.join(self._file_path, 'info.json')
        return 0

    def _from_info_json(self, ):
        import json
        with open(self.info_json, 'r') as fd:
            dct = json.load(fd)
            self.scan.mosaic_height = dct['Info']['image_h']
            self.scan.mosaic_width = dct['Info']['image_w']
            self.scan.fov_rows, self.scan.fov_cols = [dct['Info']['row'], dct['Info']['col']]

            self.fov_location = np.array(dct['fovs_pos'])
            self.scan.overlap = dct['Info']['overlap']

            self.device.app_file_ver = dct['Version']
            self.scan.scan_time = dct['create_time']
            # self.hi.BitDepth = \
            #     int(len(bin(int(self.get_value(cp, 'Property', 'video.MaxGreyLevel')))) - 2)
            # red_scale = float(self.get_value(cp, 'Property', 'video.RedScale'))
            # green_scale = float(self.get_value(cp, 'Property', 'video.GreenScale'))
            # blue_scale = float(self.get_value(cp, 'Property', 'video.BlueScale'))
            # self.si.rgb_scale = [red_scale, green_scale, blue_scale]
            # self.si.brightness = int(self.get_value(cp, 'Property', 'video.Brightness'))
            # self.si.color_enhancement = bool(int(self.get_value(cp, 'Property', 'video.ColorEnhancement')))
            # self.si.contrast = bool(int(self.get_value(cp, 'Property', 'video.Contrast')))
            # self.si.gamma = float(self.get_value(cp, 'Property', 'video.Gamma'))
            # self.si.gamma_shift = bool(int(self.get_value(cp, 'Property', 'video.GammaShift')))
            # self.si.sharpness = bool(int(self.get_value(cp, 'Property', 'video.Sharpness')))
            # self.si.distortion_correction = bool(int(self.get_value(cp, 'Property', 'video.Distort')))
            # self.si.background_balance = bool(int(self.get_value(cp, 'Property', 'video.Background')))
            # self.model = None

            # self.white_balance = None
            # self.model = None
            self.device.scanner_app = dct['Info']['camera_type']

            self.device.device_sn = dct['Info']['device_sn']
            self.scan.objective = dct['Info']['objective']
            self.scan.scale_x = self.scan.scale_y = dct['Info']['scale']
            self.scan.exposure_time = dct['cost_time_s']
            self.scan.overlap = dct['Info']['overlap']
            self.device.gain = dct['Info']['camera_gain_db']
            # self.hi.illuminance = None

    def read(self, file_path):
        self._file_path = file_path
        clog.info('Try to read file {}'.format(file_path))
        check_flag = self._check()
        check_info = ['success', 'path not exists', 'not find the exact location of the images']
        if check_flag:
            clog.error('Validation on input path failed: {}'.format(check_info[check_flag]))
            return 1
        else: clog.info('Validation on input path success')
        self._from_info_json()
        self._init()


def main():
    import datetime
    clog.set_level(clog.INFO)
    t = datetime.datetime.now()
    file_name = 'cellbin{}-{}'.format(os.getpid(), t.strftime('%Y%m%d%H%M%S.log'))
    # clog.log2file(out_dir=r'./', filename=file_name)

    mmf = ChangGuangMicroscopeFile()
    # mmf.read(file_path=r'D:\data\StereoCellStitching\StereoCell\SS200000975BR_A6')
    mmf.read(file_path=r'D:\data\CHG_cellbin\CHG\SS200000369BL_B4\SS200000369BL_B4')


if __name__ == '__main__':
    main()

import os
import numpy as np
from stio.microscopy import MicroscopeSmallImage, SlideType
from stio.utils.file_manager import search_files
from stio.utils import clog


class DaXingChengMicroscopeFile(MicroscopeSmallImage):
    def __init__(self):
        super(DaXingChengMicroscopeFile, self).__init__()
        self.device.manufacturer = SlideType.DaXingCheng.value
        self.images_path: str = ''
        self.sitcb_cfg: str = ''

    def _check(self, ):
        assert os.path.exists(self._file_path)
        assert os.path.isdir(self._file_path)
        need = ['ScanInfoToCellBin.cfg', 'PrescanImage.tif']
        image_support = ['.jpg', '.png', '.tif']
        for it in need:
            need_file = os.path.join(self._file_path, it)
            if not os.path.exists(need_file): return 1
        cand = os.listdir(self._file_path)
        images_dir = list()
        for it in cand:
            sub = os.path.join(self._file_path, it, 'Images')
            if os.path.isdir(sub):
                images = search_files(sub, exts=image_support)
                if len(images): images_dir.append(sub)
        if len(images_dir) != 1: return 2
        self.images_path = os.path.join(self._file_path, images_dir[0])
        self.scan.fov_images = search_files(self.images_path, exts=image_support)
        self.sitcb_cfg = os.path.join(self._file_path, 'ScanInfoToCellBin.cfg')
        return 0

    def _from_cfg(self, ):
        import json
        with open(self.sitcb_cfg, 'r', encoding='utf-8') as fd:
            dct = json.load(fd)
            self.scan.scale_x, self.scan.scale_y = [dct['Scale'], dct['Scale']]
            self.scan.scan_time = dct['ScanTime']
            self.scan.overlap = dct['Overlap']
            self.scan.exposure_time = dct['ExposureTime']
            self.device.device_sn = dct['DeviceSN']
            self.device.app_file_ver = dct['AppFileVer']
            self.device.scanner_app = dct['AppName']
            # TODO: Indeterminate upstream specification

    def _from_image_dir(self, ):
        # 2022.4.22  -3    -2.1.0.30.24.0.2022  -04 -22_17 -40 -25 -042
        # 2022.04.21 -quar -5 -2.1.0.7.17.0.2022 -04 -21_10 -42 -10 -424

        imgs_name_arr = list()
        for img_name in self.scan.fov_images:
            # a, b, c, col, row, d, e = img_name.split('-')[3].split('.')
            a, col, row, e = img_name.split('_')
            imgs_name_arr.append([int(row), int(col)])
        imgs_name_arr = np.array(imgs_name_arr)
        fov_rows1 = np.max(imgs_name_arr[:, 0])
        fov_cols1 = np.max(imgs_name_arr[:, 1])
        fov_rows0 = np.min(imgs_name_arr[:, 0])
        fov_cols0 = np.min(imgs_name_arr[:, 1])
        self.scan.fov_rows = fov_rows1 - fov_rows0 + 1
        self.scan.fov_cols = fov_cols1 - fov_cols0 + 1

        spatial_src = np.empty((self.scan.fov_rows, self.scan.fov_cols), dtype='S256')
        for img_name in self.scan.fov_images:
            # a, b, c, col, row, d, e = img_name.split('-')[3].split('.')
            a, col, row, e = os.path.split(img_name)[1].split('_')
            spatial_src[int(row) - fov_rows0, self.scan.fov_cols - int(col) + fov_cols0 - 1] = img_name

        self.fov_location = np.zeros((self.scan.fov_rows, self.scan.fov_cols, 2), dtype=int)
        for i in range(self.scan.fov_rows):
            for j in range(self.scan.fov_cols):
                self.fov_location[i][j] = [int((1 - self.scan.overlap) * self.scan.fov_width * i),
                                           int((1 - self.scan.overlap) * self.scan.fov_height * j)]
        self.scan.mosaic_width, self.scan.mosaic_height = \
            [int((1 - self.scan.overlap) * self.scan.fov_width * (self.scan.fov_cols - 1) + self.scan.fov_width),
             int((1 - self.scan.overlap) * self.scan.fov_height * (self.scan.fov_rows - 1) + self.scan.fov_height)]

    def read(self, file_path):
        self._file_path = file_path
        clog.info('Try to read file {}'.format(file_path))
        check_flag = self._check()
        check_info = ['success', 'path not exists', 'not find the exact location of the images']
        if check_flag:
            clog.error('Validation on input path failed: {}'.format(check_info[check_flag]))
            return 1
        else: clog.info('Validation on input path success')
        self._init(printf=False)
        self._from_image_dir()
        self._from_cfg()
        self._print_info()


def main():
    import datetime
    clog.set_level(clog.INFO)
    t = datetime.datetime.now()
    file_name = 'cellbin{}-{}'.format(os.getpid(), t.strftime('%Y%m%d%H%M%S.log'))
    # clog.log2file(out_dir=r'./', filename=file_name)

    mmf = DaXingChengMicroscopeFile()
    mmf.read(file_path=r'D:\data\bigStroke\A01282B5D6')


if __name__ == '__main__':
    main()
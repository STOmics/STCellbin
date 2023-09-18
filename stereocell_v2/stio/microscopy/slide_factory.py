import os
from stio.microscopy.motic import MoticMicroscopeFile
from stio.microscopy.cg import ChangGuangMicroscopeFile
from stio.microscopy.dxc import DaXingChengMicroscopeFile
from stio.microscopy.zeiss import ZeissMicroscopeFile
from stio.microscopy.leica import LeicaMicroscopeFile
from stio.microscopy import MicroscopeLargeImage
from stio.microscopy import SlideType


class MicroscopeBaseFileFactory(object):
    def __init__(self):
        self.manufacturer = SlideType.Motic

    @staticmethod
    def create_microscope_file_by_name(manufacturer):
        st = SlideType(manufacturer)
        if st == SlideType.Motic:
            return MoticMicroscopeFile()
        elif st == SlideType.DaXingCheng:
            return DaXingChengMicroscopeFile()
        elif st == SlideType.ChangGuang:
            return ChangGuangMicroscopeFile()
        elif st == SlideType.Zeiss:
            return ZeissMicroscopeFile()
        elif st == SlideType.Leica:
            return LeicaMicroscopeFile()
        elif st == SlideType.Olympus:
            return None
        elif st == SlideType.Unknown:
            return MicroscopeLargeImage()
        else:
            return None

    def create_microscope_file(self, slide_file: str):
        self._slide_type(slide_file)
        mf = self.create_microscope_file_by_name(self.manufacturer)
        mf.read(slide_file)
        return mf

    def _slide_type(self, slide_file: str):
        assert os.path.exists(slide_file)
        if os.path.isdir(slide_file):
            files = os.listdir(slide_file)
            if {'1.ini', 'info.ini'}.issubset(files):
                self.manufacturer = SlideType.Motic
                return 0
            if {'info.json', 'fovs'}.issubset(files):
                self.manufacturer = SlideType.ChangGuang
                return 0
            if {'PrescanImage.tif', 'ScanInfoToCellBin.cfg'}.issubset(files):
                self.manufacturer = SlideType.DaXingCheng
        else:
            suffix = os.path.splitext(slide_file)[-1]
            if suffix == '.czi':
                self.manufacturer = SlideType.Zeiss
                return 0
            if suffix in ['.tif', '.png', '.jpg', '.tiff']:
                self.manufacturer = SlideType.Unknown
                return 0
        return 1


def main():
    cad = [r'D:\data\CHG_cellbin\CHG\SS200000369BL_B4\SS200000369BL_B4',
           r'D:\data\guojing\221107\SS200001153BR_D5',
           r'D:\data\bigStroke\A01282B5D6',
           r'D:\data\CHG_cellbin\CHG\SS200000369BL_B4\SS200000369BL_B4\stitch.tif']
    mbf2 = MicroscopeBaseFileFactory()
    for c in cad:
        mbf2.create_microscope_file(slide_file=c)


if __name__ == '__main__':
    main()

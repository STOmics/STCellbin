
import numpy as np
import os
import h5py
from stio.utils import clog
from stio.ipr import IPR
from stio.ipr.meta.mif_qc import QCInfo, ModuleInfo
from stio.ipr.meta.register import Register
from stio.ipr.meta.stitch import Stitch
from stio.ipr.meta.seg import TissueSeg, CellSeg
from stio.ipr.meta.mif_hardware import ImageInfo
from stio.ipr.meta.state import ManualState, StereoResepSwitch
from stio.ipr.meta.calibration import Calibration


class IPRv0d1d2(IPR):
    """ Standard format of image process record files in cell bin """

    group_names = ['CellSeg', 'ImageInfo', 'ManualState', 'QCInfo', 'Register', 'StereoResepSwitch',
                   'Stitch', 'TissueSeg', 'Calibration']

    def __init__(self) -> None:
        super(IPRv0d1d2, self).__init__()
        self.image_info = ImageInfo()
        self.qc_info = QCInfo()
        self.stitch = Stitch()
        self.tissue_seg = TissueSeg()
        self.cell_seg = CellSeg()
        self.register = Register()
        self.manual_state = ManualState()
        self.stereo_resep_switch = StereoResepSwitch()
        self.preview: np.array = np.nan
        self.calibration = Calibration()
        self.ipr_version: str = '0.1.2'

    @staticmethod
    def _is_io_legal(file_path: str):
        suffix = os.path.splitext(os.path.basename(file_path))[1]
        if suffix not in ['.ipr', '.json']:
            raise '{} is an unsupported file type'.format(suffix)
        if suffix == '.json':
            return 0
        else:
            return 1

    def write(self, file_path: str):
        """ Only two formats are supported: hdf5(.ipr) & json(.json) """
        flag = self._is_io_legal(file_path)
        clog.info('Parameters are serialized to {}'.format(file_path))
        if flag == 0:  # json
            pass
        else:  # hdf5
            h5 = h5py.File(file_path, 'w')
            for grp in self.group_names:
                group = h5.require_group(grp)
                if grp == 'ImageInfo':
                    self.image_info.to_h5(group)
                elif grp == 'QCInfo':
                    self.qc_info.to_h5(group)
                elif grp == 'Stitch':
                    self.stitch.to_h5(group)
                elif grp == 'TissueSeg':
                    self.tissue_seg.to_h5(group)
                elif grp == 'CellSeg':
                    self.cell_seg.to_h5(group)
                elif grp == 'Register':
                    self.register.to_h5(group)
                elif grp == 'StereoResepSwitch':
                    self.stereo_resep_switch.to_h5(group)
                elif grp == 'ManualState':
                    self.manual_state.to_h5(group)
                elif grp == 'Calibration':
                    self.calibration.to_h5(group)
                else:
                    pass
            ModuleInfo.create_dataset(h5, 'Preview', self.preview, compression=True)
            h5.attrs['IPRVersion'] = self.ipr_version
            h5.close()

    def init_by_file(self, file_path: str):
        """ Only two formats are supported: hdf5(.ipr) & json(.json) """
        flag = self._is_io_legal(file_path)
        clog.info('Initialize parameters from {}'.format(file_path))
        if flag == 0:  # json
            pass
        else:  # hdf5
            h5 = h5py.File(file_path, 'a')
            for grp in self.group_names:
                group = h5[grp]
                try:
                    if grp == 'ImageInfo':
                        self.image_info.from_h5(group)
                    elif grp == 'QCInfo':
                        self.qc_info.from_h5(group)
                    elif grp == 'Stitch':
                        self.stitch.from_h5(group)
                    elif grp == 'TissueSeg':
                        self.tissue_seg.from_h5(group)
                    elif grp == 'CellSeg':
                        self.cell_seg.from_h5(group)
                    elif grp == 'Register':
                        self.register.from_h5(group)
                    elif grp == 'StereoResepSwitch':
                        self.stereo_resep_switch.from_h5(group)
                    elif grp == 'ManualState':
                        self.manual_state.from_h5(group)
                    elif grp == 'Calibration':
                        self.calibration.from_h5(group)
                    else:
                        pass
                except KeyError:
                    print("{} ipr read error!".format(grp))
            self.preview = h5['Preview'][...]
            self.ipr_version = h5.attrs['IPRVersion']
            h5.close()


def main():
    file_path = r'D:\DOWN\3333.ipr'
    ipr = IPRv0d1d2()
    # write to hdf5
    ipr.write(file_path)
    # init from hdf5
    ipr.init_by_file(file_path)
    pass


if __name__ == '__main__':
    main()

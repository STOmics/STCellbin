
import numpy as np
import os
import h5py
from stio.utils import clog
from stio.ipr import IPR
from stio.ipr.meta.qc import QCInfo, ModuleInfo
from stio.ipr.meta.register import Register
from stio.ipr.meta.stitch import Stitch
from stio.ipr.meta.seg import TissueSeg, CellSeg
from stio.ipr.meta.hardware import ImageInfo
from stio.ipr.meta.state import ManualState, StereoResepSwitch
from stio.ipr.meta.research import Research


class IPRv0d0d1(IPR):
    """ Standard format of image process record files in cell bin """

    group_names = ['CellSeg', 'ImageInfo', 'ManualState', 'QCInfo', 'Register', 'StereoResepSwitch',
                   'Stitch', 'TissueSeg', 'Research']

    def __init__(self) -> None:
        super(IPRv0d0d1, self).__init__()
        self.image_info = ImageInfo()
        self.qc_info = QCInfo()
        self.stitch = Stitch()
        self.tissue_seg = TissueSeg()
        self.cell_seg = CellSeg()
        self.register = Register()
        self.manual_state = ManualState()
        self.stereo_resep_switch = StereoResepSwitch()
        self.research = Research()
        self.preview: np.array = np.nan
        self.ipr_version: str = '0.0.1'

    @staticmethod
    def _is_io_legal(file_path: str):
        suffix = os.path.splitext(os.path.basename(file_path))[1]
        if suffix not in ['.ipr', '.json']: raise '{} is an unsupported file type'.format(suffix)
        if suffix == '.json': return 0
        else: return 1

    def write(self, file_path: str, mode="w"):
        """ Only two formats are supported: hdf5(.ipr) & json(.json) """
        flag = self._is_io_legal(file_path)
        clog.info('Parameters are serialized to {}'.format(file_path))
        if flag == 0:  # json
            pass
        else:  # hdf5
            h5 = h5py.File(file_path, mode)
            for grp in self.group_names:
                group = h5.require_group(grp)
                if grp == 'ImageInfo': self.image_info.to_h5(group)
                elif grp == 'QCInfo': self.qc_info.to_h5(group)
                elif grp == 'Stitch': self.stitch.to_h5(group)
                elif grp == 'TissueSeg': self.tissue_seg.to_h5(group)
                elif grp == 'CellSeg': self.cell_seg.to_h5(group)
                elif grp == 'Register': self.register.to_h5(group)
                elif grp == 'StereoResepSwitch': self.stereo_resep_switch.to_h5(group)
                elif grp == 'ManualState': self.manual_state.to_h5(group)
                elif grp == 'Research': self.research.to_h5(group)
                else: pass
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
                    if grp == 'ImageInfo': self.image_info.from_h5(group)
                    elif grp == 'QCInfo': self.qc_info.from_h5(group)
                    elif grp == 'Stitch': self.stitch.from_h5(group)
                    elif grp == 'TissueSeg': self.tissue_seg.from_h5(group)
                    elif grp == 'CellSeg': self.cell_seg.from_h5(group)
                    elif grp == 'Register': self.register.from_h5(group)
                    elif grp == 'StereoResepSwitch': self.stereo_resep_switch.from_h5(group)
                    elif grp == 'ManualState': self.manual_state.from_h5(group)
                    elif grp == 'Research': self.research.from_h5(group)
                    else: pass
                except KeyError:
                    print("{} ipr read error!".format(grp))
            self.preview = h5['Preview'][...]
            self.ipr_version = h5.attrs['IPRVersion']
            h5.close()

    def get_fov_size(self, ): return (self.image_info.fov_width, self.image_info.fov_height)

    def set_fov_size(self, s):
        assert len(s) == 2
        self.image_info.fov_width, self.image_info.fov_height = s

    def get_magnification(self, ): return int(self.image_info.scan_objective)

    def set_magnification(self, m: int): self.image_info.scan_objective = m

    def is_stitched_image(self, ): return self.image_info.stitched_image


def main():
    file_path = r'D:\apps\cellbinstudio\CellBinFiles\C01528B2_20221207_114514_test.ipr'
    ipr = IPRv0d0d1()
    # write to hdf5
    ipr.write(file_path)
    # init from hdf5
    ipr.init_by_file(file_path)
    pass


if __name__ == '__main__':
    main()




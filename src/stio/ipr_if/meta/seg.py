from stio.ipr.meta import ModuleInfo
from stio.utils import clog
import numpy as np
import h5py


class TissueSeg(ModuleInfo):
    def __init__(self):
        super(TissueSeg, self).__init__()
        self.tissue_mask: np.array = np.nan
        self.tissue_seg_score: np.uint8 = 0
        self.tissue_seg_shape: np.array = np.array([0, 0], dtype=int)

    def to_h5(self, grp):
        self.create_dataset(grp, 'TissueMask', self.tissue_mask, compression=True)
        grp.attrs['TissueSegScore'] = self.tissue_seg_score
        grp.attrs['TissueSegShape'] = self.tissue_seg_shape

    def from_h5(self, grp, ignore=True):
        if not ignore: self.tissue_mask = grp['TissueMask']
        else: clog.warning('Tissue mask is too large and has not been loaded into memory.')
        self.tissue_seg_score = grp.attrs['TissueSegScore']
        self.tissue_seg_shape = grp.attrs['TissueSegShape']

    @staticmethod
    def get_mask_from_h5(file_path: str):
        h5 = h5py.File(file_path, 'r')
        return h5['TissueSeg']['TissueMask']

    def to_dict(self, ):
        dct = dict()
        dct['TissueMask'] = self.tissue_mask
        dct['TissueSegScore'] = self.tissue_seg_score
        dct['TissueSegShape'] = self.tissue_seg_shape
        return dct

    def from_dict(self, dct: dict):
        self.tissue_mask = dct['TissueMask']
        self.tissue_seg_score = dct['TissueSegScore']
        self.tissue_seg_shape = dct['TissueSegShape']


class CellSeg(ModuleInfo):
    def __init__(self) -> None:
        super(CellSeg, self).__init__()
        self.cell_mask: np.array = np.nan
        self.cell_seg_shape: np.array = np.array([0, 0], dtype=int)

    def to_h5(self, grp):
        grp.attrs['CellSegShape'] = self.cell_seg_shape
        self.create_dataset(grp, 'CellMask', self.cell_mask, compression=True)

    def from_h5(self, grp, ignore=True):
        if not ignore: self.cell_mask = grp['CellMask']
        else: clog.warning('Cell mask is too large and has not been loaded into memory.')
        self.cell_seg_shape = grp.attrs['CellSegShape']

    @staticmethod
    def get_mask_from_h5(file_path: str):
        h5 = h5py.File(file_path, 'r')
        return h5['CellSeg']['CellMask']

    def to_dict(self, ):
        dct = dict()
        dct['CellSegShape'] = self.cell_seg_shape
        dct['CellMask'] = self.cell_mask
        return dct

    def from_dict(self, dct: dict):
        self.cell_mask = dct['CellMask']
        self.cell_seg_shape = dct['CellSegShape']

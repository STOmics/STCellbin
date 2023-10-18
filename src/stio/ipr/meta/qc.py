import numpy as np
from stio.ipr.meta import ModuleInfo


class QCInfo(ModuleInfo):
    def __init__(self):
        super(QCInfo, self).__init__()
        self.clarity_score: np.uint8 = 0
        self.experimenter: str = ''
        self.good_fov_count: int = 0
        self.imageQC_version: str = ''
        self.qc_pass_flag: int = 0
        self.remark_info: str = ''
        self.stain_type: str = ''  # optional
        self.total_fov_count: int = 0
        self.track_line_score: np.uint8 = 0
        self.cross_points: dict = dict()

    def to_h5(self, grp):
        grp.attrs['ClarityScore'] = self.clarity_score
        grp.attrs['Experimenter'] = self.experimenter
        grp.attrs['GoodFOVCount'] = self.good_fov_count
        grp.attrs['ImageQCVersion'] = self.imageQC_version
        grp.attrs['QCPassFlag'] = self.qc_pass_flag
        grp.attrs['RemarkInfo'] = self.remark_info
        grp.attrs['StainType'] = self.stain_type
        grp.attrs['TotalFOVCount'] = self.total_fov_count
        grp.attrs['TrackLineScore'] = self.track_line_score

        grp.require_group('CrossPoints')
        for k, v in self.cross_points.items():
            self.create_dataset(grp, 'CrossPoints/{}'.format(k), v, compression=False)

    def from_h5(self, grp):
        self.clarity_score = grp.attrs['ClarityScore']
        self.experimenter = grp.attrs['Experimenter']
        self.good_fov_count = grp.attrs['GoodFOVCount']
        self.imageQC_version = grp.attrs['ImageQCVersion']
        self.qc_pass_flag = grp.attrs['QCPassFlag']
        self.remark_info = grp.attrs['RemarkInfo']
        self.stain_type = grp.attrs['StainType']
        self.total_fov_count = grp.attrs['TotalFOVCount']
        self.track_line_score = grp.attrs['TrackLineScore']
        for key in grp['CrossPoints'].keys(): self.cross_points[key] = grp['CrossPoints'][key][:]

    def from_dict(self, dct: dict):
        self.clarity_score = dct['ClarityScore']
        self.experimenter = dct['Experimenter']
        self.good_fov_count = dct['GoodFOVCount']
        self.imageQC_version = dct['ImageQCVersion']
        self.qc_pass_flag = dct['QCPassFlag']
        self.remark_info = dct['RemarkInfo']
        self.stain_type = dct['StainType']
        self.total_fov_count = dct['TotalFOVCount']
        self.track_line_score = dct['TrackLineScore']
        self.cross_points = dct['CrossPoints']

    def to_dict(self, ):
        dct = dict()
        dct['ClarityScore'] = self.clarity_score
        dct['Experimenter'] = self.experimenter
        dct['GoodFOVCount'] = self.good_fov_count
        dct['ImageQCVersion'] = self.imageQC_version
        dct['QCPassFlag'] = self.qc_pass_flag
        dct['RemarkInfo'] = self.remark_info
        dct['StainType'] = self.stain_type
        dct['TotalFOVCount'] = self.total_fov_count
        dct['TrackLineScore'] = self.track_line_score
        return dct

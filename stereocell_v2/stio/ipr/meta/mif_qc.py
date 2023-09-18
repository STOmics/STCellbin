import numpy as np
from stio.ipr.meta import ModuleInfo
from stio.ipr.meta.calibration import StitchEval


class QCInfo(ModuleInfo):
    def __init__(self):
        super(QCInfo, self).__init__()
        self.clarity_score: np.uint8 = 0
        self.cross_points: dict = dict()
        self.clarity_scores: np.array = np.array([0, 0], dtype=np.uint8)
        self.experimenter: str = ''
        self.good_fov_count: int = 0
        self.imageQC_version: str = ''
        self.qc_pass_flag: int = 0
        self.remark_info: str = ''
        self.stain_type: str = ''  # optional
        self.total_fov_count: int = 0
        self.track_line_channel: int = 0
        self.track_line_score: np.uint8 = 0
        self.scopestitchqcscore: str = ""
        self.scopestitchqcmatrix: str = ""
        self.track_cross_qc_passflag: bool = ""
        self.scope_stitch_qc_passflag: bool = ""
        self.templatefov: np.array = np.array([0, 0], dtype=np.uint8)
        self.stitcheval = StitchEval()

    def to_h5(self, grp):
        grp.attrs['ClarityScore'] = self.clarity_score
        grp.require_group('CrossPoints')
        for k, v in self.cross_points.items():
            self.create_dataset(grp, 'CrossPoints/{}'.format(k), v, compression=False)
        grp.attrs['ClarityScores'] = self.clarity_scores
        grp.attrs['Experimenter'] = self.experimenter
        grp.attrs['GoodFOVCount'] = self.good_fov_count
        grp.attrs['ImageQCVersion'] = self.imageQC_version
        grp.attrs['QCPassFlag'] = self.qc_pass_flag
        grp.attrs['RemarkInfo'] = self.remark_info
        grp.attrs['StainType'] = self.stain_type
        grp.attrs['TotalFOVCount'] = self.total_fov_count
        grp.attrs['TrackLineChannel'] = self.track_line_channel
        grp.attrs['TrackLineScore'] = self.track_line_score
        grp.attrs['ScopeStitchQCScore'] = self.scopestitchqcscore
        grp.attrs['ScopeStitchQCMatrix'] = self.scopestitchqcmatrix
        grp.attrs['TrackCrossQCPassFlag'] = self.track_cross_qc_passflag
        grp.attrs['ScopeStitchQCPassFlag'] = self.scope_stitch_qc_passflag
        grp.attrs['TemplateFOV'] = self.templatefov
        g0 = grp.require_group('StitchEval')
        g0.attrs['BGIStitchDeviation'] = self.stitcheval.bgistitchdeviation
        g0.attrs['BGIStitchGlobalLoc'] = self.stitcheval.bgistitchgloballoc
        g0.attrs['FOVAlignConfidenc'] = self.stitcheval.fovalignconfidence
        g0.attrs['FOVTissueType'] = self.stitcheval.fovtissuetype
        g0.attrs['HorizontalJitte'] = self.stitcheval.horizontaljitter
        g0.attrs['VerticalJitte'] = self.stitcheval.verticaljitter

    def from_h5(self, grp):
        self.clarity_score = grp.attrs['ClarityScore']
        self.clarity_scores = grp.attrs['ClarityScores']
        self.experimenter = grp.attrs['Experimenter']
        self.good_fov_count = grp.attrs['GoodFOVCount']
        self.imageQC_version = grp.attrs['ImageQCVersion']
        self.qc_pass_flag = grp.attrs['QCPassFlag']
        self.remark_info = grp.attrs['RemarkInfo']
        self.stain_type = grp.attrs['StainType']
        self.total_fov_count = grp.attrs['TotalFOVCount']
        self.track_line_channel =  grp.attrs['TrackLineChannel']
        self.track_line_score = grp.attrs['TrackLineScore']
        self.scopestitchqcscore = grp.attrs['ScopeStitchQCScore']
        self.scopestitchqcmatrix = grp.attrs['ScopeStitchQCMatrix']
        self.track_cross_qc_passflag = grp.attrs['TrackCrossQCPassFlag']
        self.scope_stitch_qc_passflag = grp.attrs['ScopeStitchQCPassFlag']
        self.templatefov = grp.attrs['TemplateFOV']
        g0 = grp.require_group('StitchEval')
        self.stitcheval.bgistitchdeviation = g0.attrs['BGIStitchDeviation']
        self.stitcheval.bgistitchgloballoc = g0.attrs['BGIStitchGlobalLoc']
        self.stitcheval.fovalignconfidence = g0.attrs['FOVAlignConfidenc']
        self.stitcheval.fovtissuetype = g0.attrs['FOVTissueType']
        self.stitcheval.horizontaljitter = g0.attrs['HorizontalJitte']
        self.stitcheval.verticaljitter = g0.attrs['VerticalJitte']
        for key in grp['CrossPoints'].keys():
            self.cross_points[key] = grp['CrossPoints'][key][:]

    def from_dict(self, dct: dict):
        self.clarity_score = dct['ClarityScore']
        self.clarity_scores = dct['ClarityScores']
        self.experimenter = dct['Experimenter']
        self.good_fov_count = dct['GoodFOVCount']
        self.imageQC_version = dct['ImageQCVersion']
        self.qc_pass_flag = dct['QCPassFlag']
        self.remark_info = dct['RemarkInfo']
        self.stain_type = dct['StainType']
        self.total_fov_count = dct['TotalFOVCount']
        self.track_line_channel = dct['TrackLineChannel']
        self.track_line_score = dct['TrackLineScore']
        self.scopestitchqcscore = dct['ScopeStitchQCScore']
        self.scopestitchqcmatrix = dct['ScopeStitchQCMatrix']
        self.track_cross_qc_passflag = dct['TrackCrossQCPassFlag']
        self.scope_stitch_qc_passflag = dct['ScopeStitchQCPassFlag']
        self.templatefov = dct['TemplateFOV']
        self.cross_points = dct['CrossPoints']
        self.stitcheval.bgistitchdeviation = dct["StitchEval"]['BGIStitchDeviation']
        self.stitcheval.bgistitchgloballoc = dct["StitchEval"]['BGIStitchGlobalLoc']
        self.stitcheval.fovalignconfidence = dct["StitchEval"]['FOVAlignConfidenc']
        self.stitcheval.fovtissuetype = dct["StitchEval"]['FOVTissueType']
        self.stitcheval.horizontaljitter = dct["StitchEval"]['HorizontalJitte']
        self.stitcheval.verticaljitter = dct["StitchEval"]['VerticalJitte']

    def to_dict(self, ):
        dct = dict()
        dct['ClarityScore'] = self.clarity_score
        dct['ClarityScores'] = self.clarity_scores
        dct['Experimenter'] = self.experimenter
        dct['GoodFOVCount'] = self.good_fov_count
        dct['ImageQCVersion'] = self.imageQC_version
        dct['QCPassFlag'] = self.qc_pass_flag
        dct['RemarkInfo'] = self.remark_info
        dct['StainType'] = self.stain_type
        dct['TotalFOVCount'] = self.total_fov_count
        dct['TrackLineChannel'] = self.track_line_channel
        dct['TrackLineScore'] = self.track_line_score
        dct['ScopeStitchQCScore'] = self.scopestitchqcscore
        dct['ScopeStitchQCMatrix'] = self.scopestitchqcmatrix
        dct['TrackCrossQCPassFlag'] = self.track_cross_qc_passflag
        dct['ScopeStitchQCPassFlag'] = self.scope_stitch_qc_passflag
        dct['TemplateFOV'] = self.templatefov
        dct["StitchEval"]['BGIStitchDeviation'] = self.stitcheval.bgistitchdeviation
        dct["StitchEval"]['BGIStitchGlobalLoc'] = self.stitcheval.bgistitchgloballoc
        dct["StitchEval"]['FOVAlignConfidenc'] = self.stitcheval.fovalignconfidence
        dct["StitchEval"]['FOVTissueType'] = self.stitcheval.fovtissuetype
        dct["StitchEval"]['HorizontalJitte'] = self.stitcheval.horizontaljitter
        dct["StitchEval"]['VerticalJitte'] = self.stitcheval.verticaljitter
        return dct

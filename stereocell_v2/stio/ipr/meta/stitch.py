from stio.ipr.meta import ModuleInfo
import numpy as np


class BGIStitch(object):
    def __init__(self):
        self.stitched_global_height: int = 0
        self.stitched_global_width: int = 0
        self.stitched_global_loc: np.array = np.nan


class ScopeStitch(object):
    def __init__(self):
        self.global_height: int = 0
        self.global_width: int = 0
        self.global_loc: np.array = np.nan


class StitchEval(object):
    def __init__(self):
        self.stitch_eval_h: np.array = np.nan
        self.stitch_eval_v: np.array = np.nan
        self.max_deviation: float = 0.0
        self.global_deviation: float = 0.0


class Stitch(ModuleInfo):
    def __init__(self):
        super(Stitch, self).__init__()
        self.stitching_score: np.uint8 = 0
        self.template_source: np.array = np.array([0, 0], dtype=np.uint8)
        self.bgi_stitch = BGIStitch()
        self.scope_stitch = ScopeStitch()
        self.stitch_eval = StitchEval()

    def to_h5(self, grp):
        grp.attrs['StitchingScore'] = self.stitching_score
        grp.attrs['TemplateSource'] = self.template_source

        g0 = grp.require_group('BGIStitch')
        g0.attrs['StitchedGlobalHeight'] = self.bgi_stitch.stitched_global_height
        g0.attrs['StitchedGlobalWidth'] = self.bgi_stitch.stitched_global_width
        self.create_dataset(g0, 'StitchedGlobalLoc', self.bgi_stitch.stitched_global_loc, compression=False)

        g1 = grp.require_group('ScopeStitch')
        g1.attrs['GlobalHeight'] = self.scope_stitch.global_height
        g1.attrs['GlobalWidth'] = self.scope_stitch.global_width
        self.create_dataset(g1, 'GlobalLoc', self.scope_stitch.global_loc, compression=False)

        g2 = grp.require_group('StitchEval')
        self.create_dataset(g2, 'StitchEvalH', self.stitch_eval.stitch_eval_h, compression=False)
        self.create_dataset(g2, 'StitchEvalV', self.stitch_eval.stitch_eval_v, compression=False)
        g2.attrs['MaxDeviation'] = self.stitch_eval.max_deviation

    def from_h5(self, grp):
        self.stitching_score = grp.attrs['StitchingScore']
        self.template_source = grp.attrs['TemplateSource']

        g0 = grp.get('BGIStitch')
        self.bgi_stitch.stitchGlobalHeight = g0.attrs['StitchedGlobalHeight']
        self.bgi_stitch.stitchGlobalWidth = g0.attrs['StitchedGlobalWidth']
        self.bgi_stitch.stitchGlobalLoc = g0['StitchedGlobalLoc'][...]

        g1 = grp.get('ScopeStitch')
        self.scope_stitch.globalHeight = g1.attrs['GlobalHeight']
        self.scope_stitch.globalWidth = g1.attrs['GlobalWidth']
        self.scope_stitch.globalLoc = g1['GlobalLoc'][...]

        g2 = grp.get('StitchEval')
        self.stitch_eval.stitchEvalH = g2['StitchEvalH'][...]
        self.stitch_eval.stitchEvalV = g2['StitchEvalV'][...]
        self.stitch_eval.maxDeviation = g2.attrs['MaxDeviation']

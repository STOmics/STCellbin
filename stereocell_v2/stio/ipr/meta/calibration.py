from stio.ipr.meta import ModuleInfo
import numpy as np


class Calibration(ModuleInfo):
    def __init__(self):
        super(Calibration, self).__init__()
        self.bgi = BGI()
        self.scope = Scope()
        self.CalibrationQCPassFlag: bool = False

    def to_h5(self, grp):
        g0 = grp.require_group('BGI')
        g0.attrs['Confidence'] = self.bgi.Confidence
        g0.attrs['OffsetX'] = self.bgi.OffsetX
        g0.attrs['OffsetY'] = self.bgi.OffsetY
        g0.attrs['Scale'] = self.bgi.Scale
        g0.attrs['Rotation'] = self.bgi.Rotation
        g1 = grp.require_group('Scope')
        g1.attrs['Confidence'] = self.scope.Confidence
        g1.attrs['OffsetX'] = self.scope.OffsetX
        g1.attrs['OffsetY'] = self.scope.OffsetY
        g1.attrs['Scale'] = self.scope.Scale
        g1.attrs['Rotation'] = self.scope.Rotation
        grp.attrs['CalibrationQCPassFlag'] = self.CalibrationQCPassFlag

    def from_h5(self, grp):
        g0 = grp.require_group('BGI')
        self.bgi.Confidence = g0.attrs['Confidence']
        self.bgi.OffsetX = g0.attrs['OffsetX']
        self.bgi.OffsetY = g0.attrs['OffsetY']
        self.bgi.Scale = g0.attrs['Scale']
        self.bgi.Rotation = g0.attrs['Rotation']
        g1 = grp.require_group('Scope')
        self.scope.Confidence = g1.attrs['Confidence']
        self.scope.OffsetX = g1.attrs['OffsetX']
        self.scope.OffsetY = g1.attrs['OffsetY']
        self.scope.Scale = g1.attrs['Scale']
        self.scope.Rotation = g1.attrs['Rotation']
        self.CalibrationQCPassFlag = grp.attrs['CalibrationQCPassFlag']


class BGI(object):
    def __init__(self):
        super(BGI, self).__init__()
        self.Confidence: float = 0.0
        self.OffsetX: float = 0.0
        self.OffsetY: float = 0.0
        self.Scale: float = 0.0
        self.Rotation: float = 0.0


class Scope(BGI):
    def __init__(self):
        super(Scope, self).__init__()


class StitchEval(object):
    def __init__(self):
        super(StitchEval, self).__init__()
        self.bgistitchdeviation: np.array = np.array([0, 0], dtype=np.uint8)
        self.bgistitchgloballoc: np.array = np.array([0, 0], dtype=np.uint8)
        self.fovalignconfidence: np.array = np.array([0, 0], dtype=np.uint8)
        self.fovtissuetype: np.array = np.array([0, 0], dtype=np.uint8)
        self.horizontaljitter: np.array = np.array([0, 0], dtype=np.uint8)
        self.verticaljitter: np.array = np.array([0, 0], dtype=np.uint8)

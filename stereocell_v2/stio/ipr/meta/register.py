
from stio.ipr.meta import ModuleInfo
import numpy as np


class Register(ModuleInfo):
    def __init__(self):
        super(Register, self).__init__()
        self.counter_rot90: int = 0
        self.flip: bool = True
        self.matrix_shape: np.array = np.array([0, 0])
        self.offset_x: float = 0.0
        self.offset_y: float = 0.0
        self.register_score: int = 0
        self.rotation: float = 0.0
        self.scale_x: float = 0.0
        self.scale_y: float = 0.0
        self.x_start: int = 0
        self.y_start: int = 0
        self.matrix_template: np.array = np.nan

    def to_h5(self, grp):
        grp.attrs['CounterRot90'] = self.counter_rot90
        grp.attrs['Flip'] = self.flip
        grp.attrs['MatrixShape'] = self.matrix_shape
        grp.attrs['OffsetX'] = self.offset_x
        grp.attrs['OffsetY'] = self.offset_y
        grp.attrs['RegisterScore'] = self.register_score
        grp.attrs['ScaleX'] = self.scale_x
        grp.attrs['ScaleY'] = self.scale_y
        grp.attrs['Rotation'] = self.rotation
        grp.attrs['XStart'] = self.x_start
        grp.attrs['YStart'] = self.y_start
        self.create_dataset(grp, 'MatrixTemplate', self.matrix_template, compression=False)

    def from_h5(self, grp):
        self.y_start = grp.attrs['YStart']
        self.x_start = grp.attrs['XStart']
        self.flip = grp.attrs['Flip']
        self.offset_x = grp.attrs['OffsetX']
        self.offset_y = grp.attrs['OffsetY']
        self.rotation = grp.attrs['Rotation']
        self.register_score = grp.attrs['RegisterScore']
        self.matrix_shape = grp.attrs['MatrixShape']
        self.counter_rot90 = grp.attrs['CounterRot90']
        self.scale_x = grp.attrs['ScaleX']
        self.scale_y = grp.attrs['ScaleY']
        self.matrix_template = grp['MatrixTemplate'][...]  # dataset

    def from_dict(self, dct: dict):
        self.y_start = dct['YStart']
        self.x_start = dct['XStart']
        self.flip = dct['Flip']
        self.offset_x = dct['OffsetX']
        self.offset_y = dct['OffsetY']
        self.rotation = dct['Rotation']
        self.register_score = dct['RegisterScore']
        self.matrix_shape = dct['MatrixShape']
        self.counter_rot90 = dct['CounterRot90']
        self.scale_x = dct['ScaleX']
        self.scale_y = dct['ScaleY']
        self.matrix_template = dct['MatrixTemplate']

    def to_dict(self, ):
        dct = dict()
        dct['CounterRot90'] = self.counter_rot90
        dct['Flip'] = self.flip
        dct['MatrixShape'] = self.matrix_shape
        dct['OffsetX'] = self.offset_x
        dct['OffsetY'] = self.offset_y
        dct['RegisterScore'] = self.register_score
        dct['ScaleX'] = self.scale_x
        dct['ScaleY'] = self.scale_y
        dct['Rotation'] = self.rotation
        dct['XStart'] = self.x_start
        dct['YStart'] = self.y_start
        dct['MatrixTemplate'] = self.matrix_template
        return dct

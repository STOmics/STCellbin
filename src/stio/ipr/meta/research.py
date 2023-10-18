import numpy as np
from stio.ipr.meta import ModuleInfo


class Research(ModuleInfo):
    def __init__(self):
        super(Research, self).__init__()
        self.fovs_tag = None

    def set_fovs_tag(self, t):
        self.fovs_tag = t

    def to_h5(self, grp):
        self.create_dataset(grp, 'FOVS_Tag', self.fovs_tag, compression=False)

    def from_h5(self, grp):
        self.fovs_tag = grp['FOVS_Tag']

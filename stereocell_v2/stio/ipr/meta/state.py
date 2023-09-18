from stio.ipr.meta import ModuleInfo


class ManualState(ModuleInfo):
    def __init__(self):
        super(ManualState, self).__init__()
        self.stitch: bool = False
        self.tissue_seg: bool = False
        self.cell_seg: bool = False
        self.register: bool = False

    def to_h5(self, grp):
        grp.attrs['cellseg'] = self.cell_seg
        grp.attrs['register'] = self.register
        grp.attrs['stitch'] = self.stitch
        grp.attrs['tissueseg'] = self.tissue_seg

    def from_h5(self, grp):
        self.cell_seg = grp.attrs['cellseg']
        self.register = grp.attrs['register']
        self.stitch = grp.attrs['stitch']
        self.tissue_seg = grp.attrs['tissueseg']

    def to_dict(self, ):
        dct = dict()
        dct['cellseg'] = self.cell_seg
        dct['register'] = self.register
        dct['stitch'] = self.stitch
        dct['tissueseg'] = self.tissue_seg
        return dct

    def from_dict(self, dct):
        self.cell_seg = dct['cellseg']
        self.register = dct['register']
        self.stitch = dct['stitch']
        self.tissue_seg = dct['tissueseg']


class StereoResepSwitch(ModuleInfo):
    def __init__(self):
        super(StereoResepSwitch, self).__init__()
        self.stitch: bool = False
        self.tissue_seg: bool = False
        self.cell_seg: bool = False
        self.register: bool = False

    def to_h5(self, grp):
        grp.attrs['cellseg'] = self.cell_seg
        grp.attrs['register'] = self.register
        grp.attrs['stitch'] = self.stitch
        grp.attrs['tissueseg'] = self.tissue_seg

    def from_h5(self, grp):
        self.cell_seg = grp.attrs['cellseg']
        self.register = grp.attrs['register']
        self.stitch = grp.attrs['stitch']
        self.tissue_seg = grp.attrs['tissueseg']

    def to_dict(self, ):
        dct = dict()
        dct['cellseg'] = self.cell_seg
        dct['register'] = self.register
        dct['stitch'] = self.stitch
        dct['tissueseg'] = self.tissue_seg
        return dct

    def from_dict(self, dct: dict):
        self.cell_seg = dct['cellseg']
        self.register = dct['register']
        self.stitch = dct['stitch']
        self.tissue_seg = dct['tissueseg']

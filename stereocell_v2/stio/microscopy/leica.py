from stio.microscopy import MicroscopeSmallImage, SlideType


class LeicaMicroscopeFile(MicroscopeSmallImage):
    def __init__(self):
        super(LeicaMicroscopeFile, self).__init__()
        self.hi.manufacturer = SlideType.Leica.value
        self.images_path: str = ''
        self.sitcb_cfg: str = ''

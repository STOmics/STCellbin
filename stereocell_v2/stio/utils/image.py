import tifffile
import numpy as np


class Image(object):
    def __init__(self):
        self.suffix: str = ''
        self.image = None
        self.channel: int = 1
        self.dtype = None
        self.width: int = 0
        self.height: int = 0

    def read(self, image):
        import os

        self.suffix = os.path.splitext(image)[1]
        if type(image) is str:
            if self.suffix == '.tif': self.image = tifffile.imread(image)
            elif self.suffix in ['.jpg', 'png']: self.image = None
            else: return 1
        elif type(image) is np.array: self.image = image
        else: return 1

        if self.image.ndim == 3: self.height, self.width, self.channel = self.image.shape
        else:
            self.height, self.width = self.image.shape
            self.channel = 1
        self.dtype = self.image.dtype
        return 0

    def write(self, output_path: str, compression=False):
        if compression: tifffile.imwrite(output_path, self.image, compression="zlib", compressionargs={"level": 8})
        else: tifffile.imwrite(output_path, self.image)

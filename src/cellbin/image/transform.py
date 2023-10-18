"""
图像变换类
"""
import pyvips
import numpy as np
import math

format_to_dtype = {
    'uchar': np.uint8,
    'char': np.int8,
    'ushort': np.uint16,
    'short': np.int16,
    'uint': np.uint32,
    'int': np.int32,
    'float': np.float32,
    'double': np.float64,
    'complex': np.complex64,
    'dpcomplex': np.complex128,
}

# map np dtypes to vips
dtype_to_format = {
    'uint8': 'uchar',
    'int8': 'char',
    'uint16': 'ushort',
    'int16': 'short',
    'uint32': 'uint',
    'int32': 'int',
    'float32': 'float',
    'float64': 'double',
    'complex64': 'complex',
    'complex128': 'dpcomplex',
}


def crop_around_center(image, width, height):
    """
    Given a NumPy / OpenCV 2 image, crops it to the given width and height,
    around it's centre point
    """

    image_size = (image.shape[1], image.shape[0])
    image_center = (int(image_size[0] * 0.5), int(image_size[1] * 0.5))

    if width > image_size[0]:
        width = image_size[0]

    if height > image_size[1]:
        height = image_size[1]

    x1 = int(image_center[0] - width * 0.5)
    x2 = int(image_center[0] + width * 0.5)
    y1 = int(image_center[1] - height * 0.5)
    y2 = int(image_center[1] + height * 0.5)

    return image[y1:y2, x1:x2]


def rotatedRectWithMaxArea(w, h, angle):
    """
    Given a rectangle of size wxh that has been rotated by 'angle' (in
    radians), computes the width and height of the largest possible
    axis-aligned rectangle (maximal area) within the rotated rectangle.
    """
    if w <= 0 or h <= 0:
        return 0, 0

    width_is_longer = w >= h
    side_long, side_short = (w, h) if width_is_longer else (h, w)

    # since the solutions for angle, -angle and 180-angle are all the same,
    # if suffices to look at the first quadrant and the absolute values of sin,cos:
    sin_a, cos_a = abs(math.sin(angle)), abs(math.cos(angle))
    if side_short <= 2. * sin_a * cos_a * side_long or abs(sin_a - cos_a) < 1e-10:
        # half constrained case: two crop corners touch the longer side,
        #   the other two corners are on the mid-line parallel to the longer line
        x = 0.5 * side_short
        wr, hr = (x / sin_a, x / cos_a) if width_is_longer else (x / cos_a, x / sin_a)
    else:
        # fully constrained case: crop touches all 4 sides
        cos_2a = cos_a * cos_a - sin_a * sin_a
        wr, hr = (w * cos_a - h * sin_a) / cos_2a, (h * cos_a - w * sin_a) / cos_2a

    return wr, hr


class ImageTransform(pyvips.Image):
    def __init__(self):
        self.image = None

    def set_image(self, image_path: str):
        """
        :param image_path: str | array
        """
        if isinstance(image_path, str):
            self.image = self.new_from_file(image_path)
        elif isinstance(image_path, np.ndarray):
            self.image = self.new_from_array(image_path)
        else:
            print("Image path type error.")

    def offset(self, x_offset: int = 0, y_offset: int = 0, dst_size: tuple = None):
        offset = [x_offset, y_offset]
        self.__rigid_transform(offset=offset, dst_shape=dst_size)
        arr = self.to_image()
        return arr

    def scale(self, x_scale: float, y_scale: float):
        self.__affine_transform(scale_x=x_scale, scale_y=y_scale)
        arr = self.to_image()
        return arr

    def resize(self, dst_size: tuple):
        x_scale = dst_size[1] / self.image.width
        y_scale = dst_size[0] / self.image.height
        arr = self.scale(x_scale=x_scale, y_scale=y_scale)
        return arr

    def rot90(self, rot90_type: int):
        self.__rigid_transform(rot_type=rot90_type)
        arr = self.to_image()
        return arr

    def rot(self, angle):
        self.__affine_transform(rotation=angle)
        arr = self.to_image()
        return arr

    def rot_scale(self, x_scale: float, y_scale: float, angle: float, ):
        self.__affine_transform(scale_x=x_scale, scale_y=y_scale, rotation=angle)
        arr = self.to_image()
        return arr

    def flip(self, flip_type: str):
        '''
        :param flip_type: 'ver' | 'hor'
        '''
        self.__rigid_transform(flip=flip_type)
        arr = self.to_image()
        return arr

    def rot_and_crop(self, angle):
        """
        Given the angle, return the maximum rectangle within the rotated rectangle.

        Args:
            angle (): angle in degree

        Returns:
            arr_cropped: cropped rotated image

        """
        image_height, image_width = self.image.height, self.image.width
        rot_arr = self.rot(angle)
        arr_cropped = crop_around_center(
            rot_arr,
            *rotatedRectWithMaxArea(
                w=image_width,
                h=image_height,
                angle=math.radians(angle)
            )
        )
        return arr_cropped

    def __rigid_transform(self, flip=None, rot_type=None, offset=None, dst_shape=None):
        """
        2023/4/6 @dengzhonghan 将扩展画布优先于offset挪动
        2023/5/6 @dengzhonghan 画布扩展与offset挪动合并，不然需要判断两幅图尺寸大小的关系
        刚性
        """
        if flip is not None:
            if flip == 'ver':
                self.image = self.image.flipver()
            elif flip == 'hor':
                self.image = self.image.fliphor()

        if rot_type is not None:
            theta = np.radians(-rot_type * 90)
            m = [np.cos(theta), -np.sin(theta), np.sin(theta), np.cos(theta)]
            self.image = self.image.affine(m, interpolate=pyvips.Interpolate.new("nearest"),
                                           background=[0])
        # if dst_shape is not None:
        #     h, w = dst_shape
        #     self.image = self.image.affine([1, 0, 0, 1],
        #                                    interpolate=pyvips.Interpolate.new("nearest"),
        #                                    oarea=[0, 0, w, h])
        if offset is not None and dst_shape is not None:
            x, y = offset
            h, w = dst_shape
            self.image = self.image.affine([1, 0, 0, 1],
                                           interpolate=pyvips.Interpolate.new("nearest"),
                                           idx=x, idy=y, oarea=[0, 0, w, h])

    def __affine_transform(self, scale_x=None, scale_y=None, rotation=None):
        """
        仿射
        """
        if scale_x is None: scale_x = 1
        if scale_y is None: scale_y = 1
        if rotation is None: rotation = 0

        theta = np.radians(rotation)
        m = [scale_x * np.cos(theta), scale_x * np.sin(theta),
             -scale_y * np.sin(theta), scale_y * np.cos(theta)]
        self.image = self.image.affine(m, interpolate=pyvips.Interpolate.new("nearest"), background=[0])

    @staticmethod
    def numpy2vips(a):
        height, width, bands = a.shape
        linear = a.reshape(width * height * bands)
        vi = pyvips.Image.new_from_memory(linear.data, width, height, bands,
                                          dtype_to_format[str(a.dtype)])
        return vi

    @staticmethod
    def vips2numpy(vi):
        return np.ndarray(buffer=vi.write_to_memory(),
                          dtype=format_to_dtype[vi.format],
                          shape=[vi.height, vi.width, vi.bands])

    def to_image(self):
        '''
        pyvips -> array
        '''
        arr = self.vips2numpy(self.image)
        if arr.ndim == 3:
            if arr.shape[2] != 3:
                arr = arr[:, :, 0]
        return arr


def main():
    pass


if __name__ == '__main__':
    main()

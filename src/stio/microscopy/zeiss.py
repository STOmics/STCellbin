from stio.microscopy import MicroscopeSmallImage, SlideType
# import javabridge
# import bioformats
# import numpy as np
# import pycziutils
# from pylibCZIrw import czi as pyczi
# from tqdm.contrib import itertools as it
import os
from slideio import open_slide
import tifffile
import numpy as np
import glog
from aicsimageio.readers import CziReader

try:
    from _aicspylibczi import BBox, TileInfo
    from aicspylibczi import CziFile
except ImportError:
    raise ImportError(
        "aicspylibczi is required for this reader. "
        "Install with `pip install aicsimageio[czi]`"
    )


class ZeissMicroscopeFile(MicroscopeSmallImage):
    def __init__(self):
        super(ZeissMicroscopeFile, self).__init__()
        self.device.manufacturer = SlideType.Zeiss.value
        self.images_path = None
        self.sizez = None
        self.sizet = None
        self.sizec = None
        self.fov_rows: int = 0
        self.fov_rows: int = 0
        self.channel_infos = {}
        self.scan.exposure_time = 0.0
        self.device.rgb_scale = []

        self.current_channel: str = None

    def read(self, file_path: str, stitch=False):
        self.have_overlap = True
        self.stitch = stitch
        if not os.path.exists(file_path):
            glog.info(f"{file_path} does not exist")
        if os.path.isfile(file_path):
            self.file_path, self.czi_filename = os.path.split(file_path)
            reader = CziReader(file_path)
            self.reader = reader
            czi = CziFile(file_path)
            h = reader.dims.Y
            w = reader.dims.X
            b = reader.dims.M
            self.fov_height = h
            self.fov_width = w
            print(reader.dims)
            "如果reader生成失败，就用默认的0.1 overlap对大图的shape进行计算"
            # todo: 补充上非mosaic的czi格式的图像读取接口
            if czi.is_mosaic():
                bboxes = list(czi.get_all_mosaic_tile_bounding_boxes(S=reader.current_scene_index).values())
                # 默认根据第一个和第二个图片来判断overlap, 如果该overlap差异太大，则用第二个和第三个，直到找到一个合适的overlap
                for i in range(b):
                    if round((bboxes[i + 1].x - bboxes[i].x) / self.fov_width) == 1:
                        self.Overlap = round(
                            (self.fov_width - bboxes[i + 1].x + bboxes[i].x) / self.fov_width, 3)
                        print('overlap', self.Overlap)
                        break
                mosaic_positions = [[bbox.x, bbox.y] for bbox in bboxes]
                mosaic_loc = np.array(mosaic_positions)
                print('MIN of position: ', np.min(mosaic_loc[:, 0]), np.min(mosaic_loc[:, 1]))
                mosaic_loc[:, 0] -= np.min(mosaic_loc[:, 0])
                mosaic_loc[:, 1] -= np.min(mosaic_loc[:, 1])
                mosaic_index = [
                    [
                        round(i[0] / ((1 - self.Overlap) * self.fov_width)),
                        round(i[1] / ((1 - self.Overlap) * self.fov_height))
                    ] for i in mosaic_loc
                ]
                self.ScanCols = (np.max(np.array(mosaic_index)[:, 0]) + 1).item()
                self.ScanRows = (np.max(np.array(mosaic_index)[:, 1]) + 1).item()
                self.loc = np.zeros((self.ScanRows,
                                     self.ScanCols, 2), dtype=int)
                self.GlobalWidth = np.max(mosaic_loc[:, 0]).item() + self.fov_width
                self.GlobalHeight = np.max(mosaic_loc[:, 1]).item() + self.fov_height

                glog.info(f"extracting imgs from czi file: {file_path}")

                # 获取物镜倍数，pixelsize
                print(self.file_path)
                slide = open_slide(file_path, 'CZI')
                scene = slide.get_scene(0)
                self.PixelSizeY = scene.resolution[1] * 1000000
                self.PixelSizeX = scene.resolution[0] * 1000000
                self.ScanObjective = scene.magnification

                self.ImagePath = file_path
                self.Manufacturer = self.device.manufacturer
                self.FOVWidth = self.fov_width
                self.FOVHeight = self.fov_height
                self.fov_dtype = reader.dtype

                # cal global location
                for j in range(b):
                    box = mosaic_loc[j]
                    col = int(mosaic_index[j][0])
                    row = int(mosaic_index[j][1])
                    self.loc[row, col] = [int(box[0]), int(box[1])]
                # 根据通道名选取数据
                # channel_names = reader.channel_names
                # for channel_name in channel_names:
                #     # current_save_path = os.path.join(save_path, channel_name)
                #     # if not os.path.exists(current_save_path): os.makedirs(current_save_path)
                #     fov_save_path = os.path.join(save_path, "{}".format(channel_name))
                #     if not os.path.exists(fov_save_path): os.makedirs(fov_save_path)
                #     channel_index = channel_names.index(channel_name)
                #     data = reader.get_image_dask_data('MYX', C=channel_index)
                #     if stitch:
                #         print('data[0].dtype:', data[0].dtype)
                #         stitched_img = np.zeros((self.GlobalHeight, self.GlobalWidth), dtype=data[0].dtype)
                #     for j in range(b):
                #         mosaic_data = data[j].compute()
                #         box = mosaic_loc[j]
                #         if stitch:
                #             # print('index:', j, ' box:', box, ' stitched_img.shape:',
                #             #       stitched_img.shape, 'overlap:', self.scope_info.ImageInfo.Overlap)
                #             # print('stitch_loc: height, width', int(box[1]), ':', int(box[1])+mosaic_data.shape[0],
                #             #       int(box[0]), ':', int(box[0])+mosaic_data.shape[1], 'm.height, m.width', mosaic_data.shape)
                #             stitched_img[int(box[1]):int(box[1]) + mosaic_data.shape[0],
                #             int(box[0]):int(box[0]) + mosaic_data.shape[1]] = mosaic_data
                #         col = int(mosaic_index[j][0])
                #         row = int(mosaic_index[j][1])
                #         self.loc[row, col] = [int(box[0]), int(box[1])]
                # tifffile.imwrite(
                #     os.path.join(fov_save_path, '{}_{}_{}_m{}.tif'.format(
                #         reader.channel_names[0], str(row).zfill(4), str(col).zfill(4), j)), mosaic_data)
                #     self._file_path["{}".format(channel_name)] = fov_save_path
                #
                # if stitch:
                #     # tifffile.imwrite(os.path.join(current_save_path, 'stitched.tif'), stitched_img[::3, ::3])
                #     tifffile.imwrite(os.path.join(save_path, 'stitched.tif'), stitched_img)
                # transformer to ipr
                self.scan.mosaic_height = self.GlobalHeight
                self.scan.mosaic_width = self.GlobalWidth
                self.fov_location = self.loc
                self.scan.overlap = self.Overlap
                self.scan.fov_rows, self.scan.fov_cols = self.loc.shape[:2]
                self.scan.fov_dtype = self.fov_dtype
                self.fov_rows, self.fov_cols = self.loc.shape[:2]
                self.scan.fov_width = self.fov_width
                self.scan.fov_height = self.fov_height

                # save other attribute
                self.mosaic_loc = mosaic_loc
                self.mosaic_index = mosaic_index
                # glog.info(f"got {len(os.listdir(save_path))} from czi")
            else:
                glog.info("reader of czi generate failed")
        else:
            glog.info("{} is not file".format(file_path))

    def get_fovs_tag(self, ):
        if self.current_channel == None:
            return self.channel_infos
        elif self.current_channel in list(self.channel_infos):
            return self.channel_infos[self.current_channel]["fov_tag"]
        else:
            raise ValueError("{} not in channels".format(self.current_channel))

    def parse_single_img_info(self, fov_save_path, channel_name):
        # if not os.path.exists(current_save_path): os.makedirs(current_save_path)
        if not os.path.exists(os.path.join(fov_save_path, channel_name)):
            os.makedirs(os.path.join(fov_save_path, channel_name))
        channel_index = self.reader.channel_names.index(channel_name)
        data = self.reader.get_image_dask_data('MYX', C=channel_index)
        current_fov_tag = np.empty((self.fov_rows, self.fov_cols), dtype='S256')

        for j in range(self.reader.dims.M):
            mosaic_data = data[j].compute()
            col = int(self.mosaic_index[j][0])
            row = int(self.mosaic_index[j][1])
            current_fov_name = '{}_{}_{}_m{}.tif'.format(channel_name, str(row).zfill(4), str(col).zfill(4), j)
            current_fov_path = os.path.join(fov_save_path, channel_name, current_fov_name)
            tifffile.imwrite(current_fov_path, mosaic_data)
            current_fov_tag[row, col] = os.path.join(channel_name, current_fov_name)
        if channel_name not in list(self.channel_infos.keys()):
            self.channel_infos[channel_name]["Name"] = channel_name
        self.channel_infos[channel_name]["fov_tag"] = current_fov_tag
        self.channel_infos[channel_name]["Path"] = fov_save_path

        return current_fov_tag

    def parse_all_img_info(self, save_path=None):
        """
        write image to save_path
        """
        channel_names = self.reader.channel_names
        self.channel_infos = {}

        for channel_name in channel_names:
            channel_info = {}
            channel_info["Name"] = channel_name
            if save_path is not None:
                self.parse_single_img_info(save_path, channel_name)

            self.channel_infos[channel_name] = channel_info
        return self.channel_infos


#     # 读取显微镜拼接大图
#     def read_stitched_image(self):
#         imagereader = bioformats.ImageReader(self.images_path)
#         image = imagereader.read(rescale=False, wants_max_intensity=True)
#         self.stitched_image = np.transpose(image[0], (2, 0, 1))
#         return self.stitched_image
#
#     # 读取原始FOV小图，返回所有维度按先行后列得FOV列表
#     def read_fov_image(self):
#         reader = pycziutils.get_tiled_reader(self.images_path)
#         tiled_czi_ome_xml = pycziutils.get_tiled_omexml_metadata(self.images_path)
#         tiled_properties_dataframe = pycziutils.parse_planes(tiled_czi_ome_xml)
#         images = []
#         for i, row in tiled_properties_dataframe.iterrows():
#             image = reader.read(series=row["image"], )
#             image = np.transpose(image, (2, 0, 1))
#             images.append(image)
#         self.scan.fov_images = images
#         return images
#
#     def read_meta_data(self):
#         xml = bioformats.get_omexml_metadata(self.images_path)
#         self.scan.overlap = xml.split("Overlap")[1].split("Value")[1].replace(">[", "").\
#             replace("</", "").replace("]", "")
#         xml = bioformats.omexml.OMEXML(xml)
#         # height
#         self.scan.mosaic_height = xml.image().Pixels.SizeX
#         # width
#         self.scan.mosaic_width = xml.image().Pixels.SizeY
#         # stack_count
#         self.sizez = xml.image().Pixels.SizeZ
#         # timepoint_count
#         self.sizet = xml.image().Pixels.SizeT
#         # channel_count
#         self.sizec = xml.image().Pixels.SizeC
#         # 图像位深
#         self.scan.fov_dtype = xml.image().Pixels.PixelType
#         # 拍摄时间
#         self.scan.scan_time = xml.image().AcquisitionDate
#
#     # 写入图像,会自动写入metadata,array.shape示例(16, 512, 512, 3)
#     def write_image(self, array):
#         self.scan.mosaic_height = array.shape[-1]
#         self.sizez = array.shape[0]
#         with pyczi.create_czi(self.images_path, exist_ok=True) as czidoc_w:
#             for z, ch in it.product(range(self.sizez), range(self.scan.mosaic_height)):
#                 # get the 2d array for the current plane and add axis to get (Y, X, 1) as shape
#                 array2d = array[z, ..., ch][..., np.newaxis]
#                 # write the plane with shape (Y, X, 1) to the new CZI file
#                 czidoc_w.write(data=array2d, plane={"Z": z, "C": ch})
#
#
# def main():
#     # 启动java虚拟机
#     javabridge.start_vm(class_path=bioformats.JARS)
#     mmf = ZeissMicroscopeFile(r"D:\DOWN\zeiss\20210427-T242-Z2-L-M024-01222.czi")
#     # mmf.read_meta_data()
#     # print(mmf.scan.mosaic_height)
#     # print(mmf.scan.mosaic_width)
#     # print(mmf.sizez)
#     # print(mmf.sizet)
#     # print(mmf.sizec)
#     # print(mmf.scan.scan_time)
#     # print(mmf.scan.fov_dtype)
#     # print(mmf.bit_depth())
#     # large = mmf.read_fov_image()
#     # fov_images = mmf.read_stitched_image()
#     a = np.zeros((16, 1024, 512, 3), dtype=np.uint8)
#     mmf.write_image(a)
#     # 关闭java虚拟机
#     javabridge.kill_vm()
#
#
# if __name__ == '__main__':
#     main()


def main():
    mmf = ZeissMicroscopeFile()
    czi_file = r"D:\03_data\08_test_data\02_mIF\multi_flou\Y00666C1\image\raw\Y00666C1.czi"
    czi_file = r"D:\03_data\08_test_data\02_mIF\multi_flou\Y00666C4\image\raw\Y00666C5.czi"
    save_path = r"D:\03_data\08_test_data\02_mIF\multi_flou\Y00666C4\image\qc"
    mmf.read(czi_file)
    print("stop")


if __name__ == '__main__':
    main()

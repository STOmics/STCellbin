# from gefpy.plot import get_exp_from_gemfile,get_binsize_from_gemfile,save_exp_heat_map_by_binsize,save_exp_heat_map,cgef_stat
import gzip
import warnings
import pandas as pd
import numpy as np
from os import path as osp
import os
# from gefpy.bgef_writer_cy import gem2tif
# from gefpy.bgef_writer_cy import generate_bgef
# from gefpy.gef_to_gem_cy import gefToGem
# from gefpy.bgef_reader_cy import BgefR
# from gefpy.cgef_writer_cy import generate_cgef
# import tifffile

# __bit8_range__ = 256
# __bit16_range__ = 65535

class MatrixLoader(object):
    def __init__(self, file, output=None):
        self.gene_mat = np.array([])
        self.x_start = 65535
        self.y_start = 65535
        self.binsize = None
        self._file = file
        self._output = output

    def f_gene2img_pd(self, chunk_size=1024 * 1024 * 10):
        """
        :param file: Gene Matrix
        :param chunk_size: Rows of Single read
        :return: uint8(img)
        """
        img = np.zeros((1, 1), np.uint8)
        suffix = osp.splitext(self._file)[1]
        if suffix in ['.txt', '.gz', '.gef', '.gem']:
            if suffix == '.gef':
                _, _, self.x_start, self.y_start, gene_mat = self.gef2image(self._file)

            elif suffix == ".gem":
                gene_mat = gem2tif(self._file, self._output)
                self.x_start = 0
                self.y_start = 0

            else:
                if suffix == '.gz': fd = gzip.open(self._file, 'rb')
                else: fd = open(self._file, 'rb')
                title = ""
                eoh = 0  # set fd at file head
                for line in fd:
                    line = line.decode("utf-8")
                    if not line.startswith('#'):
                        title = line
                        break
                    else: eoh = fd.tell()
                fd.seek(eoh)
                # parse file title，set chunk size
                title = title.strip('\n').split('\t')
                umi_count_name = [i for i in title if "ount" in i][0]
                title = ["x", "y", umi_count_name]
                df = pd.read_csv(fd, sep='\t', header=0, usecols=title,
                                 dtype=dict(zip(title, [np.uint16] * 3)),
                                 chunksize=chunk_size)
                # incompatibility warning: pytables & pandas
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    try:
                        for chunk in df:
                            # DataFrame to Image
                            tmp_h = chunk['y'].max() + 1
                            tmp_w = chunk['x'].max() + 1
                            tmp_min_y = chunk['y'].min()
                            tmp_min_x = chunk['x'].min()
                            if tmp_min_x < self.x_start:
                                self.x_start = tmp_min_x
                            if tmp_min_y < self.y_start:
                                self.y_start = tmp_min_y

                            h, w = img.shape[:2]

                            chunk = chunk.groupby(['x', 'y']).agg(UMI_sum=(umi_count_name, 'sum')).reset_index()
                            chunk['UMI_sum'] = chunk['UMI_sum'].mask(chunk['UMI_sum'] > 255, 255)
                            tmp_img = np.zeros(shape=(tmp_h, tmp_w), dtype=np.uint8)
                            tmp_img[chunk['y'], chunk['x']] = chunk['UMI_sum']

                            # Resize matrix to prepare for operation
                            ext_w = tmp_w - w
                            ext_h = tmp_h - h
                            if ext_h > 0:
                                img = np.pad(img, ((0, abs(ext_h)), (0, 0)), 'constant')
                            elif ext_h < 0:
                                tmp_img = np.pad(tmp_img, ((0, abs(ext_h)), (0, 0)), 'constant')
                            if ext_w > 0:
                                img = np.pad(img, ((0, 0), (0, abs(ext_w))), 'constant')
                            elif ext_w < 0:
                                tmp_img = np.pad(tmp_img, ((0, 0), (0, abs(ext_w))), 'constant')

                            # overflow prevention
                            tmp_img = 255 - tmp_img  # old b is gone shortly after new array is created
                            np.putmask(img, tmp_img < img, tmp_img)  # a temp bool array here, then it's gone
                            img += 255 - tmp_img  # a temp array here, then it's gone
                    except:
                        pass

                df.close()
                gene_mat = img[self.y_start:, self.x_start:]
        return gene_mat, self.x_start, self.y_start

    def gem2tif(self, gem_path, tif_path):
        gem2tif(gem_path, tif_path)
        while True:
            if os.path.exists(tif_path):
                break
        img = tifffile.imread(tif_path)
        return img

    @staticmethod
    # 读取gem文件 返回fileError, x, y, count
    def get_exp_from_gemfile(input_file, dot_size):
        fileError, x, y, count = get_exp_from_gemfile(input_file,dot_size)
        return fileError, x, y, count

    # 读取gem文件中的binsize
    def get_binsize_from_gemfile(self,):
        isHeaderInFile,self.binsize = get_binsize_from_gemfile()
        return isHeaderInFile,self.binsize

    # 保存gem至热图
    def save_exp_heat_map_by_binsize(self, input_file, output_png, bin_size):
        save_exp_heat_map_by_binsize(input_file, output_png, bin_size)

    # 保存bgef至热图
    def save_exp_heat_map(self, input_gef, output_png):
        save_exp_heat_map(input_gef, output_png)

    # cgef转png
    def cgef2png(self, input_cgef, png_path):
        cgef_stat(input_cgef,png_path)

    # gem转bgef,region为指定ROI区域
    def gem2bgef(self, gem_file, bgef_file, binsize, region):
        generate_bgef(gem_file, bgef_file,bin_sizes=binsize,region=region)

    # bgef转gem
    def bgef2gem(self, gem_file, bgef_file, binsize):
        obj = gefToGem(gem_file, bgef_file)
        obj.bgef2gem(bgef_file,binsize)

    # 从bGEF生成cGEF,example:
    # mask_file = "../test_data/FP200000617TL_B6/FP200000617TL_B6_mask.tif"
    # bgef_file = "../test_data/FP200000617TL_B6/stereomics.h5"
    # cgef_file = "../test_data/FP200000617TL_B6/FP200000617TL_B6.gefpy.cgef"
    # block_sizes = [256, 256]
    def bgef2cgef(self, cgef_file, bgef_file, mask_file, block_sizes):
        generate_cgef(cgef_file, bgef_file, mask_file, block_sizes)

    def read_bgef(self, path):
        bgef = BgefR(path)
        exp = bgef.get_expression()

        return exp

    def gef2image(self, gef, ResultPath=None):
        import  h5py
        from PIL import Image as PIL_Image
        with h5py.File(gef, 'r') as f:
            data = f['/geneExp/bin1/expression'][:]
            x_start = f['/geneExp/bin1/expression'].attrs['minX']
            y_start = f['/geneExp/bin1/expression'].attrs['minY']
        if not data.size:
            return None
        df = pd.DataFrame(data, columns=['x', 'y', 'count'], dtype=np.uint32)
        # clog.info("min x: {} min y: {}".format(df['x'].min(), df['y'].min()))
        df['x'] = df['x'] - df['x'].min()
        df['y'] = df['y'] - df['y'].min()
        max_x = df['x'].max() + 1
        max_y = df['y'].max() + 1
        # clog.info("image dimension: {} x {} (width x height)".format(max_x, max_y))
        new_df = df.groupby(['x', 'y']).agg(UMI_sum=('count', 'sum')).reset_index()
        image = np.zeros(shape=(max_y, max_x), dtype=np.uint8)
        image[new_df['y'], new_df['x']] = new_df['UMI_sum']

        return np.sum(image, 0).astype('float64'), np.sum(image, 1).astype('float64'), x_start, y_start, image


if __name__ == '__main__':
    file = r"D:\Data\mif\SS200000059_NC_fov\SS200000059_NC.raw.gef"
    output = r"D:\Data\mif\SS200000059_NC_fov\SS200000059_NC.raw.tif"
    ml = MatrixLoader(file=file, output=output)
    gene_mat, x_start, y_start = ml.f_gene2img_pd()
    tifffile.imwrite(output, gene_mat)

import torch
import os

import sys
import h5py
import argparse
from shutil import copyfile
import json
from datetime import datetime
from json import loads, dumps, dump
import tarfile
from glob import glob
import pandas as pd
import gzip
import shutil
from importlib.metadata import version
import traceback
import multiprocessing as mp
import tifffile

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(ROOT)

from stereocell_v2.cellbin.modules.stitching import Stitching
from stereocell_v2.cellbin.modules.registration import Registration
from stereocell_v2.cellbin.modules.tissue_segmentation import TissueSegmentation

from stereocell_v2.stio.matrix_loader import MatrixLoader
from stereocell_v2.cellbin.modules.cell_labelling import CellLabelling
from stereocell_v2.cellbin.utils import clog
from stereocell_v2.cellbin.image import Image
from stereocell_v2.cellbin.dnn.weights import auto_download_weights
# from cellbin.image.augmentation import clarity_enhance_method
from stereocell_v2.cellbin.utils.file_manager import search_files, rc_key
from stereocell_v2.cellbin.image.augmentation import f_resize
from stereocell_v2.cellbin.image.augmentation import f_ij_16_to_8
from stereocell_v2.cellbin.image.augmentation import f_gray2bgr

from stereocell_v2.cellbin.image.augmentation import f_ij_auto_contrast
from stereocell_v2.cellbin.image.augmentation import f_rgb2gray
from stereocell_v2.cellbin.modules.iqc.clarity_qc import ClarityQC
from stereocell_v2.calibration.match_calibration import FFTRegister
from stereocell_v2.stio.chip import STOmicsChip

# Constant
PROG_VERSION = 'SAP'
JIQUN_CF = "/hwfssz1/ST_BIOINTEL/P20Z10200N0039/06.groups/st_ster/pipeline_config.json"
JIQUN_ZOO = "/hwfssz1/ST_BIOINTEL/P20Z10200N0039/06.groups/st_ster/cellbin_weights"
IMAGE_SUPPORT = ['.jpg', '.png', '.tif', '.tiff']

"""
Eval score of accuracy of tissuecut result
"""
import sys
import copy
import math

import numpy as np
import cv2

from stereocell_v2.cellbin.image.mask import f_fill_all_hole

CNT_PARAGRAPH = 20


# FIXME: Please refactor this function!!!!!
def eval_point(img_contrast, img_mask):
    """
    Main entry for eval point of tissuecut result
    """
    img_c = copy.deepcopy(img_contrast)
    mask = copy.deepcopy(img_mask)
    mask[mask > 0] = 255
    mask = f_fill_all_hole(mask)

    dconts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    dconts_area = []
    for cnt in dconts:
        dconts_area.append(cnt_area(cnt))
    max_dcont_area = max(dconts_area)

    densities = []
    for i, cnt in enumerate(dconts):
        area = dconts_area[i]
        l = cnt_len(cnt)
        if CNT_PARAGRAPH * 10 > l or area * 3 < max_dcont_area:
            continue
        morp = np.zeros(mask.shape[:2], np.uint8)
        cv2.drawContours(morp, dconts, i, 255, -1)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        morp_out = cv2.morphologyEx(
            morp, cv2.MORPH_DILATE, kernel, iterations=13
        ) - cv2.morphologyEx(morp, cv2.MORPH_DILATE, kernel, iterations=3)
        mask_not = cv2.bitwise_not(mask)
        morp_out = cv2.bitwise_and(morp_out, mask_not, mask=mask_not)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        morp_in = cv2.morphologyEx(
            morp, cv2.MORPH_ERODE, kernel, iterations=3
        ) - cv2.morphologyEx(morp, cv2.MORPH_ERODE, kernel, iterations=13)
        morp_in = cv2.bitwise_and(morp_in, mask, mask=mask)

        _, cnts_out, _, pt_out = edge_to_cnts(morp_out, [])
        _, cnts_in, _, _ = edge_to_cnts(morp_in, pt_out)

        pair_lst, _ = pair_cnts(cnts_out, cnts_in)

        for idx_out, idx_in in pair_lst:
            density_out = get_pixels_density(
                cv2.bitwise_and(img_c, img_c, mask=mask_not), cnts_out[idx_out], 127
            )
            density_in = get_pixels_density(
                cv2.bitwise_and(img_c, img_c, mask=mask), cnts_in[idx_in], 127
            )
            if density_out == 0:
                densities.append(sys.maxsize)
                continue
            densities.append(density_in / density_out)

    if len(densities) < 1:
        density = 0
    else:
        density = np.min(densities)
    return int(get_score(density))


def get_score(x):
    """
    Get score
    """
    x_max = 100
    x_min = 0
    if x > x_max:
        return 100

    x = (x - x_min) / (x_max - x_min)
    y = 100 * np.power(x, 0.5)

    return y


def cnt_len(cnt):
    """
    Return arc length of the contour
    """
    area = cv2.arcLength(cnt, True)
    return area


def cnt_area(cnt):
    """
    Return area of the contour
    """
    area = cv2.contourArea(cnt)
    return area


def adj_cnt(cnt1, cnt2, pts):
    """
    Have no idea with this function
    """
    if len(pts) > 0:
        # find the nearest start point in previous cnt
        # pylint: disable=unpacking-non-sequence
        x1, y1 = np.int32(np.sum(cnt1, axis=0)[0] / len(cnt1))
        min_dis = sys.maxsize
        min_idx = 0
        for i, p in enumerate(pts):
            x2, y2 = p[1]
            flag = p[2]
            if flag != 0:
                continue
            dis = np.square(x1 - x2) + np.square(y1 - y2)
            if dis < min_dis:
                min_dis = dis
                min_idx = i

        x, y = pts[min_idx][0]
        pts[min_idx][2] = 1

        # find the nearest point in cnt1
        ti = 0
        min_dis = sys.maxsize
        for i, c in enumerate(cnt1):
            x2, y2 = c[0][:2]
            dis = np.square(x - x2) + np.square(y - y2)
            if dis < min_dis:
                min_dis = dis
                ti = i
        a = cnt1[:ti]
        b = cnt1[ti:]
        if len(a) == 0:
            cnt1 = b
        elif len(b) == 0:
            cnt1 = a
        else:
            cnt1 = np.concatenate((b, a))

    x0, y0 = cnt1[0][0][:2]
    # move the start point to the same index
    dis = sys.maxsize
    ti = 0
    for i, c in enumerate(cnt2):
        tx, ty = c[0][:2]
        l = math.sqrt((x0 - tx) ** 2 + (y0 - ty) ** 2)
        if l < dis:
            ti = i
            dis = l
    a = cnt2[:ti]
    b = cnt2[ti:]
    if len(a) == 0:
        cnt2 = b
    elif len(b) == 0:
        cnt2 = a
    else:
        cnt2 = np.concatenate((b, a))

    # check the direction
    x1, y1 = cnt1[CNT_PARAGRAPH][0][:2]
    x2, y2 = cnt2[CNT_PARAGRAPH][0][:2]
    x3, y3 = cnt2[::-1][CNT_PARAGRAPH][0][:2]
    l1 = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    l2 = math.sqrt((x1 - x3) ** 2 + (y1 - y3) ** 2)
    if l1 > l2:
        cnt2 = cnt2[::-1]
    return cnt1, cnt2


def edge_to_cnts(morp, pt):
    """
    Have no idea with this function
    """
    conts, hiers = cv2.findContours(morp, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    # find all target cnts pair
    cont_arr = []
    for i, c in enumerate(conts):
        if hiers[0][i][3] != -1:
            continue
        l = cnt_len(c)
        if l > CNT_PARAGRAPH and hiers[0][i][2] > -1:
            tmp = []
            for j, h in enumerate(hiers[0]):
                if h[3] == i:
                    tmp.append(conts[j])
            tmp.sort(key=cnt_area, reverse=True)
            if len(tmp[0]) > CNT_PARAGRAPH:
                cont_arr.append([c, tmp[0]])
    out_pt = []
    for cnt in cont_arr:
        # adjust points list order
        cnt[0], cnt[1] = adj_cnt(cnt[0], cnt[1], pt)
        centroid = np.int32(np.sum(cnt[1], axis=0)[0] / len(cnt[1]))
        out_pt.append([cnt[1][0][0][:2], centroid, 0])
        step = 10
        s1 = math.ceil(len(cnt[0]) / step)
        s2 = math.ceil(len(cnt[1]) / step)
        for i in range(step):
            x1, y1 = cnt[0][i * s1][0][:2]
            x2, y2 = cnt[1][i * s2][0][:2]
            cv2.line(morp, (x1, y1), (x2, y2), 0, 5)
    conts, hiers = cv2.findContours(morp, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    dconts_area = []
    for i in conts:
        dconts_area.append(cnt_area(i))
    max_dcont_area = max(dconts_area)

    del_idxs = []
    for i in range(len(conts)):
        if dconts_area[i] * 3 < max_dcont_area:
            del_idxs.append(i)

    if len(del_idxs) > 0:
        conts = np.array(conts, dtype=object)
        conts = np.delete(conts, del_idxs)
        conts = tuple(conts.tolist())
        hiers = np.delete(hiers, del_idxs, axis=1)
    return True, conts, hiers, out_pt


def pair_cnts(cnts1, cnts2):
    """
    Have no idea with this function
    """
    pair_lst = []
    centroid_lst = []

    cnts2_dict = {}
    for i, c in enumerate(cnts2):
        # pylint: disable=unpacking-non-sequence
        x, y = np.int32(np.sum(c, axis=0)[0] / len(c))
        cnts2_dict[i] = [x, y]

    for i, c in enumerate(cnts1):
        # pylint: disable=unpacking-non-sequence
        x, y = np.int32(np.sum(c, axis=0)[0] / len(c))
        min_dis = sys.maxsize
        min_idx = -1
        min_xy = [x, y]
        for k, v in cnts2_dict.items():
            if v == [-1, -1]:
                continue
            x2, y2 = v
            dis = np.square(x - x2) + np.square(y - y2)
            if dis < min_dis:
                min_dis = dis
                min_idx = k
                min_xy = [x2, y2]
        if min_idx > -1:
            pair_lst.append([i, min_idx])
            centroid_lst.append([[x, y], min_xy])
            cnts2_dict[min_idx] = [-1, -1]
    return pair_lst, centroid_lst


def get_pixels_density(img, cnt, thre=127):
    """
    Get pixels density of the image
    """
    x, y, w, h = cv2.boundingRect(cnt)
    mask_win = np.zeros((h, w), dtype=np.uint8)
    sub_arr = np.array([[x, y]])
    cnt_win = copy.deepcopy(cnt) - sub_arr
    cv2.drawContours(mask_win, [cnt_win], 0, 255, -1)
    img_win = img[y: y + h, x: x + w]
    img_win = cv2.bitwise_and(img_win, img_win, mask=mask_win)
    return np.sum(img_win > thre) / np.sum(mask_win > 0)


def transfer_16bit_to_8bit(image_16bit):
    """
    Transfer the bit deepth of image from 16bit to 8bit
    """
    min_16bit = np.min(image_16bit)
    max_16bit = np.max(image_16bit)
    div = 255 / (max_16bit - min_16bit)

    image_8bit = np.zeros(image_16bit.shape, dtype=np.uint8)
    chunk_size = 10000
    for idx in range(image_16bit.shape[0] // chunk_size + 1):
        s = slice(idx * chunk_size, (idx + 1) * chunk_size)
        image_8bit[s] = np.array(
            np.rint((image_16bit[s, :] - min_16bit) * div), dtype=np.uint8
        )

    return image_8bit


def resize(img, shape=(1024, 2048), mode="NEAREST"):
    """
    Make image size larger or smaller
    """
    from PIL import Image

    imode = Image.NEAREST
    if mode == "BILINEAR":
        imode = Image.BILINEAR
    elif mode == "BICUBIC":
        imode = Image.BICUBIC
    elif mode == "LANCZOS":
        imode = Image.LANCZOS
    elif mode == "HAMMING":
        imode = Image.HAMMING
    elif mode == "BOX":
        imode = Image.BOX
    ori_image = Image.fromarray(img)
    image_thumb = ori_image.resize((shape[1], shape[0]), imode)
    image_thumb = np.array(image_thumb).astype(np.uint8)
    return image_thumb


class TissuecutIntensity:
    """
    Tissuecut by intensity filter
    """

    def __init__(self, img_file, threshold):
        self.img_file = img_file
        self.threshold = threshold

    def tissue_infer_threshold(self):
        """
        Main entry for tissuecut with intensity
        """
        img = tifffile.imread(self.img_file)

        threshold, mask = self.if_tissue_mask(img)
        if mask.dtype != "uint8":
            mask = transfer_16bit_to_8bit(mask)
        score = self.get_score(img, mask)

        return mask, score, threshold

    def get_score(self, img, tissue):
        """
        Get score for IF stain type
        """
        tmp_img = f_ij_auto_contrast(img)
        if tmp_img.dtype != "uint8":
            tmp_img = f_ij_16_to_8(tmp_img)

        tmp_img = np.squeeze(tmp_img)
        tmp_img = resize(tmp_img, (512, 512))

        resize_arr = resize(tissue, (512, 512))
        kenel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        closure_arr = cv2.morphologyEx(resize_arr, cv2.MORPH_CLOSE, kenel)

        try:
            score = eval_point(tmp_img, closure_arr)
        except:  # pylint: disable=bare-except
            score = 20.0
        return score

    def if_tissue_mask(self, image):
        """
        if tissue mask
        :param image: W*H
        :return: mask, threshold
        """
        if self.threshold:
            # use threshold from input parameters
            _, mask = cv2.threshold(image, self.threshold, 255, cv2.THRESH_BINARY)
            threshold = self.threshold
        else:
            # OTSU method for calculating threshold
            assert image.ndim == 2, f"image must be 2 dim. shape:{image.shape}"
            image = cv2.GaussianBlur(image, (5, 5), 0)
            threshold, mask = cv2.threshold(image, 0, 255, cv2.THRESH_OTSU)
        return threshold, mask


def generate_gem(mask_paths, gem_path, out_path):
    def row(gem_path):
        if gem_path.endswith('.gz'):
            with gzip.open(gem_path, 'rb') as f:
                first_line = bytes.decode(f.readline())
                if '#' in first_line:
                    rows = 6
                else:
                    rows = 0
        else:
            with open(gem_path, 'rb') as f:
                first_line = bytes.decode(f.readline())
                if '#' in first_line:
                    rows = 6
                else:
                    rows = 0
        return rows

    clog.info("Reading data..")
    gem = pd.read_csv(gem_path, sep='\t', skiprows=row(gem_path))
    assert "MIDCount" in gem.columns
    gem['x'] -= gem['x'].min()
    gem['y'] -= gem['y'].min()

    for mp in mask_paths:
        mask_path = mp
        filename = mask_path.replace('\\', '/').split('/')[-1].split('.')[0]
        i = Image()
        i.read(mask_path)
        mask = i.image
        mask[mask > 0] = 1
        mask = mask.astype(np.uint8)
        _, maskImg = cv2.connectedComponents(mask, connectivity=8)
        cur_gem = gem.copy(deep=True)
        cur_gem['CellID'] = maskImg[cur_gem['y'], cur_gem['x']]

        cell_gem = os.path.join(out_path, f'{filename}.gem')

        cur_gem.to_csv(cell_gem, sep='\t', index=False)
        # os.system('gzip {}'.format(os.path.join(out_path, f'{filename}.gem')))
        with open(cell_gem, 'rb') as f_in:
            with gzip.open(cell_gem + '.gz', 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        os.remove(cell_gem)


class binary_mask_rle(object):
    """
    ref.: https://en.wikipedia.org/wiki/Run-length_encoding
    """

    def __init__(self):
        pass

    def encode(self, binary_mask):
        '''
        binary_mask: numpy array, 1 - mask, 0 - background
        Returns run length as string formated
        '''
        binary_mask[binary_mask >= 1] = 1
        pixels = binary_mask.flatten()
        pixels = np.concatenate([[0], pixels, [0]])
        runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
        runs[1::2] -= runs[::2]
        runs = np.reshape(runs, (-1, 2))
        return runs

    def decode(self, mask_rle, shape):
        '''
        mask_rle: run-length as string formated (start length)
        shape: (height,width) of array to return
        Returns numpy array, 1 - mask, 0 - background
        '''
        # s = mask_rle.split()
        # starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
        starts = mask_rle[:, 0]
        lengths = mask_rle[:, 1]
        starts -= 1
        ends = starts + lengths
        binary_mask = np.zeros(shape[0] * shape[1], dtype=np.uint8)
        for lo, hi in zip(starts, ends):
            binary_mask[lo:hi] = 1
        return binary_mask.reshape(shape)


def file_rc_index(file_name):
    tags = os.path.split(file_name)[1].split('_')
    xy = list()
    for tag in tags:
        if (len(tag) == 4) and tag.isdigit(): xy.append(tag)
    c = xy[0]
    r = xy[1]
    return [int(r), int(c)]


def imagespath2dict(images_path):
    fov_images = search_files(images_path, exts=IMAGE_SUPPORT)
    src_fovs = dict()
    for it in fov_images:
        col, row = file_rc_index(it)
        src_fovs[rc_key(row, col)] = it

    return src_fovs


def outline(image, line_width):
    import cv2 as cv
    image = np.where(image != 0, 1, 0).astype(np.uint8)
    edge = np.zeros((image.shape), dtype=np.uint8)
    contours, hierachy = cv.findContours(image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    r = cv.drawContours(edge, contours, -1, (255, 255, 255), line_width)
    return r


def _write_attrs(gp, d):
    """ Write dict to hdf5.Group as attributes. """
    for k, v in d.items():
        gp.attrs[k] = v


def json_serialize(obj, file_path: str):
    with open(file_path, 'w', encoding='utf-8') as fd:
        str_dct = dumps(obj, default=lambda o: o.__dict__)
        dump(loads(str_dct), fd, indent=2, ensure_ascii=False)


def splitImage(im, tag, imgSize, h5_path, bin_size):
    """ Split image into patches with imgSize and save to h5 file. """
    # get number of patches
    height, width = im.shape[:2]
    # num_x = int(width/imgSize) + 1
    # num_y = int(height/imgSize) + 1
    num_x = math.ceil(width / imgSize)
    num_y = math.ceil(height / imgSize)

    with h5py.File(h5_path, 'a') as out:
        group = out.require_group(f'{tag}/bin_{bin_size}')

        # write attributes
        attrs = {'sizex': width,
                 'sizey': height,
                 'XimageNumber': num_x,
                 'YimageNumber': num_y}
        _write_attrs(group, attrs)

        # write dataset
        for x in range(0, num_x):
            for y in range(0, num_y):
                # deal with last row/column images
                x_end = min(((x + 1) * imgSize), width)
                y_end = min(((y + 1) * imgSize), height)
                if im.ndim == 3:
                    small_im = im[y * imgSize:y_end, x * imgSize:x_end, :]
                else:
                    small_im = im[y * imgSize:y_end, x * imgSize:x_end]

                data_name = f'{x}/{y}'
                # h_, w_ = small_im.shape
                # if h_ * w_ == 0: continue
                try:
                    # normal dataset creation
                    group.create_dataset(data_name, data=small_im)
                except Exception as e:
                    # if dataset already exists, replace it with new data
                    del group[data_name]
                    group.create_dataset(data_name, data=small_im)


def createPyramid(imgs, h5_path, imgSize=256, x_start=0, y_start=0, mag=(2, 10, 50, 100, 15)):
    """ Create image pyramid and save to h5. """
    img = imgs['ssDNA']
    # get height and width
    height, width = img.shape[:2]
    # im = np.rot90(im, 1)  ## 旋转图片，配准后的图片应该不用旋转了

    # write image metadata
    with h5py.File(h5_path, 'a') as h5_out:
        meta_group = h5_out.require_group('metaInfo')
        info = {'imgSize': imgSize,
                'x_start': x_start,
                'y_start': y_start,
                'sizex': width,
                'sizey': height,
                'version': '0.0.1'}
        _write_attrs(meta_group, info)

    # write image pyramid of bin size
    for k, img_p in imgs.items():
        # if img.dtype != 'uint8':
        #     img_p = transfer_16bit_to_8bit(img_p)
        for bin_size in mag:
            if k == 'CellMask':
                img_p = outline(img_p, line_width=2)
            if k == 'TissueMask':
                img_p = outline(img_p, line_width=100)
            im_downsample = img_p[::bin_size, ::bin_size]
            splitImage(im_downsample, k, imgSize, h5_path, bin_size)


class Pipeline(object):
    def __init__(self):
        self._ipr_path = None
        self._gem_path = None

        self.t_seg = None
        self.c_seg = None
        self.clarity_eval = None
        self.stitcher = Stitching()
        self.registrator = Registration()
        self.chip_template_getter = STOmicsChip()
        self.bmr = binary_mask_rle()

        self._input_path = None
        self.image_paths = None
        self._output_path = None
        self.is_stitched = None

        self._stereo_chip = None
        self._stain_type = None
        self._cell_seg_type = None
        self.chip_name = None
        self.config_file = None
        self.zoo_dir = None
        self.c_config = None
        self.t_config = None
        self.clarity_config = None
        self.stitch_config = None
        self.regist_config = None
        self.weight_names = None
        self.x_start = None
        self.y_start = None

        self._stitch_img = None
        self._regist_img = None
        # self._tissue_img = None
        # self._cell_img = None
        # self.log_file = None

        self.mask = None

        # self.file_name = None
        self.version = None
        self.debug_mode = False  # TODO: 后面这个需要加个接口! @dzh

        # 命名
        self.stitch_img_name = 'fov_stitched'
        self.stitch_template = 'stitch_template'
        self.gene_template = 'matrix_template'
        self.stitch_transform_name = 'fov_stitched_transformed'
        self.transform_template = 'transform_template'
        self.regist_img_name = 'regist'
        self.gene_img_name = 'gene'
        self.tissue_mask_name = 'tissue_cut'
        self.cell_mask_name = 'mask'
        self.cell_correct_mask_name = 'cell_mask_correct'
        self.clarity_eval_name = 'clarity_eval'
        self.filtered_cell_mask_name = 'cell_mask_filter'
        self.filtered_cell_correct_mask_name = 'cell_mask_correct_filter'

        self.img_ext = '.tif'
        self.txt_ext = '.txt'

        clog.info("Init success.")

    def set_stereo_chip(self, c):
        self._stereo_chip = c

    def set_cell_seg_type(self, ct):
        self._cell_seg_type = ct

    def set_ipr_path(self, i):
        self._ipr_path = i

    def set_config_file(self, cf):
        with open(cf, 'r') as f:
            th_dict = json.load(f)
        self.config_file = th_dict
        self.c_config = self.config_file['cell_seg']
        self.t_config = self.config_file['tissue_seg']
        self.clarity_config = self.config_file['clarity_eval']
        self.stitch_config = self.config_file['stitch']
        self.regist_config = self.config_file.get('registration', {})
        cpu_count = mp.cpu_count()
        self.stitch_config['running_config']['num_threads'] = min(cpu_count // 2,
                                                                  self.stitch_config['running_config']['num_threads'])
        self.config_file['cell_correct']['num_threads'] = min(cpu_count // 2,
                                                              self.config_file['cell_correct']['num_threads'])
        clog.info(f"Using threads for stitch  {self.stitch_config['running_config']['num_threads']}\n"
                  f"using threads for  cell correct {self.config_file['cell_correct']['num_threads']}")

    def set_zoo_dir(self, path: str):
        self.zoo_dir = path

    def set_version(self, v: str):
        self.version = v

    def auto_load_weights(self, ):
        self.weight_names = [
            self.c_config[self._cell_seg_type][self._stain_type],
            self.t_config[self._stain_type],
        ]
        auto_download_weights(self.zoo_dir, self.weight_names)  # if empty, auto download weights

    def prepare_input(self):
        clog.info(f"Preparing input for pipeline (images)")
        fname, ext = os.path.splitext(self._input_path)
        if ext == '.gz':
            # 解压文件 （延用开发版本）
            with tarfile.open(self._input_path, 'r') as tfo:
                for tarinfo in tfo:
                    if tarinfo.name == '':
                        continue
                    if os.path.splitext(tarinfo.name)[1] == '' or os.path.splitext(tarinfo.name)[1] == '.tif' or \
                            os.path.splitext(tarinfo.name)[1] == '.czi' or os.path.splitext(tarinfo.name)[1] == '.tiff':
                        tfo.extract(tarinfo, self._output_path)
            if not self.is_stitched:
                # 找到小图路径
                for name in os.listdir(self._output_path):
                    cur_path = os.path.join(self._output_path, name)
                    clog.info(f"Current path -> {cur_path}")
                    if os.path.isdir(cur_path):
                        img_dir = cur_path
                        break
                else:
                    raise Exception(f"Could not find fov directory")

            else:
                # 找到大图路径
                for f in os.listdir(self._output_path):
                    large_path = os.path.join(self._output_path, f)
                    if large_path.endswith('.tif'):
                        img_dir = large_path

        elif ext == '':  # 小图路径
            img_dir = self._input_path
            self.img_paths = {}
            img_dir_splits = img_dir.split(',')
            for sp in img_dir_splits:
                file_name = os.path.basename(sp).split(".")[0]
                if self.chip_name == file_name:
                    self.img_paths['DAPI'] = sp
                elif self.chip_name in file_name and "_IF" in file_name:
                    if_name = file_name.split("_")[-2]
                    self.img_paths[if_name] = sp
            print("asd")
        elif ext in IMAGE_SUPPORT:  # 单张大图
            self.is_stitched = True
            img_dir = self._input_path
            self.img_paths = {}
            img_dir_splits = img_dir.split(',')
            for sp in img_dir_splits:
                file_name = os.path.basename(sp).split(".")[0]
                if self.chip_name == file_name:
                    self.img_paths['DAPI'] = sp
                elif self.chip_name in file_name and "_IF" in file_name:
                    if_name = file_name.split("_")[-2]
                    self.img_paths[if_name] = sp
        clog.info(f"Source image path(dir) {img_dir}")
        if not self.is_stitched:
            self.image_paths = imagespath2dict(img_dir)
        else:
            self.image_paths = img_dir

    def initialize_clarity_eval(self):
        try:
            clog.info('Initialize for clarity eval')
            self.clarity_eval = ClarityQC()
            self.clarity_eval.load_model(
                model_path=os.path.join(self.zoo_dir, self.clarity_config[self._stain_type]),
                batch_size=self.clarity_config['running_config']['batch_size'],
                gpu=self.clarity_config['running_config']['gpu']
            )
            clog.info('Load weights (Clarity Eval-{}) finished.'.format(self._stain_type))
        except Exception as e:
            clog.error(traceback.format_exc())

    def initialize_t_seg(self):
        try:
            clog.info('Initialize for tissue seg')
            clog.info(f'Tissue seg weight: {os.path.join(self.zoo_dir, self.t_config[self._stain_type])}')
            self.t_seg = TissueSegmentation(
                model_path=os.path.join(self.zoo_dir, self.t_config[self._stain_type]),
                stype=self._stain_type,
                gpu=self.t_config['running_config']['gpu'],
                num_threads=self.t_config['running_config']['num_threads']
            )
            clog.info('Load weights (tissue-{}) finished.'.format(self._stain_type))
        except Exception as e:
            clog.error(traceback.format_exc())

    def initialize_c_seg(self):
        try:
            clog.info('Initialize for cell seg')
            self.c_seg = CellSegmentation(
                model_path=os.path.join(self.zoo_dir, self.c_config[self._cell_seg_type][self._stain_type]),
                gpu=self.c_config["running_config"]["gpu"],
                num_threads=self.c_config['running_config']['num_threads']
            )
            clog.info('Load weights (cell-{}-{}) finished.'.format(self._cell_seg_type, self._stain_type))
        except Exception as e:
            clog.error(traceback.format_exc())

    def load_initial_info(self, ):
        try:
            self.protein_type = []
            clog.info('Load initial info (StainType, ChipSN...)')
            with h5py.File(self._ipr_path, 'r') as conf:
                for grp in conf.keys():
                    if grp not in ['ManualState', 'StereoResepSwitch', 'Research']:
                        self.protein_type.append(grp)
                self._stain_type = conf['DAPI']['QCInfo'].attrs['StainType']
                self._stain_type = self._stain_type.upper()
                self.chip_name = conf['DAPI']['ImageInfo'].attrs['STOmicsChipSN']
                if not self.chip_template_getter.is_chip_number_legal(self.chip_name):
                    raise Exception(f"{self.chip_name} not supported")
                    # 获取芯片号对应的芯片模板
                short_sn = self.chip_template_getter.get_valid_chip_no(self.chip_name)
                self._stereo_chip = self.chip_template_getter.get_chip_grids(short_sn)
                self.is_stitched = conf['DAPI']['ImageInfo'].attrs['StitchedImage']
        except Exception as e:
            clog.error(traceback.format_exc())

    def initial_ipr_attrs(self, ):
        try:
            clog.info('Initial some ipr attrs and datasets for pipeline (TissueSeg, CellSeg, ManualState...)')
            with h5py.File(self._ipr_path, 'a') as conf:
                # if 'StereoResepVersion' not in conf['ImageInfo'].attrs.keys():
                #     conf['ImageInfo'].attrs['StereoResepVersion'] = PROG_VERSION
                if "IPRVersion" not in conf.keys():
                    conf.attrs['IPRVersion'] = '0.0.1'
                if "StereoResepSwitch" not in conf.keys():
                    group = conf.create_group('StereoResepSwitch')
                    group.attrs['stitch'] = True
                    group.attrs['tissueseg'] = True
                    group.attrs['cellseg'] = True
                    group.attrs['register'] = True
                if "ManualState" not in conf.keys():
                    group = conf.create_group('ManualState')
                    group.attrs['stitch'] = False
                    group.attrs['tissueseg'] = False
                    group.attrs['cellseg'] = False
                    group.attrs['register'] = False
                if 'TissueSeg' not in conf.keys():
                    conf.create_group('TissueSeg')
                if 'CellSeg' not in conf.keys():
                    conf.create_group('CellSeg')
        except Exception as e:
            clog.error(traceback.format_exc())

    def _stitching(self, p_type):
        clog.info('Stitching')
        # try:
        # TODO 替代io模块, ipr临时测通, 不涉及写入
        with h5py.File(self._ipr_path, 'r') as conf:
            research_stitch_module = conf['Research']
            location = research_stitch_module[f'{p_type}']['Stitch']['StitchFovLocation'][...]
            track_template = research_stitch_module[f'{p_type}']['Stitch']['GlobalTemplate'][...]
            # microscope_stitch = research_stitch_module.attrs['MicroscopeStitch']
            # if microscope_stitch == 0:
            #     h_jitter = research_stitch_module['HorizontalJitter'][...]
            #     v_jitter = research_stitch_module['VerticalJitter'][...]

            rows = conf[f'{p_type}']['ImageInfo'].attrs['ScanRows']
            cols = conf[f'{p_type}']['ImageInfo'].attrs['ScanCols']

        fov_stitched_path = os.path.join(self._output_path, self.stitch_img_name + f'_{p_type}' + self.img_ext)
        #########
        if not self.is_stitched:
            self.stitcher.set_size(rows, cols)
            self.stitcher.set_global_location(location)
            p_type_img_dir = self.img_paths[p_type]
            for i in os.listdir(p_type_img_dir):
                cur_path = os.path.join(p_type_img_dir, i)
                if os.path.isdir(cur_path):
                    img_path = cur_path
            img_paths = imagespath2dict(img_path)
            self.stitcher.stitch(src_fovs=img_paths, output_path=self._output_path)

            try:
                fft_detect_channel = self.stitch_config['fft_channel'].get(self._stain_type, 0)
            except Exception as e:
                fft_detect_channel = 0

            clog.info(f'Stitch module fft channel: {fft_detect_channel}')
            # TODO 得到行向和列向拼接偏移量评估
            # if microscope_stitch == 0:
            #     self.stitcher.set_jitter(h_jitter, v_jitter)
            #     x_jitter_eval, y_jitter_eval = self.stitcher._get_stitch_eval()
            # else:

            self.stitcher._get_jitter(
                src_fovs=img_paths,
                process=self.stitch_config['running_config']['num_threads'],
                fft_channel=fft_detect_channel
            )
            x_jitter_eval, y_jitter_eval = self.stitcher._get_jitter_eval()

            self._stitch_img = self.stitcher.get_image()
            Image.write_s(self._stitch_img, fov_stitched_path,
                          compression=True)

            with h5py.File(self._ipr_path, 'a') as conf:
                # 适配stio ipr，image studio ipr
                stitch_module = conf[f'{p_type}']['Stitch']
                stitch_module.attrs['StitchingScore'] = -1
                bgi_stitch_module_name = 'BGIStitch'
                if bgi_stitch_module_name not in stitch_module.keys():
                    stitch_module.require_group(bgi_stitch_module_name)
                bgi_stitch_module = stitch_module[bgi_stitch_module_name]
                try:
                    bgi_stitch_module.attrs['StitchedGlobalHeight'] = self._stitch_img.shape[0]
                    bgi_stitch_module.attrs['StitchedGlobalWidth'] = self._stitch_img.shape[1]
                    if 'StitchedGlobalLoc' in bgi_stitch_module.keys():
                        del bgi_stitch_module['StitchedGlobalLoc']
                except:
                    pass
                bgi_stitch_module.create_dataset('StitchedGlobalLoc', data=np.array(location))
                stitch_eval_module_name = 'StitchEval'
                if stitch_eval_module_name not in stitch_module.keys():
                    stitch_module.require_group(stitch_eval_module_name)
                stitch_eval_module = stitch_module[stitch_eval_module_name]
                if 'StitchEvalH' in stitch_eval_module.keys():
                    del stitch_eval_module['StitchEvalH']
                    del stitch_eval_module['StitchEvalV']
                stitch_eval_module.create_dataset('StitchEvalH', data=x_jitter_eval)
                stitch_eval_module.create_dataset('StitchEvalV', data=y_jitter_eval)
                conf['StereoResepSwitch'].attrs['stitch'] = False
        else:
            try:
                cur_image_path = self.img_paths[p_type]
                copyfile(cur_image_path, fov_stitched_path)  # TODO: 处理大图！！
            except Exception as e:
                clog.error(traceback.format_exc())

        np.savetxt(os.path.join(self._output_path, self.stitch_template + self.txt_ext), track_template)
        clog.info('Stitching Done')

        # except Exception as e:
        #     clog.error(traceback.format_exc())

    def calibration(self, dapi_img, if_img):
        """
        start calibration based on dapi image
        Args:
            dapi_img: np.ndarray, must be 2 dimension
            if_img: np.ndarray, must be 2 dimension

        Returns: dict, which including offset, scale, angle, dst_shape and confidence
        """
        self.calibrate = FFTRegister()
        dapi_img, if_img = self.calibrate.pad_same_image(dapi_img, if_img)
        result = self.calibrate.calibration(dapi_img, if_img, similarty=True)
        return if_img, result

    def do_calibration(self, ):
        for cur_p_type in self.protein_type:
            if cur_p_type == "DAPI":
                i = Image()
                stitch_img_path = os.path.join(self._output_path,
                                               self.stitch_img_name + f"_{cur_p_type}" + self.img_ext)
                i.read(image=stitch_img_path)
                dapi_stitch_image = i.image

        for cur_p_type in self.protein_type:
            if cur_p_type != "DAPI":
                i = Image()
                stitch_img_path = os.path.join(self._output_path,
                                               self.stitch_img_name + f"_{cur_p_type}" + self.img_ext)
                i.read(image=stitch_img_path)
                if_stitch_image = i.image
                clog.info(f"{cur_p_type} calibration")
                if_img, calibrate_result = self.calibration(dapi_stitch_image, if_stitch_image)
                clog.info(f"{cur_p_type} calibration calc finished")
                if_img = self.calibrate.transform_img_vips(if_img,
                                                           offset=calibrate_result["offset"],
                                                           scale=calibrate_result["scale"],
                                                           angle=calibrate_result["angle"],
                                                           dst_shape=calibrate_result["dst_shape"],
                                                           )
                tifffile.imwrite(stitch_img_path, if_img)

    def _registration(self, p_type):
        clog.info('Registration')
        # try:
        # TODO 替代io模块, ipr临时测通, 不涉及写入
        with h5py.File(self._ipr_path, 'r') as conf:
            scale_x = conf[f'{p_type}']['Register'].attrs['ScaleX']
            scale_y = conf[f'{p_type}']['Register'].attrs['ScaleY']
            rotate = conf[f'{p_type}']['Register'].attrs['Rotation']
            track_template = conf['Research'][f'{p_type}']['Stitch']['GlobalTemplate'][...]
        rgb_image = None  # grayscale image is None, rgb image is not None
        i = Image()
        stitch_img_path = os.path.join(self._output_path, self.stitch_img_name + f"_{p_type}" + self.img_ext)
        i.read(image=stitch_img_path)
        self._stitch_img = i.image

        if self._stitch_img.ndim == 3:
            rgb_image = self._stitch_img.copy()
            regist_channel = self.regist_config.get('channel', {}).get('HE', None)
            if regist_channel is None:
                if self._stain_type == 'HE':
                    clog.info(f"Stitch image is {self._stitch_img.ndim} channel, stain type is HE and regist "
                              f"channel is None, convert, using cell seg rgb2gray method")
                    self._stitch_img = f_rgb2gray(self._stitch_img, need_not=True)
                else:
                    clog.info(f"Stitch image is {self._stitch_img.ndim} channel and regist channel is None,"
                              f"convert using regular rgb2gray method")
                    self._stitch_img = f_rgb2gray(self._stitch_img)
            else:
                clog.info(f"Stitch image is {self._stitch_img.ndim} channel and regist channel is {regist_channel},"
                          f"using {regist_channel} as regist channel")
                self._stitch_img = self._stitch_img[:, :, regist_channel]
        else:
            clog.info(f"Stitch image is {self._stitch_img.ndim} channel, no need to convert")

        # _, _, x_start, y_start, gene_exp = gef2image(self._gem_path, self._output_path)
        ml = MatrixLoader(self._gem_path, self._output_path)
        gene_exp, self.x_start, self.y_start = ml.f_gene2img_pd()

        #########
        self.registrator.mass_registration_stitch(
            fov_stitched=self._stitch_img,
            vision_image=gene_exp,
            chip_template=self._stereo_chip,
            track_template=track_template,
            scale_x=1 / scale_x,
            scale_y=1 / scale_y,
            rotation=rotate,
            flip=True
        )
        vision_cp = self.registrator.vision_cp
        self.registrator.transform_to_regist()
        self._regist_img = self.registrator.regist_img  # TODO 配准图返回
        regist_img_copy = self._regist_img.copy()
        regist_score = self.registrator.register_score(regist_img_copy, gene_exp)  # 会改变第一个入参
        if rgb_image is not None:
            clog.info(f"Stitch image is rgb, generating rgb regist image")
            fov_transform = self.registrator.stitch_to_transform(
                fov_stitch=rgb_image,
                scale_x=scale_x,
                scale_y=scale_y,
                rotation=rotate,
            )
            self.registrator.fov_transformed = fov_transform
            self.registrator.transform_to_regist()
            self._regist_img = self.registrator.regist_img  # 配准图返回

        np.savetxt(os.path.join(self._output_path, self.gene_template + self.txt_ext), vision_cp)
        regist_path = os.path.join(self._output_path,
                                   f'{self.chip_name}_{self.regist_img_name}' + f"_{p_type}" + self.img_ext)
        Image.write_s(self._regist_img, regist_path, compression=True)
        Image.write_s(
            gene_exp,
            os.path.join(self._output_path, f'{self.chip_name}_{self.gene_img_name}' + self.img_ext),
            compression=True)

        ####
        # 适配流程
        h_, w_ = self.registrator.fov_transformed.shape[:2]
        json_serialize({'height': h_, 'width': w_}, os.path.join(self._output_path, 'attrs.json'))
        if w_ > h_:
            thumb = f_resize(f_ij_16_to_8(self.registrator.fov_transformed.copy()), (1500, int(h_ * 1500 / w_)))
        else:
            thumb = f_resize(f_ij_16_to_8(self.registrator.fov_transformed.copy()), (int(w_ * 1500 / h_), 1500))
        Image.write_s(thumb, os.path.join(self._output_path, 'transform_thumb.png'))

        # 组织最大外接矩阵
        bbox = [0, 0, self._regist_img.shape[1], self._regist_img.shape[0]]
        with open(os.path.join(self._output_path, f'{self.chip_name}_tissue_bbox.csv'), 'w') as f:
            f.write('left\tupper\tright\tlower\n')
            f.write('\t'.join(list(map(str, bbox))))
        ####

        Image.write_s(
            self.registrator.fov_transformed,
            os.path.join(self._output_path, self.stitch_transform_name + f"_{p_type}" + self.img_ext),
            compression=True)
        np.savetxt(os.path.join(self._output_path, self.transform_template + self.txt_ext),
                   self.registrator.adjusted_stitch_template_unflip)

        with h5py.File(self._ipr_path, 'a') as conf:
            if 'MatrixTemplate' in conf[f"{p_type}"]['Register'].keys():
                del conf[f"{p_type}"]['Register']['MatrixTemplate']
            if 'TransformTemplate' in conf[f"{p_type}"]['Stitch'].keys():
                del conf[f"{p_type}"]['Stitch/TransformTemplate']
            conf[f"{p_type}"]['Stitch'].create_dataset(
                'TransformTemplate',
                data=self.registrator.adjusted_stitch_template_unflip[:, :2]
            )
            conf[f"{p_type}"]['Register'].create_dataset('MatrixTemplate', data=vision_cp[:, :2], compression='gzip')
            conf[f"{p_type}"]['Register'].attrs['CounterRot90'] = self.registrator.rot90
            conf[f"{p_type}"]['Register'].attrs['OffsetX'] = self.registrator.offset[0]
            conf[f"{p_type}"].attrs['OffsetY'] = self.registrator.offset[1]
            conf[f"{p_type}"].attrs['RegisterScore'] = regist_score
            conf[f"{p_type}"].attrs['MatrixShape'] = gene_exp.shape
            conf[f"{p_type}"].attrs['XStart'] = self.x_start
            conf[f"{p_type}"].attrs['YStart'] = self.y_start
            conf[f"{p_type}"].attrs['Flip'] = True
            conf['StereoResepSwitch'].attrs['register'] = False
        for cur_p_type in self.protein_type:
            if cur_p_type != "DAPI":
                i = Image()
                stitch_img_path = os.path.join(self._output_path,
                                               self.stitch_img_name + f"_{cur_p_type}" + self.img_ext)
                i.read(image=stitch_img_path)
                stitch_img = i.image
                cur_fov_transform = self.registrator.stitch_to_transform(
                    stitch_img,
                    scale_x=1 / scale_x,
                    scale_y=1 / scale_y,
                    rotation=rotate,
                )
                self.registrator.fov_transformed = cur_fov_transform
                self.registrator.transform_to_regist()
                self._regist_img = self.registrator.regist_img  # 配准图返回
                regist_path = os.path.join(
                    self._output_path,
                    f'{self.chip_name}_{self.regist_img_name}' + f"_{cur_p_type}" + self.img_ext
                )
                Image.write_s(self._regist_img, regist_path, compression=True)
                Image.write_s(
                    self.registrator.fov_transformed,
                    os.path.join(self._output_path, self.stitch_transform_name + f"_{cur_p_type}" + self.img_ext),
                    compression=True)

        del self._stitch_img
        del gene_exp
        del self.registrator.fov_transformed
        del self._regist_img
        self._regist_img = None
        clog.info('Registration Done')

        # except Exception as e:
        #     clog.error(traceback.format_exc())

    def if_tissue_segmenatation(self, ):
        for cur_p_type in self.protein_type:
            if cur_p_type != "DAPI":
                regist_path = os.path.join(self._output_path,
                                           f'{self.chip_name}_{self.regist_img_name}' + f"_{cur_p_type}" + self.img_ext)
                tissue_intensity = TissuecutIntensity(regist_path, threshold=None)
                mask, score, threshold = tissue_intensity.tissue_infer_threshold()
                save_path = os.path.join(
                    self._output_path,
                    f'{self.chip_name}_{self.tissue_mask_name}' + f"_{cur_p_type}" + self.img_ext
                )
                Image.write_s(
                    mask,
                    save_path,
                    compression=True
                )

    def _tissue_segmentation(self, p_type='DAPI'):
        clog.info('Tissue Segmentation')
        try:
            regist_path = os.path.join(self._output_path,
                                       f'{self.chip_name}_{self.regist_img_name}' + f"_{p_type}" + self.img_ext)
            i = Image()
            i.read(image=regist_path)
            self._regist_img = i.image
            clog.info(f"Tissue seg using img: {regist_path}")
            mask = self.t_seg.run(self._regist_img)
            Image.write_s(
                mask,
                os.path.join(self._output_path, f'{self.chip_name}_{self.tissue_mask_name}' + self.img_ext),
                compression=True
            )
            with h5py.File(self._ipr_path, 'a') as conf:
                ff = conf[p_type]
                tissue_module = 'TissueSeg'
                if tissue_module not in ff.keys():
                    ff.require_group(tissue_module)
                ff[tissue_module].attrs['TissueSegShape'] = mask.shape
                conf['StereoResepSwitch'].attrs['tissueseg'] = False
            del mask
            clog.info('Tissue Done')
        except Exception as e:
            clog.error(traceback.format_exc())

    # def _clarity_eval(self):
    #     clog.info('Clarity Eval')
    #     try:
    #         if self._regist_img is None:
    #             regist_path = os.path.join(self._output_path, f'{self.chip_name}_{self.regist_img_name}' + self.img_ext)
    #             i = Image()
    #             i.read(image=regist_path)
    #             self._regist_img = i.image
    #         self.clarity_eval.set_enhance_func(clarity_enhance_method.get(self._stain_type, None))
    #         self.clarity_eval.run(
    #             img=self._regist_img,
    #         )
    #         self.clarity_eval.post_process()
    #         clarity_mask = self.clarity_eval.black_img
    #         Image.write_s(
    #             clarity_mask,
    #             os.path.join(self._output_path, f"{self.chip_name}_clarity_mask" + self.img_ext),
    #             compression=True
    #         )
    #         clarity_heatmap = self.clarity_eval.draw_img
    #         cv2.imwrite(
    #             os.path.join(self._output_path, f"{self.chip_name}_{self.clarity_eval_name}" + self.img_ext),
    #             clarity_heatmap,
    #         )
    #         clog.info('Clarity Eval Finished')
    #     except Exception as e:
    #         clog.error(traceback.format_exc())

    def _cell_segmentation(self, f_type):
        from stereocell_v2.cellpose.main import CellSegmentation
        gpu = True
        photo_size = self.c_config['cell_p']['photo_size']
        photo_step = self.c_config['cell_p']['photo_step']
        dmin = self.c_config['cell_p']['dmin']
        dmax = self.c_config['cell_p']['dmax']
        step = self.c_config['cell_p']['step']
        clog.info(f"Cell seg params: \n "
                  f"photo_size: {photo_size} \n"
                  f"photo_step: {photo_step} \n"
                  f"dmin: {dmin} \n"
                  f"dmax: {dmax} \n"
                  f"step: {step} \n")
        clog.info('Cell Segmentation')
        try:
            cell_mask_path = os.path.join(
                self._output_path,
                f'{self.chip_name}_{self.cell_mask_name}_{f_type}' + self.img_ext
            )
            if os.path.exists(cell_mask_path):
                clog.info('Cell Mask Exist, Skip')
            else:
                regist_path = os.path.join(self._output_path,
                                           f'{self.chip_name}_{self.regist_img_name}_{f_type}' + self.img_ext)
                clog.info(f"Cell segmentation is using {regist_path}")
                # i = Image()
                # i.read(image=regist_path)
                # self._regist_img = i.image
                # self.mask = self.c_seg.run(self._regist_img)
                # trace = self.c_seg.get_trace(self.mask)
                cell_seg = CellSegmentation(
                    regist_path, cell_mask_path,
                    gpu=gpu,
                    photo_size=photo_size,
                    photo_step=photo_step,
                    dmin=dmin,
                    dmax=dmax,
                    step=step
                )
                cell_seg.segment_cells()

                # Image.write_s(self.mask, cell_mask_path, compression=True)
                # with h5py.File(self._ipr_path, 'a') as conf:
                #     cell_seg = conf[f"{f_type}"].require_group('CellSeg')
                #     # cell_seg_module.attrs['CellMaskPath'] = cell_mask_path
                #     # cell_seg_module.create_dataset('CellSegTrace', data=np.array(trace), compression='gzip')
                #     cell_seg_module = 'CellSeg'
                #     if cell_seg_module not in conf.keys():
                #         conf.require_group(cell_seg_module)
                #     conf[f"{f_type}"][cell_seg_module].attrs['CellSegShape'] = self.mask.shape
                #     conf['StereoResepSwitch'].attrs['cellseg'] = False
            # del self.c_seg
            clog.info('Cell Segmentation Done')
        except Exception as e:
            clog.error(traceback.format_exc())

    def _cell_labeling(self, p_type='DAPI'):
        clog.info('Cell Labeling')
        try:
            correct_mask_path = os.path.join(
                self._output_path,
                f'{self.chip_name}_{self.cell_correct_mask_name}' + self.img_ext
            )
            if os.path.exists(correct_mask_path):
                clog.info(f"Correct Mask Exists, Skip")
            else:
                if self.mask is None:
                    cell_mask_path = os.path.join(
                        self._output_path,
                        f'{self.chip_name}_{self.cell_mask_name}_{p_type}' + self.img_ext
                    )
                    i = Image()
                    i.read(image=cell_mask_path)
                    mask = i.image
                else:
                    mask = self.mask
                ml = MatrixLoader(self._gem_path)
                if self._gem_path.lower().endswith(('.bgef', '.gef')):
                    new_file = os.path.join(self._output_path, f"{self.chip_name}_exp.gem")
                    ml.bgef2gem(bgef_file=self._gem_path, gem_file=new_file, binsize=1)
                    # gem_file = os.path.join(self._output_path, f"{self.chip_name}_exp.gem")

                    # obj = gefToGem(new_file, f"{self.chip_name}")
                    # obj.bgef2gem(self._gem_path, 1)
                else:
                    # gem gz
                    new_file = self._gem_path
                # 修正
                cl = CellLabelling(
                    mask,
                    new_file
                )
                cl.set_process(self.config_file["cell_correct"]['num_threads'])
                correct_mask, self.exp_matrix = cl.run_fast()

                # mask写出
                # correct_mask = cl.draw_corrected_mask(correct_mask)

                Image.write_s(correct_mask, correct_mask_path, compression=True)
        except Exception as e:
            clog.error(traceback.format_exc())

    def _iqc(self, ):
        clog.info('Image QC')
        pass

    def _check_qc_flag(self, ):

        with h5py.File(self._ipr_path) as conf:
            qc_flag = conf['DAPI']['QCInfo'].attrs['QCPassFlag']

        return True if qc_flag == 1 else False

    def save_preview(self):
        try:
            ir = Image()
            ir.read(glob(os.path.join(self._output_path, '*_transformed.tif'))[0])
            img = ir.image
            if img.dtype != 'uint8':
                img = f_ij_16_to_8(img)
            irt = Image()
            irt.read(glob(os.path.join(self._output_path, '*_tissue_cut.tif'))[0])
            tissue = irt.image
            if tissue.dtype != 'uint8':
                tissue = f_ij_16_to_8(tissue)
            tissue[tissue > 0] = 255
            img = f_resize(img)
            h, w = img.shape[:2]
            mask = f_resize(tissue, (w, h))
            if img.ndim == 2:
                img = f_gray2bgr(img)
            contours, hierachy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(img, contours, -1, (255, 255, 0), 1)
            img = f_ij_auto_contrast(img)
            with h5py.File(self._ipr_path, 'r+') as conf:
                if 'Preview' in conf.keys():
                    del conf['Preview']
                conf.create_dataset('Preview', data=img, compression='gzip')
            clog.info('Success to create preview')
        except Exception as e:
            clog.error(traceback.format_exc())

    def cell_mask_post_process(self, p_type='DAPI'):
        """
        Based on cell mask and cell correct mask
        Generate
        - filtered cell mask
        - filtered cell correct mask
        """
        try:
            clog.info("Filtering cell mask and correct mask based on tissue mask")
            cell_mask_path = os.path.join(
                self._output_path,
                f'{self.chip_name}_{self.cell_mask_name}_{p_type}' + self.img_ext
            )
            # correct_mask_path = os.path.join(
            #     self._output_path,
            #     f'{self.chip_name}_{self.cell_correct_mask_name}' + self.img_ext
            # )
            tissue_mask_path = os.path.join(
                self._output_path,
                f'{self.chip_name}_{self.tissue_mask_name}' + self.img_ext
            )
            i = Image()
            i.read(cell_mask_path)
            cell_mask = i.image
            # i.read(correct_mask_path)
            # correct_mask = i.image
            i.read(tissue_mask_path)
            tissue_mask = i.image
            filtered_cell_mask = cell_mask * tissue_mask
            # filtered_cell_correct_mask = correct_mask * tissue_mask
            i.write_s(
                image=filtered_cell_mask,
                output_path=os.path.join(
                    self._output_path,
                    f'{self.chip_name}_{self.filtered_cell_mask_name}_{p_type}' + self.img_ext
                ),
                compression=True
            )
            # i.write_s(
            #     image=filtered_cell_correct_mask,
            #     output_path=os.path.join(
            #         self._output_path,
            #         f'{self.chip_name}_{self.filtered_cell_correct_mask_name}' + self.img_ext
            #     ),
            #     compression=True
            # )
            clog.info("Finished filtering")
        except Exception as e:
            clog.error(traceback.format_exc())

    def gem_file_post_process(self, f_type):
        """
        Based on these four mask and input gene matrix (gem file)
        Generate corresponding four gem file
        """
        # try:
        clog.info("Generating new gene files based on different masks")
        masks = [
            self.cell_mask_name + f'_{f_type}',
            self.filtered_cell_mask_name + f'_{f_type}',
        ]
        clog.info(f"Gem file post process using masks: {masks}")
        if self._gem_path.lower().endswith(('.bgef', '.gef')):
            gem_path = os.path.join(self._output_path, f"{self.chip_name}_exp.gem")
            if not os.path.exists(gem_path):
                ml = MatrixLoader(self._gem_path)
                ml.bgef2gem(bgef_file=self._gem_path, gem_file=gem_path, binsize=1)
        else:
            gem_path = self._gem_path

        mask_paths = []
        for mask_name in masks:
            cur_mask_path = os.path.join(
                self._output_path,
                f'{self.chip_name}_{mask_name}' + self.img_ext
            )
            mask_paths.append(cur_mask_path)
        generate_gem(
            mask_paths=mask_paths,
            gem_path=gem_path,
            out_path=self._output_path
        )
        clog.info("Finished generating new gene files")
        # except Exception as e:
        #     clog.error(str(e))

    def run_x_bin(self, gem_path, input_path, ipr_path, output=None):
        clog.info(f"Using version {self.version}")
        self._gem_path = gem_path
        self._input_path = input_path
        self._output_path = os.path.join(output, "registration")
        os.makedirs(self._output_path, exist_ok=True)
        clog.info(f"Pipeline output dir {self._output_path}")
        ipr_name = os.path.split(ipr_path)[-1]
        self._ipr_path = os.path.join(self._output_path, ipr_name)
        copyfile(ipr_path, self._ipr_path)
        if self._check_qc_flag():

            clog.info(f"Pipeline using ipr {self._ipr_path}")
            self.load_initial_info()  # 初始化多个模块需要用到的信息from ipr，主要是读取
            self.initial_ipr_attrs()  # 初始化pipeline需要用到的一些group，如果不用stio的话是需要pipeline自己生成的
            self.prepare_input()  # 解压缩qc的targz，找到图片路径
            clog.info('Start pipline')

            if self.config_file["operation"]["Stitching"]:
                for p_type in self.protein_type:
                    self._stitching(p_type)
                self.do_calibration()
            if self.config_file["operation"]["Register"]:
                self._registration('DAPI')
            # 初始化分割模型
            if self.config_file["operation"]["Tissue_Segment"]:
                auto_download_weights(self.zoo_dir, [self.t_config[self._stain_type]])
                self.initialize_t_seg()
                self._tissue_segmentation()
                # auto_download_weights(self.zoo_dir, [self.clarity_config[self._stain_type]])
                # self.initialize_clarity_eval()
                # self._clarity_eval()
                # self._get_tissue_seg_score()
                # self.save_preview()
                # self.if_tissue_segmenatation()
            if self.config_file["operation"]["Cell_Segment"]:
                auto_download_weights(self.zoo_dir, [self.c_config[self._cell_seg_type][self._stain_type]])
                # self.initialize_c_seg()
                for p_type in self.protein_type:
                    if p_type == 'DAPI':
                        continue
                    self._cell_segmentation(p_type)

            # if self.config_file["operation"]["Tissue_Segment"] and self.config_file["operation"]["Cell_Segment"]:
            #     self.regist_to_rpi()
            # if self.config_file["operation"]["Cell_Correct"]:
            #     self._cell_labeling()
            if self.config_file["operation"]["Cell_Segment"] and self.version.upper() == 'SAP':
                for p_type in self.protein_type:
                    if p_type == 'DAPI':
                        continue
                    self.cell_mask_post_process(p_type)
                    self.gem_file_post_process(p_type)

        else:
            clog.info('QC failed, skip pipeline.')

    def encode_mask(self, mask):
        encode_mask = self.bmr.encode(mask)
        return encode_mask

    def _get_tissue_seg_score(self):
        try:
            irt = Image()
            irt.read(os.path.join(self._output_path, f'{self.chip_name}_tissue_cut.tif'))
            tissue_mask = irt.image
            irc = Image()
            irc.read(os.path.join(self._output_path, f"{self.chip_name}_clarity_mask" + self.img_ext))
            clarity_mask = irc.image
            iou_result = iou(clarity_mask, tissue_mask)
            iou_result = round(iou_result, 3) * 100
            clog.info(f"Tissue seg score {iou_result}")
            with h5py.File(self._ipr_path, 'a') as conf:
                tissue_module = 'TissueSeg'
                if tissue_module not in conf.keys():
                    conf.require_group(tissue_module)
                conf[tissue_module].attrs['TissueSegScore'] = iou_result
            clog.info("Finish calculating tissue seg score")
        except Exception as e:
            clog.error(traceback.format_exc())

    def regist_to_rpi(self):
        clog.info("Saving tissue/cell mask to ipr, generate regist rpi")
        # need regist, tissue mask, cell mask
        irt = Image()
        irt.read(os.path.join(self._output_path, f'{self.chip_name}_tissue_cut.tif'))
        tissue_mask = irt.image
        irc = Image()
        irc.read(os.path.join(self._output_path, f'{self.chip_name}_mask.tif'))
        cell_mask = irc.image
        irg = Image()
        irg.read(os.path.join(self._output_path, f'{self.chip_name}_regist.tif'))
        regist_tif = irg.image
        with h5py.File(self._ipr_path, 'r+') as f:
            if 'CellMask' in f['CellSeg'].keys():
                del f['CellSeg']['CellMask']
            if 'TissueMask' in f['TissueSeg'].keys():
                del f['TissueSeg']['TissueMask']
            f['TissueSeg'].create_dataset('TissueMask', data=self.encode_mask(tissue_mask), compression='gzip')
            f['CellSeg'].create_dataset('CellMask', data=self.encode_mask(cell_mask), compression='gzip')
        with h5py.File(self._ipr_path, 'r') as f:
            x_start = f['Register'].attrs['XStart']
            y_start = f['Register'].attrs['YStart']
        result = dict()
        result['ssDNA'] = regist_tif
        result['TissueMask'] = tissue_mask
        result['CellMask'] = cell_mask
        rpi_path = os.path.join(self._output_path, f'{self.chip_name}.rpi')
        createPyramid(result, rpi_path, x_start=x_start, y_start=y_start)
        clog.info("Finished regist to rpi")

    @staticmethod
    def get_time():
        dt = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        return dt


def run_pipeline(input_, output, ipr_path, matrix, cell_type, config, zoo_dir, version='SAP'):
    clog.info("--------------------------------start--------------------------------")
    image_root = input_
    output = output
    ipr_path = ipr_path
    gem_file = matrix
    cell_type = cell_type.upper()

    p = Pipeline()
    p.set_cell_seg_type(cell_type)
    p.set_config_file(config)
    p.set_zoo_dir(zoo_dir)
    p.set_version(version)

    p.run_x_bin(gem_path=gem_file, input_path=image_root, ipr_path=ipr_path, output=output)
    clog.info("--------------------------------end--------------------------------")


def main(args, para):
    clog.log2file(
        out_dir=args.output,
    )
    clog.set_level()

    try:
        clog.info(f"Cellbin Version {version('cell-bin')}")
    except Exception:
        clog.error(traceback.format_exc())

    clog.info(args)
    try:
        run_pipeline(
            input_=args.input,
            output=args.output,
            ipr_path=args.ipr_path,
            matrix=args.matrix,
            cell_type=args.cell_type.upper(),
            config=args.config,
            zoo_dir=args.zoo_dir,
            version=args.version
        )
    except Exception:
        clog.error(traceback.format_exc())


def arg_parser():
    usage = '''
    python pipeline.py -i /media/Data/dzh/neq_qc_test_data/SS200000464BL_C4/images/C4 
    -m /media/Data/dzh/neq_qc_test_data/SS200000464BL_C4/SS200000464BL_C4.raw.gef 
    -r /media/Data/dzh/neq_qc_test_data/SS200000464BL_C4/SS200000464BL_C4_20230403_125857_0.1.ipr 
    -o /media/Data/dzh/neq_qc_test_data/SS200000464BL_C4/test_out 
    -cf /home/dengzhonghan/Desktop/code/cellbin/test/moudule/pipeline_config.json 
    -z /media/Data/dzh/neq_qc_test_data/SS200000464BL_C4/weights
    '''

    clog.info(PROG_VERSION)
    parser = argparse.ArgumentParser(usage=usage)
    # Must have
    parser.add_argument("--version", action="version", version=PROG_VERSION)
    parser.add_argument("-i", "--input", action="store", dest="input", type=str, required=True,
                        help="Tar file / Input image dir.")
    parser.add_argument("-m", "--matrix", action="store", dest="matrix", type=str, required=True,
                        help="Input gene matrix.")
    parser.add_argument("-r", "--ipr_path", action="store", dest="ipr_path", type=str, required=True, help="Ipr path.")
    parser.add_argument("-o", "--output", action="store", dest="output", type=str, required=True,
                        help="Result output dir.")

    # Optional
    parser.add_argument("-cf", "--config", action="store", dest="config", type=str, default=JIQUN_CF,
                        help="Config file (Json)")
    parser.add_argument("-z", "--zoo", action="store", dest="zoo_dir", type=str, default=JIQUN_ZOO,
                        help="DNN weights dir")

    parser.add_argument("-ct", "--cell_seg_module", action="store", dest="cell_type", type=str, default='CELL',
                        help="Cell Seg type")
    parser.add_argument("-v", "--prog_version", action="store", dest="version", type=str, default=PROG_VERSION,
                        help="SAP or UAT")

    parser.set_defaults(func=main)
    (para, args) = parser.parse_known_args()
    para.func(para, args)


if __name__ == '__main__':
    arg_parser()

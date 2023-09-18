###################################################
"""reference template for image, must need QC data.
create by lizepeng, 2023/1/31 10:00
"""
####################################################

import copy
import numpy as np
import os
import math
import cv2 as cv
import tifffile
import sys
sys.setrecursionlimit(20000) #递归调用堆栈长度

from scipy.spatial import ConvexHull
from cellbin.utils import clog

class TemplateReference:
    def __init__(self, ):
        #input
        self.chip_no: list = list()
        self.scale_x: float = 1.0
        self.scale_y: float = 1.0
        self.rotation: float = 0.0
        self.qc_pts: dict = {}
        self.template_qc_pts: dict = {}

        #
        self.x_intercept = None
        self.y_intercept = None
        self.fov_loc_array = None
        self.template_center_pt: list = []
        self.template_qc: list = []
        self.mosaic_height = None
        self.mosaic_width = None
        self.flag_skip_reference = False

        self._correct_thresh = 20
        self._pair_thresh = 10
        self._qc_thresh = 5
        self._range_thresh = 5000
        self._cluster_num = 10
        self.flag_skip_reference = False

        # output
        self.template: list = list()
        self.template_src = None

    ################
    '''init parm'''
    ################
    def set_scale(self, scale_x: float, scale_y: float):
        self.scale_x = self._to_digit(scale_x)
        self.scale_y = self._to_digit(scale_y)
        assert self.scale_x is not None and self.scale_y is not None, "Input is not a number."

    def set_rotate(self, r: float):
        self.rotation = self._to_digit(r)
        assert self.rotation is not None, "Input is not a number."

    def set_threshold(self,
                      pair_thresh=None,
                      qc_thresh=None,
                      range_thresh=None,
                      correct_thresh=None,
                      cluster_num=None):
        """阈值参数定义"""
        if pair_thresh is not None:
            self._pair_thresh = pair_thresh
        if qc_thresh is not None:
            self._qc_thresh = qc_thresh
        if range_thresh is not None:
            self._range_thresh = range_thresh
        if correct_thresh is not None:
            self._correct_thresh = correct_thresh
        if cluster_num is not None:
            self._cluster_num = cluster_num

    def set_chipno(self, chip_no):
        '''
        :param chip_no: 芯片标准周期
        :return:
        '''
        assert type(chip_no) == list or type(chip_no) == np.ndarray, "ChipNO must be a list or array."
        self.chip_no = chip_no

    def set_fov_location(self, global_loc):
        '''
        :param global_loc: 全局拼接坐标
        :return:
        '''
        self.fov_loc_array = global_loc
        self.mosaic_width, self.mosaic_height = self.fov_loc_array[-1, -1] + self.fov_loc_array[1, 1]

    def set_qc_points(self, index, pts):
        """
        index: template FOV , row_col or [row, col]
        pts: {index: [x, y, ind_x, ind_y], ...}
        """
        if self.fov_loc_array is None:
            clog.info("Please init global location.")
            return

        if isinstance(index, str):
            row, col = index.split("_")
        elif isinstance(index, list):
            row, col = index
        else:
            clog.info("FOV index error.")
            return

        row = int(row)
        col = int(col)

        assert isinstance(pts, dict), "QC Points is error."
        for ind in pts.keys():
            points = np.array(pts[ind])
            if len(points) > 0:
                _row, _col = [int(i) for i in ind.split('_')]
                if _row == row and _col == col:
                    self.template_qc_pts[ind] = points
                    index = ind
                else:
                    self.qc_pts[ind] = points

        self.template_center_pt = copy.deepcopy(self.template_qc_pts[index][0])
        self.template_center_pt[:2] += self.fov_loc_array[row, col]

    ################
    '''reference'''
    ################
    def _delete_outline_points(self, points_re, points_qc, range_size=5000):
        '''
        离群点删除
        :param points_re:
        :param points_qc:
        :param range_size: 框选尺寸
        :return:
        '''
        _points_qc = list()
        _points_re = list()
        for k, point in enumerate(points_qc):
            if np.abs(point[0] - self.template_center_pt[0]) <= range_size and \
                    np.abs(point[1] - self.template_center_pt[1]) <= range_size:
                _points_qc.append(point)
                _points_re.append(points_re[k])

        return np.array(_points_re), np.array(_points_qc)

    def _check_parm(self):
        assert self.scale_x is not None and self.scale_y is not None, "Scale is need init."
        assert self.rotation is not None, "Rotate is need init."
        assert self.chip_no is not None and len(self.chip_no) != 0, "ChipNO is need init."
        # assert len(self.qc_pts) != 0 and len(self.template_qc_pts) != 0, "QC points is need init."

    def _global_qc_points_to_global(self):
        '''
        当QC点是全局点时使用 一般不用！！！
        '''
        points_list = [self.qc_pts]
        for type_points in points_list:
            for fov_name in type_points.keys():
                row, col = [int(i) for i in fov_name.split("_")]
                for point in type_points[fov_name]:
                    temp = copy.deepcopy(point)
                    self.template_qc.append(temp[:2])
                    break

    def _qc_points_to_gloabal(self, all_points=False):
        '''QC点映射到全局坐标'''
        self.template_qc = list()
        points_list = [self.qc_pts]
        for type_points in points_list:
            for fov_name in type_points.keys():
                row, col = [int(i) for i in fov_name.split("_")]
                if type_points[fov_name].ndim == 1:
                    temp = copy.deepcopy(type_points[fov_name])
                    temp[:2] += self.fov_loc_array[row, col]
                    self.template_qc.append(temp[:2])
                else:
                    for point in type_points[fov_name]:
                        temp = copy.deepcopy(point)
                        try:
                            temp[:2] += self.fov_loc_array[row, col]
                            self.template_qc.append(temp[:2])
                            if not all_points:
                                break
                        except:
                            break
    @staticmethod
    def pair_to_template(temp_qc, temp_re, threshold=10): #TODO 临时变量 非常重要！！！！！！！！！！！！！！！！！！！
        '''one point of temp0 map to only one point of temp1'''
        import scipy.spatial as spt

        temp_src = np.array(temp_re)[:, :2]
        temp_dst = np.array(temp_qc)[:, :2]
        tree = spt.cKDTree(data=temp_src)
        distance, index = tree.query(temp_dst, k=1)

        thr_index = index[distance < threshold]
        points_qc = temp_dst[distance < threshold]
        points_re = np.array(temp_re)[thr_index]

        return [points_re, points_qc]

    @staticmethod
    def resove_affine_matrix(H):
        theta = (math.degrees(math.atan2(H[1, 0], H[0, 0])) + math.degrees(math.atan2(H[1, 1], H[0, 1])) - 90) / 2
        s_x = math.sqrt(H[0, 0] ** 2 + H[1, 0] ** 2)
        s_y = (H[0, 0] * H[1, 1] - H[1, 0] * H[0, 1]) / s_x
        confidence = (H[0, 0] * H[0, 1] + H[1, 0] * H[1, 1]) / s_x
        clog.info("result: \ntheta: {}, \ns_x: {}, \ns_y: {}, \nconfidence:{}".format(theta, s_x, s_y, 1 - abs(confidence)))
        return theta, s_x, s_y

    def _mean_to_scale_and_rotate(self, points_re, points_qc):
        '''求模板点和QC点 scale和rotate 差异的均值'''
        scale_x_list = []
        scale_y_list = []
        rotate_list = []
        for point_re, point_qc in zip(points_re, points_qc):

            if point_qc[0] != self.template_center_pt[0]:

                #旋转角
                rotation_dst = math.degrees(
                    math.atan((point_qc[1] - self.template_center_pt[1]) / (point_qc[0] - self.template_center_pt[0])))
                rotation_src = math.degrees(
                    math.atan((point_re[1] - self.template_center_pt[1]) / (point_re[0] - self.template_center_pt[0])))

                _rotate = self.rotation + (rotation_dst - rotation_src)

                src_x = point_re[0] - self.template_center_pt[0]
                src_y = point_re[1] - self.template_center_pt[1]

                dst_x = point_qc[0] - self.template_center_pt[0]
                dst_y = point_qc[1] - self.template_center_pt[1]

                dis = math.sqrt(src_x ** 2 + src_y ** 2)

                _src_x = dis * math.cos(math.radians(np.abs(rotation_dst)))
                _src_y = dis * math.sin(math.radians(np.abs(rotation_dst)))

                _scale_x = self.scale_x / np.abs(_src_x / dst_x)
                _scale_y = self.scale_y / np.abs(_src_y / dst_y)

                scale_x_list.append(_scale_x)
                scale_y_list.append(_scale_y)
                rotate_list.append(_rotate)

        return np.mean(scale_x_list), np.mean(scale_y_list), np.mean(rotate_list)

    @staticmethod
    def _leastsq_to_scale_and_rotate(point_re, point_qc):
        '''最小化模板点和QC点距离 并求解出结果'''
        # point_re = np.array([[61.237, 35.355], [-35.355, 61.237], [-61.237, -35.355], [35.355, -61.237]])
        # point_qc = np.array([[100, 100], [-100, 100], [-100, -100], [100, -100]])
        from scipy.optimize import leastsq, minimize

        for k, point in enumerate(point_re):
            if point[0] == 0 and point[1] == 0:
                point_re = np.delete(point_re, k, axis=0)
                point_qc = np.delete(point_qc, k, axis=0)
                break

        def _error(p, point_re, point_qc):
            _scale_x, _scale_y, _rotate = p

            src_x = point_re[:, 0]
            src_y = point_re[:, 1]

            _t = (src_y) / (src_x + 0.000000000000001)
            _d = [math.atan(i) for i in _t]
            rotation_src = np.array([math.degrees(i) for i in _d])

            src_x = point_re[:, 0] * (1 + _scale_x)
            src_y = point_re[:, 1] * (1 + _scale_y)

            dis = [math.sqrt(i) for i in src_x ** 2 + src_y ** 2]

            dst_x = dis * np.array([math.cos(math.radians(np.abs(i))) for i in (rotation_src + _rotate)])
            dst_y = dis * np.array([math.sin(math.radians(np.abs(i))) for i in (rotation_src + _rotate)])

            dst_x = [-i if point_re[k, 0] < 0 else i for k, i in
                     enumerate(dst_x)]
            dst_y = [-i if point_re[k, 1] < 0 else i for k, i in
                     enumerate(dst_y)]

            error = (point_qc[:, 0] - dst_x) ** 2 + (point_qc[:, 1] - dst_y) ** 2
            error = [math.sqrt(i) for i in error]
            # print(np.sum(error))
            return np.sum(error) * 0 + max(error) * 1

        para = minimize(_error, x0=np.zeros(3, ), args=(point_re, point_qc))

        return para

    def _caculate_scale_and_rotate(self, points_re, points_qc, mode='minimize', update=True, center=None):
        '''
        使用模板点和QC点计算出scale和rotate
        '''
        if points_re.shape[1] == 4:
            points_re = points_re[:, :2]

        if center is None:
            center = self.template_center_pt

        points_re[:, 0] -= center[0]
        points_re[:, 1] = center[1] - points_re[:, 1]

        points_qc[:, 0] -= center[0]
        points_qc[:, 1] = center[1] - points_qc[:, 1]

        if mode == 'minimize':
            # 最小值距离优化法
            para = self._leastsq_to_scale_and_rotate(points_re, points_qc)
            _scale_x, _scale_y, _rotate = para.x
            if update:
                self.rotation -= _rotate
                self.scale_x *= (1 + _scale_x)
                self.scale_y *= (1 + _scale_y)
        elif mode == 'mean':
            # 均值法
            scale_x, scale_y, rotate = self._mean_to_scale_and_rotate(points_re, points_qc)
            if update:
                self.rotation = rotate
                self.scale_x = scale_x
                self.scale_y = scale_y
        elif mode == 'homo':
            # 单应性法
            matrix, mask = cv.estimateAffinePartial2D(points_re, points_qc)
            # matrix, mask = cv.findHomography(points_re, points_qc, 0)
            theta, s_x, s_y = self.resove_affine_matrix(matrix)
            if update:
                self.rotation += theta
                self.scale_x = self.scale_x * s_x
                self.scale_y = self.scale_y * s_y

    def first_template_correct(self, target_points, index, max_item=3, center_points=None):
        """
        Args:
            target_points: 模板FOV的 QC检点（非track线检点）
        """

        if target_points is None:
            return

        self._check_parm()
        if center_points is None:
            for k, v in self.template_qc_pts.items():
                source_points = v
                break
            center_points = source_points[0, :]

        for item in range(max_item):
            self._point_inference(center_points, (self._range_thresh, self._range_thresh))
            points_re, points_qc = self.pair_to_template(target_points, self.template, self._correct_thresh)
            self._caculate_scale_and_rotate(points_re, points_qc, center=center_points)
            self._point_inference(center_points, (self._range_thresh, self._range_thresh))
            points_re, points_qc = self.pair_to_template(target_points, self.template, self._pair_thresh)

            _dis = np.Inf
            for i in range(len(points_re)):
                dis = np.sqrt((points_re[i][0] - points_qc[i][0]) ** 2 +
                              (points_re[i][1] - points_qc[i][1]) ** 2)

                if dis < _dis:
                    _dis = dis
                    center_points = points_re[i]

        row, col = [int(i) for i in index.split('_')]
        center_points[:2] += self.fov_loc_array[row, col]
        self.template_center_pt = center_points

    def reference_template(self, mode='single'):
        '''
        mode: have three type ['single', 'double', 'multi']
        *   single: only reference template FOV
        *   double: reference template FOV & minimize the points distance
        *   multi: reference template FOV & minimize the points distance & change the template center point
        '''
        self._check_parm()
        self._point_inference(self.template_center_pt, (self.mosaic_height, self.mosaic_width))
        max_item = int(max(self.mosaic_height, self.mosaic_width) / self._range_thresh)

        if mode != 'single':
            self._qc_points_to_gloabal()
            # self._global_qc_points_to_global()
            points_qc = np.zeros([0, 2])
            count = 0  # 循环次数
            while count < max_item:
                points_re, points_qc = self.pair_to_template(self.template_qc, self.template, self._pair_thresh)

                points_re, points_qc = self._delete_outline_points(points_re, points_qc,
                                                                    self._range_thresh * (count + 1))
                if len(points_re) == 0:
                    self.flag_skip_reference = True
                    clog.info("Can't reference template, skip this mode.")
                    return
                self._caculate_scale_and_rotate(points_re, points_qc)
                self._point_inference(self.template_center_pt, (self.mosaic_height, self.mosaic_width))
                count += 1

            _, _, _, _, first_recall = self.get_template_eval(area_eval=False)
            double_info = {'dis': first_recall, 'point': self.template_center_pt,
                           'scale_x': self.scale_x, 'scale_y': self.scale_y,
                           'rotate': self.rotation}

            if mode == 'multi':
                _points_qc = np.concatenate((points_qc, points_re[:, 2:]), axis=1)
                candidate_points = self._find_center_close_point(_points_qc, reverse=True)
                candidate_info = list()

                for center_point in candidate_points:
                    self._qc_points_to_gloabal()
                    self._point_inference(center_point, (self.mosaic_height, self.mosaic_width))
                    points_re, points_qc = self.pair_to_template(self.template_qc, self.template, self._pair_thresh) #后续优化距离时放宽阈值 效果更好

                    try: #可能存在换模板中心点时到错误的位置上
                        self.template_center_pt = center_point
                        self._caculate_scale_and_rotate(points_re, points_qc, update=True)
                        self._point_inference(self.template_center_pt, (self.mosaic_height, self.mosaic_width))
                        _, _, _, _, recall = self.get_template_eval(area_eval=False)
                        candidate_info.append({'dis': recall, 'point': center_point,
                                               'scale_x': self.scale_x, 'scale_y': self.scale_y,
                                               'rotate': self.rotation})
                    except:
                        pass

                candidate_info = sorted(candidate_info, key=lambda x: x['dis'], reverse=True)
                if len(candidate_info) > 0:
                    min_info = candidate_info[0]

                    if min_info['dis'] > first_recall:
                        result_info = min_info
                    else: result_info = double_info

                    self.scale_x = result_info['scale_x']
                    self.scale_y = result_info['scale_y']
                    self.rotation = result_info['rotate']
                    self._point_inference(result_info['point'], (self.mosaic_height, self.mosaic_width))

        clog.info("Reference template done!")

    def get_template_eval(self, area_eval=True):
        '''
        :return:  max(dis), mean(dis) 获得此时模板与QC点的最大值与均值
        '''
        if self.flag_skip_reference:
            # TODO 参数均返回-1
            return -1, -1, -1, -1, -1
        self._qc_points_to_gloabal(all_points=True)
        self._outline_template()
        points_re, points_qc = self.pair_to_template(self.template_qc, self.template, self._pair_thresh)
        distances = list()
        for point_re, point_qc in zip(points_re, points_qc):
            dis = np.sqrt((point_re[0] - point_qc[0]) ** 2 + (point_re[1] - point_qc[1]) ** 2)
            distances.append(dis)

        _points_re, _points_qc = self.pair_to_template(self.template_qc, self.template, self._qc_thresh)

        area_rate = -1
        if area_eval:
            try:
                area_rate = self.points_area_eval(_points_qc, self._cluster_num)
            except: pass

        return area_rate, np.mean(distances), np.std(distances), \
               len(_points_re) / len(self.template_qc), \
               len(_points_re) / len(self.template)

    def points_area_eval(self, points, min_num=10):
        """
        用来计算点簇集合
        Args:
            points:
            min_num: 最小点簇的数量
        """
        adjacency_list = list()
        max_dis = max(self.scale_x, self.scale_y) * np.max(self.chip_no)
        dis_matrix = np.linalg.norm(points - points[:, None], axis=-1)
        for index in range(len(points)):
            select_index = (dis_matrix[index, :] < max_dis) & (dis_matrix[index, :] != 0)
            select_index = np.where(select_index == True)[0]
            adjacency_list.append(select_index)

        cluster_list = self._cluster_points(adjacency_list)
        cluster_list = [i for i in cluster_list if len(i) > min_num]
        index_list = [i for k in cluster_list for i in k]

        cluster_points = points[index_list]
        rate = self._max_points_area(cluster_points, [self.mosaic_height, self.mosaic_width])
        return rate

    @staticmethod
    def _max_points_area(points, shape):
        h, w = shape
        hull = ConvexHull(points)
        return hull.volume / (h * w)

    @staticmethod
    def _cluster_points(adjacency_list: list = None):
        """
        邻接表进行聚类
        """
        points_all_index_list = list(range(len(adjacency_list)))
        cluster_list = list()

        def recurrence(temp_list, points_index_set, adjacency_list, k):
            """递归找相邻对"""
            k += 1
            if len(temp_list) == 0:
                return points_index_set, k
            for index in temp_list:
                if index not in points_index_set:
                    adj = adjacency_list[index]
                    points_index_set.add(index)
                    src_list = []
                    for temp in adj:
                        if temp not in points_index_set:
                            src_list.append(temp)
                    points_index_set, k = recurrence(src_list, points_index_set, adjacency_list, k)
            return points_index_set, k

        while len(points_all_index_list) > 0:
            points_index_set = set()
            temp_list = [points_all_index_list[0]]
            points_index_set, k = recurrence(temp_list, points_index_set, adjacency_list, 0)
            cluster_list.append(points_index_set)
            for i in points_index_set:
                points_all_index_list.remove(i)

        return cluster_list

    def get_global_eval(self):
        """
        :return: 全局模板的误差矩阵 用于展示
        """
        import scipy.spatial as spt

        if self.flag_skip_reference:
            # TODO 参数返回-1
            return -1

        self._qc_points_to_gloabal(all_points=True)
        self._outline_template()
        points_re, points_qc = self.pair_to_template(self.template_qc, self.template, 50) #TODO 用于每个QC点都匹配上一个模板点

        rows, cols = self.fov_loc_array.shape[:2]
        loc_points = self.fov_loc_array.reshape([rows * cols, 2])

        temp_src = np.array(loc_points)[:, :2]
        temp_dst = np.array(points_qc)[:, :2]
        tree = spt.cKDTree(data=temp_src)
        distance, index = tree.query(temp_dst, k=1)

        result = dict()
        mat = np.zeros([rows, cols]) - 1

        for k, loc in enumerate(index):
            x, y = loc_points[loc]
            row, col = [[row, col] for row in range(rows) for col in range(cols) \
                        if self.fov_loc_array[row, col, 0] == x and self.fov_loc_array[row, col, 1] == y][0]

            _x, _y = temp_dst[k]
            _col = col - 1 if _x < x else col
            _row = row - 1 if _y < y else row
            _dis = np.sqrt((points_re[k, 0] - points_qc[k, 0]) ** 2 + (points_re[k, 1] - points_qc[k, 1]) ** 2)

            if f'{_row}_{_col}' in result.keys():
                result[f'{_row}_{_col}'].append(_dis)
            else: result[f'{_row}_{_col}'] = [_dis]

        for k, v in result.items():
            row, col = [int(i) for i in k.split('_')]
            mat[row, col] = np.mean(v)

        return mat

    def _outline_template(self):
        temp_template = list()
        for temp in self.template:
            if temp[0] >= 0 and temp[1] >= 0\
                and temp[0] <= self.mosaic_width and temp[1] <= self.mosaic_height:
                temp_template.append(temp)

        self.template = temp_template

    def _find_center_close_point(self, points, n=5, reverse=True):
        '''
        找寻距离全图中心点最近的已推导模板点
        :param points: [x, y, ind_x, ind_y]
        :return: self.template_center_pt
        '''
        points[:, 0] = self.template_center_pt[0] + points[:, 0]
        points[:, 1] = self.template_center_pt[1] - points[:, 1]

        center_point_x = self.mosaic_width / 2
        center_point_y = self.mosaic_height / 2

        dis_list = list()
        for point in points:
            dis = np.sqrt((point[0] - center_point_x) ** 2 + (point[1] - center_point_y) ** 2)
            dis_list.append(dis)

        if reverse:
            min_points = np.array(dis_list).argsort()[-n:][::-1]
        else:
            min_points = np.array(dis_list).argsort()[:n]

        return points[min_points]

    def _point_inference(self, src_pt: tuple, region: tuple):
        '''
        search stand template from bin file by key(chip_no).
        src_pt :(x, y, ind_x, ind_y)
        region: (height, width)
        '''
        if len(self.template) > 0:
            self.template = list()

        x0, y0, ind_x, ind_y = src_pt

        k0 = np.tan(np.radians(self.rotation))
        if k0 == 0: k0 = 0.00000001
        k1 = -1 / k0

        y_intercept0 = y0 - k0 * x0
        x_intercept0 = (y0 - k1 * x0) * k0

        dy = abs(k0 * region[1])
        y_region = (-dy, region[0] + dy)
        dx = abs(k0 * region[0])
        x_region = (-dx, region[1] + dx)

        self.y_intercept = self._get_intercept(self.scale_y, y_intercept0, y_region, ind_y, self.chip_no[1])
        self.x_intercept = self._get_intercept(self.scale_x, x_intercept0, x_region, ind_x, self.chip_no[0])
        self._create_cross_points(k0)

    def _get_intercept(self, scale, intercept0, region, ind, templ):
        idx = intercept0
        intercept = [[idx, ind]]
        s, e = region
        item_count = len(templ)
        # face to large
        while idx < e:
            ind = int(ind % item_count)
            item_len = (templ[ind] * scale) / np.cos(np.radians(self.rotation))
            idx += item_len
            intercept.append([idx, (ind + 1) % item_count])
            ind += 1
        # face to small
        idx, ind = intercept[0]
        while idx > s:
            ind -= 1
            ind = int(ind % item_count)
            item_len = (templ[ind] * scale) / np.cos(np.radians(self.rotation))
            idx -= item_len
            intercept.append([idx, ind])
        return sorted(intercept, key=(lambda x: x[0]))

    def _create_cross_points(self, k):
        for x_ in self.x_intercept:
            for y_ in self.y_intercept:
                x, ind_x = x_
                y, ind_y = y_
                x0 = (x - k * y) / (pow(k, 2) + 1)
                y0 = k * x0 + y
                self.template.append([x0, y0, ind_x, ind_y])

    def _to_digit(self, n):
        try: return float(n)
        except: return None

    ################
    '''output'''
    ################
    def save_template(self, output_path, template=None):
        '''
        :param output_path:
        :param template: 可传入其他模板保存
        :return: 保存模板点
        '''
        if template is None:
            _template = self.template
        else: _template = template

        if _template is not None and len(_template) > 0:
            if not os.path.exists(output_path): os.makedirs(output_path)
            np.savetxt(os.path.join(output_path, 'template.txt'), np.array(_template))
        else:
            clog.info("Template save failed.")

    def homography_image(self, image=None):
        '''
        :param image: str | array
        :return: 返回单应性矩阵和变换后的template点
        '''

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

        def vips2numpy(vi):
            return np.ndarray(buffer=vi.write_to_memory(),
                              dtype=format_to_dtype[vi.format],
                              shape=[vi.height, vi.width, vi.bands])

        points_re, points_qc = self.pair_to_template(self.template_qc, self.template, self._qc_thresh)
        matrix, mask = cv.findHomography(points_qc, points_re[:, :2], 0)
        homo_template = [temp[:2] - matrix[:2, 2] for temp in self.template]
        if image is None:
            return matrix, homo_template
        else:
            import pyvips
            if type(image) == str:
                _image = pyvips.Image.new_from_file(image)
            else:
                _image = pyvips.Image.new_from_array(image)
            m = list(matrix[:2, :2].flatten())
            _image = _image.affine(m, interpolate=pyvips.Interpolate.new("nearest"), background=[0])
            mat = vips2numpy(_image)
            if mat.ndim == 3:
                if mat.shape[2] != 3:
                    mat = mat[:, :, 0]

            return mat, homo_template


if __name__ == '__main__':
    import h5py

    ipr_path = r"D:\AllData\temp_data\D01862B1_20230505_115103_0.1.ipr"
    pts = {}
    with h5py.File(ipr_path) as conf:
        qc_pts = conf['QCInfo/CrossPoints/']
        for i in qc_pts.keys():
            pts[i] = conf['QCInfo/CrossPoints/' + i][:]
        scalex = conf['Register'].attrs['ScaleX']
        scaley = conf['Register'].attrs['ScaleY']
        rotate = conf['Register'].attrs['Rotation']
        # loc = conf['Stitch/BGIStitch/StitchedGlobalLoc'][...]
        loc = conf['Research/Stitch/StitchFovLocation'][...]
        index = conf['Stitch'].attrs['TemplateSource']

    chipno = [[240, 300, 330, 390, 390, 330, 300, 240, 420],
              [240, 300, 330, 390, 390, 330, 300, 240, 420]]

    # chipno = [[112, 144, 208, 224, 224, 208, 144, 112, 160],
    #           [112, 144, 208, 224, 224, 208, 144, 112, 160]]

    tr = TemplateReference()
    tr.set_scale(scalex, scaley)
    tr.set_rotate(rotate)
    tr.set_chipno(chipno)
    tr.set_fov_location(loc)
    tr.set_qc_points(index, pts)

    # tr.first_template_correct()
    tr.reference_template(mode='multi')
    dct = tr.get_template_eval()
    mat = tr.get_global_eval()
    print(1)


    # mat, template = tr.homography_image(r'')

import datetime
import cv2
import argparse
import os
import math
import time
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.mixture import GaussianMixture
from multiprocessing import Process

# from cellbin.contrib.cell_mask import Mask
# from cellbin.modules.cell_labelling import CellLabelling

from warnings import filterwarnings

filterwarnings('ignore')


class GMM(object):
    def __init__(self, gem_file, threshold, process):
        self.exp_matrix = gem_file
        self.out = []
        self.gmm_result = None
        self.threshold = threshold
        self.process = process
        self.radius = 50

    def __creat_gxp_data(self):
        data = self.exp_matrix
        cell_data = data[data.CellID != 0].copy()
        cell_coor = cell_data.groupby('CellID').mean()[['x', 'y']].reset_index()
        return data, cell_data, cell_coor

    def GMM_func(self, data, cell_coor, x, p_num):
        t0 = time.time()
        p_data = []
        if not os.path.exists(os.path.join(self.out_path, 'bg_adjust_CellID')):
            os.mkdir(os.path.join(self.out_path, 'bg_adjust_CellID'))
        for idx, i in enumerate(x):
            if (idx + 1) % 10 == 0:
                t1 = time.time()
                print("proc {}: {}/{} done, {:.2f}s.".format(p_num, idx, len(x), t1 - t0))
            try:
                clf = GaussianMixture(n_components=3, covariance_type='spherical')
                # Gaussian Mixture Model GPU version
                cell_test = data[(data.x < cell_coor.loc[i].x + self.radius) & (data.x > cell_coor.loc[i].x - self.radius) & (
                        data.y > cell_coor.loc[i].y - self.radius) & (data.y < cell_coor.loc[i].y + self.radius)]
                # fit GaussianMixture Model
                clf.fit(cell_test[cell_test.CellID == cell_coor.loc[i].CellID][['x', 'y', 'MIDCount']].values)
                cell_test_bg_ori = cell_test[cell_test.CellID == 0]
                bg_group = cell_test_bg_ori.groupby(['x', 'y']).agg(MID_max=('MIDCount', 'max')).reset_index()
                cell_test_bg = pd.merge(cell_test_bg_ori, bg_group, on=['x', 'y'])
                # threshold 20
                score = pd.Series(-clf.score_samples(cell_test_bg[['x', 'y', 'MID_max']].values))
                cell_test_bg['score'] = score
                threshold = self.threshold
                cell_test_bg['CellID'] = np.where(score < threshold, cell_coor.loc[i].CellID, 0)
                # used multiprocessing have to save result to file
                p_data.append(cell_test_bg)
            except Exception as e:
                print(e)
                # with open(os.path.join(self.out_path, 'error_log.txt'), 'a+') as f:
                #     f.write('Cell ID: {}\n'.format(cell_coor.loc[i].CellID))

        out = pd.concat(p_data)
        out.drop('MID_max', axis=1, inplace=True)
        # out.to_csv(os.path.join(self.out_path, 'bg_adjust_CellID', '{}.txt'.format(p_num)), sep='\t', index=False)

    def _GMM_score(self, data, cell_coor):
        processes = []
        qs = math.ceil(len(cell_coor.index) / int(self.process))
        for i in tqdm(range(self.process)):
            idx = np.arange(i * qs, min((i + 1) * qs, len(cell_coor.index)))
            if len(idx) == 0: continue
            p = Process(target=self.GMM_func, args=(data, cell_coor, idx, i))
            p.start()
            processes.append(p)
        [pi.join() for pi in processes]
        for p in processes:
            self.out.append(p.get())

    def _GMM_correction(self, cell_data):

        bg_data = []
        error = []

        for i in tqdm(self.out):
            bg_data.append(i[i.CellID != 0])
        adjust_data = pd.concat(bg_data).sort_values('score')
        adjust_data = adjust_data.drop_duplicates(subset=['geneID', 'x', 'y', 'MIDCount'], keep='first').rename(
            columns={'score': 'tag'})
        adjust_data['tag'] = '1'
        cell_data['tag'] = '0'
        correct_data = pd.concat([adjust_data, cell_data])
        self.gmm_result = correct_data

    def cell_correct(self):
        t0 = time.time()
        data, cell_data, cell_coor = self.__creat_gxp_data()
        t1 = time.time()
        print('Load data :', (t1 - t0))
        self._GMM_score(data, cell_coor)
        t2 = time.time()
        print('Calc score :', (t2 - t1))
        self._GMM_correction(cell_data)
        t3 = time.time()
        print('Correct :', (t3 - t2))
        print('Total :', (t3 - t0))


def args_parse():
    usage = """ Usage: %s Cell expression file (with background) path, multi-process """
    arg = argparse.ArgumentParser(usage=usage)
    arg.add_argument('-m', '--mask_path', help='cell mask',
                     default=r"D:\Cell_Bin\data\cellbin_meeting\mouse_liver\SS200000427TR_E5\SS200000427TR_E5_20230325_130444_0.1_mask.tif")
    arg.add_argument('-g', '--gem_path', help='gem file',
                     default=r"D:\Cell_Bin\data\cellbin_meeting\mouse_liver\SS200000427TR_E5\SS200000427TR_E5.gem.gz")
    arg.add_argument('-o', '--out_path', help='output path', default='./')
    arg.add_argument('-p', '--process', help='n process', type=int, default=10)
    arg.add_argument('-t', '--threshold', help='threshold', type=int, default=20)

    return arg.parse_args()


def main():
    args = args_parse()
    correction = GMM(args.mask_path, args.gem_path, args.out_path, args.threshold, args.process)
    correction.cell_correct()


if __name__ == '__main__':
    main()

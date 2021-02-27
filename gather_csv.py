# -*- coding: utf-8 -*-
import argparse
import logging
import time
import os
import csv
import json
from datetime import datetime
import sys
# rosのcv2との競合を避けるための処理
remove_path = '/opt/ros/melodic/lib/python2.7/dist-packages'
if remove_path in sys.path:
    sys.path.remove(remove_path)
import cv2

import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser

from sklearn import cluster, datasets, mixture
from sklearn.neighbors import kneighbors_graph

class DBSCAN():
    def __init__(self, path, data, eps, min_samples):
        self.ex_path = path
        self.data = data
        self.eps = eps
        self.min_samples = min_samples
        self.dbscan_data = None
        self.cluster_num = None
        self.noise_num = None
        self.calc_time = None

    def calc(self):
        self.dbscan_data = cluster.DBSCAN(
            eps=self.eps, min_samples=self.min_samples, metric='euclidean').fit_predict(self.data)
        self.noise_num = sum(self.dbscan_data == -1)
        if self.noise_num == 0:
            self.cluster_num = len(set(self.dbscan_data))
        else:
            self.cluster_num = len(set(self.dbscan_data)) - 1


def cluster_plots(path, data, colors='gray', title1='Dataset 1'):  # グラフ作成
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.set_title(title1, fontsize=14)
    body_part = 1
    body_part_index = body_part * 2
    ax1.scatter(data[:, body_part_index], data[:, body_part_index+1], s=8, lw=0, c=colors)
    plt.xlim(0,640)
    plt.ylim(0,480)
    fig.savefig(path + '/dbscan_result.png')

# csvを読んで要素をfloatにする
def str2float(path):
    with open(path) as f:
        reader = csv.reader(f)
        data = np.array([row for row in reader]).astype(np.float32)
    return data





if __name__=='__main__':
    parser = ArgumentParser()
    parser.add_argument('--dir', type=str, default="",
                        help='directory of the experiment')
    parser.add_argument('--cluster', type=int, default=3,
                        help='expect number of cluster')
    parser.add_argument('--loop', type=bool, default=False,
                        help='loop of DBSCAN for define the eps')
    args = parser.parse_args()


    # OpoenPoseデータの読み込み
    csv_dir = '/home/kubotalab-hsr/catkin_ws/src/ros-unity/scripts/openpose_data/2021-02-26-06-00/csv/' 
    i = 0
    pose_data = []
    while True:
        try:
            csv_file = csv_dir+str(i)+'.csv'
            data = str2float(csv_file).tolist()[0]
            pose_data.append(data)
            i += 1
        except:
            break
    print(np.shape(pose_data))
    data = np.array(pose_data)

    # 信頼値の削除
    max_body_parts = 18
    confidence_score_indexes = list(range(2, 3*max_body_parts, 3))
    data = np.delete(data, confidence_score_indexes, 1)

    # epsを決めるためのグラフ作成
    if args.loop:
        good_eps, eps_array, cluster_nums, noise_nums = [], [], [], []
        for eps in range(100, 501,10):
            dbscan = DBSCAN(args.dir, data, eps, min_samples=10)
            dbscan.calc()
            eps_array.append(eps)
            cluster_nums.append(dbscan.cluster_num)
            noise_nums.append(dbscan.noise_num)
            if dbscan.cluster_num==1:
                break
            elif dbscan.cluster_num==args.cluster:
                good_eps.append(eps)
        
        print('good eps :'+str(good_eps[0])+' ~ '+str(good_eps[-1]))
        fig = plt.figure()
        ax1 = fig.add_subplot(2, 1, 1)
        ax1.plot(eps_array, cluster_nums)
        ax1.set_xlabel("Eps")
        ax1.set_ylabel("number of cluster")
        
        ax2 = fig.add_subplot(2, 1, 2)
        ax2.plot(eps_array, noise_nums)
        ax2.set_xlabel("Eps")
        ax2.set_ylabel("number of noise")
        plt.tight_layout()
        fig.savefig(args.dir + "/eps.png")
        plt.show()
    
    eps = int(input('input eps：'))
    dbscan = DBSCAN(args.dir, data, eps, min_samples=60)
    dbscan.calc()

    with open(csv_dir + '/correct_data.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(dbscan.dbscan_data)
    print('データ形状：' + str(np.shape(data)) )
    print("クラスタ：" + str(dbscan.cluster_num) +  "ノイズ：" + str(dbscan.noise_num) )

    cluster_plots(args.dir,data, dbscan.dbscan_data)
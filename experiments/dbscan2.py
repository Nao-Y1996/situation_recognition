#!/usr/bin/env python
# -*- coding: utf-8 -*-
# import rospy
import numpy as np
import matplotlib.pyplot as plt
import time
import csv
import os
import shutil
import glob

from argparse import ArgumentParser
from sklearn import cluster, datasets, mixture
from sklearn import metrics
# from sklearn.metrics import silhouette_score
# from sklearn.neighbors import kneighbors_graph

import sys
sys.path.remove('/opt/ros/melodic/lib/python2.7/dist-packages')
import cv2
import create_movie

class DBSCAN():
    def __init__(self, data, eps, min_samples,metric):
        self.data = data
        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric
        self.dbscan_data = None
        self.cluster_num = None
        self.noise_num = None
        self.calc_time = None

    def calc(self):
        self.dbscan_data = cluster.DBSCAN(
            eps=self.eps, min_samples=self.min_samples, metric=self.metric).fit_predict(self.data) # 'cityblock', 'cosine', 'euclidean', 'l1', 'l2', 'manhattan'
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
    ax1.scatter(data[:, body_part_index],
                data[:, body_part_index+1], s=8, lw=0, c=colors)
    # plt.xlim(0, 640)
    # plt.ylim(0, 480)
    fig.savefig(path + '/dbscan_result.png')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dir', type=str, default="",
                        help='directory of the experiment')
    parser.add_argument('--cluster', type=int, default=3,
                        help='expect number of cluster')
    parser.add_argument('--loop4eps', type=bool, default=False,
                        help='loop of DBSCAN for define the eps')
    parser.add_argument('--per_second', type=bool, default=False,
                        help='create data per second')
    parser.add_argument('--analyse', type=bool, default=False,
                        help='compare manual correct data with clustering result')
    args = parser.parse_args()

    max_body_parts = 18
    image_path = args.dir+"/images/"
    pose_path = args.dir + '/pose.csv'
    

    # OpenPoseデータを読み込む
    pose = []
    with open(pose_path) as f:
        reader = csv.reader(f)
        for row in reader:
            row = [float(v) for v in row[:max_body_parts*3]] # 最初の1人のデータだけ読み取る
            pose.append(row)
        data = np.array(pose).astype(np.float32)
    # 信頼値の削除
    confidence_score_indexes = list(range(2, 3*max_body_parts, 3))
    data = np.delete(data, confidence_score_indexes, 1)

    # nullを各jointの重心で補間したデータの作成 
    pose_null2center_path = args.dir + '/pose_null_center.csv'
    for i, row in enumerate(data):
        pose = row.reshape([max_body_parts,2]) # poseを2次元配列に変換
        visible_joints_num_array = np.count_nonzero(pose!=0.0, axis=0)
        # 1つでもjointが見えていれば、見えていない部分([0.0, 0.0])は重心で置き換える
        if (visible_joints_num_array != [0,0]).all():
            joints_center = np.sum(pose, axis=0) / visible_joints_num_array
            # [0.0, 0.0] --> [重心]
            for j, joint in enumerate(pose):
                if (joint==[0.0, 0.0]).all():
                    pose[j] = joints_center
            data[i] = pose.reshape([max_body_parts*2])
    with open(pose_null2center_path, 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(data)

    # DBSCANで使う距離を変えてクラスタリングを実行
    for metric in [ 'cityblock', 'euclidean' ]: # 'cityblock', 'euclidean', 'l1', 'l2', 'manhattan'
        
        # ディレクトリの設定
        result_path = args.dir+'/results_center/' + metric
        clustering_result_path = result_path + '/clustering_result.csv'
        clusterd_images_path = result_path + '/clusterd_images/'
        distance_path = result_path + '/distance_' + metric + '.csv'
        if not os.path.exists(clusterd_images_path):
            os.makedirs(clusterd_images_path)
        
        # 各時刻同士のposeデータの距離を記録
        d = []
        print(np.shape(data))
        for i in range(np.shape(data)[0]-1):
            if metric=='cityblock':
                d.append(np.sum(np.abs(data[i] - data[i+1])))
            if metric=='euclidean':
                d.append(np.sqrt(np.power(data[i] - data[i+1], 2).sum()))
        plt.plot(list(range(np.shape(data)[0]-1)), d )
        plt.savefig(result_path + '/distance_' + metric + '.png')

        d = np.array(d).reshape([-1,1])
        # none_zero_num = np.count_nonzero(d!=0.0, axis=0)
        # distance_average = sum(d) / none_zero_num
        # print(distance_average)
        with open(distance_path, 'w') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(d)

        # epsを決めるためのグラフ作成
        expect_cluster, expect_cluster_plus1 = False, False
        if args.loop4eps:
            print('------距離関数：' + metric + 'でグラフ作成中------')
            good_eps, eps_array, cluster_nums, noise_nums = [], [], [], []
            for eps in np.arange(0.01, 5, 0.01):
                print('\r--- eps = %f' % eps, end='')
                dbscan = DBSCAN(data, eps, min_samples=180,metric=metric)
                dbscan.calc()
                eps_array.append(eps)
                cluster_nums.append(dbscan.cluster_num)
                noise_nums.append(dbscan.noise_num)
                # クラスタ数が、期待するクラス多数-2担ったらループ終了
                if dbscan.cluster_num == args.cluster + 1 and expect_cluster_plus1==False:
                    expect_cluster_plus1 = True
                if dbscan.cluster_num == args.cluster and expect_cluster==False and expect_cluster_plus1:
                    expect_cluster = True
                if expect_cluster and dbscan.cluster_num == args.cluster-2:
                    break
                if dbscan.cluster_num == args.cluster:
                    good_eps.append(eps)
            try:
                print('good eps :'+str(good_eps[0])+' ~ '+str(good_eps[-1]))
            except IndexError:
                print('良いEpsが見つかりません')
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
            fig.savefig(result_path + "/eps.png")
            plt.show()

        # 適切なEpsを手動で入力
        eps = float(input('input eps：'))
        dbscan = DBSCAN( data, eps, min_samples=180,metric=metric)
        dbscan.calc()
        
        # クラスタリング結果を保存
        with open(result_path + '/clustering_result.csv', 'w') as f:
            writer = csv.writer(f)
            writer.writerows(dbscan.dbscan_data.reshape(-1,1))
        print('データ形状：' + str(np.shape(data)))
        print("クラスタ：" + str(dbscan.cluster_num) + "ノイズ：" + str(dbscan.noise_num))
        # cluster_plots(args.dir, data, dbscan.dbscan_data)

        # DBSCANの分類結果をimageに描画して動画作成
        # 分類結果を描画した画像の保存先の作成
        _eps = str(eps).replace('.','-')
        clusterd_images_path = clusterd_images_path + 'Eps_'+_eps+'/'
        if not os.path.exists(clusterd_images_path):
            os.makedirs(clusterd_images_path)
        
        # 分類結果の読み込み
        with open(clustering_result_path) as f:
            reader = csv.reader(f)
            clustering_result = [[int(v) for v in row] for row in reader]

        # 各フレームに分類結果を描画
        for index, row in enumerate(clustering_result):
            num = row[0]
            # 画像の読み込み
            image_file = image_path + str(index)+'.png'
            img = cv2.imread(image_file)    
            # 分類結果の書き込み（描画）
            if index%2000==0:
                print('---分類結果の描画---image:'+str(index))
            cv2.putText(img,"situation:"+str(num),(400, 420),  cv2.FONT_HERSHEY_SIMPLEX, 1,(0, 255, 0), 2)
            cv2.imwrite(clusterd_images_path + "/"+str(index)+'.png', img)
        
        # 動画の作成
        movie_path = result_path + '/clustering_result_Eps_' + _eps + '.mp4'
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        video = cv2.VideoWriter(movie_path, fourcc, 20.0, (640, 480))
        if not video.isOpened():
            print("can't be opened")
            sys.exit()
        i = 0
        while True:
            try:
                image_read_path = glob.glob(clusterd_images_path+str(i)+".png")[0]
                img = cv2.imread(image_read_path)
                if img is None:
                    print("can't read")
                    break
                video.write(img)
                if i % 2000 == 0:
                    print('image: -- %d' % i)
                i += 1
            except:
                break
        video.release()

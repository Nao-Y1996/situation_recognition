#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ライブラリのインポート
# import rospy
import numpy as np
# import matplotlib.pyplot as plt
# import time
import csv
import os
# import shutil
import cv2
import glob

from argparse import ArgumentParser
# from sklearn.metrics import silhouette_score
# from sklearn import cluster, datasets, mixture
# from sklearn.neighbors import kneighbors_graph

# import create_movie



# DBSCAN の分類結果をimageに描画して動画作成
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dir', type=str, default="",
                        help='directory of the experiment')
    args = parser.parse_args()

    image_path = args.dir+"/images/"
    correct_data_path = args.dir + '/correct_data.csv'
    clusterd_images_path = args.dir+'/clusterd_images/'

    # 分類結果を描画した画像の保存先の作成
    if not os.path.exists(clusterd_images_path):
        os.makedirs(clusterd_images_path)

    # 分類結果の読み込み
    with open(correct_data_path) as f:
        reader = csv.reader(f)
        correct_data = [[int(v) for v in row] for row in reader]

    # 各フレームに分類結果を描画
    for index, row in enumerate(correct_data):

        num = row[0]
        # 画像の読み込み
        image_file = image_path + str(index)+'.png'
        img = cv2.imread(image_file)    
        # 分類結果の書き込み（描画）
        if index%500==0:
            print('---分類結果の描画---image:'+str(index))
        cv2.putText(img,"situation:"+str(num),(400, 420),  cv2.FONT_HERSHEY_SIMPLEX, 1,(0, 255, 0), 2)
        cv2.imwrite(clusterd_images_path + str(index)+'.png', img)

    # 動画の作成
    movie_path = args.dir+'/clustering_result.mp4'
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    video = cv2.VideoWriter(movie_path, fourcc, 20.0, (640, 480))
    if not video.isOpened():
        print("can't be opened")
        sys.exit()
    i = 0
    while True:
        try:
            image_read_path = glob.glob(clusterd_images_path+"/"+str(i)+".png")[0]
            img = cv2.imread(image_read_path)

            if img is None:
                print("can't read")
                break
            video.write(img)
            if i % 500 == 0:
                print('image: -- %d' % i)
            i += 1
        except:
            break


        
#!/usr/bin/env python
# -*- coding: utf-8 -*-
# import rospy
import numpy as np
import csv
import math
import os
import json
import time
import argparse
import logging
import datetime

import numpy as np

from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh


import sys
# sys.path.remove('/opt/ros/melodic/lib/python2.7/dist-packages')
import cv2


# -- 各層の継承元 --
class BaseLayer:
    def __init__(self, n_upper, n):
        self.w = wb_width * np.random.randn(n_upper, n)  # 重み（行列）
        self.b = wb_width * np.random.randn(n)  # バイアス（ベクトル）

        self.h_w = np.zeros(( n_upper, n)) + 1e-8
        self.h_b = np.zeros(n) + 1e-8
        
    def update(self, eta):      
        self.h_w += self.grad_w * self.grad_w
        self.w -= eta / np.sqrt(self.h_w) * self.grad_w
        
        self.h_b += self.grad_b * self.grad_b
        self.b -= eta / np.sqrt(self.h_b) * self.grad_b
# -- 中間層 --
class MiddleLayer(BaseLayer):
    def forward(self, x):
        self.x = x
        self.u = np.dot(x, self.w) + self.b
        self.y = np.where(self.u <= 0, 0, self.u)  # ReLU
    
    def backward(self, grad_y):
        delta = grad_y * np.where(self.u <= 0, 0, 1)  # ReLUの微分

        self.grad_w = np.dot(self.x.T, delta)
        self.grad_b = np.sum(delta, axis=0)
        
        self.grad_x = np.dot(delta, self.w.T) 
# -- 出力層 --
class OutputLayer(BaseLayer):     
    def forward(self, x):
        self.x = x
        u = np.dot(x, self.w) + self.b
        self.y = np.exp(u)/np.sum(np.exp(u), axis=1, keepdims=True)  # ソフトマックス関数

    def backward(self, t):
        delta = self.y - t
        
        self.grad_w = np.dot(self.x.T, delta)
        self.grad_b = np.sum(delta, axis=0)
        
        self.grad_x = np.dot(delta, self.w.T) 
# --ドロップアウト--
class Dropout:
    def __init__(self, dropout_ratio):
        self.dropout_ratio = dropout_ratio  # ニューロンを無効にする確率

    def forward(self, x, is_train):  # is_train: 学習時はTrue
        if is_train:
            rand = np.random.rand(*x.shape)  # 入力と同じ形状の乱数の行列
            self.dropout = np.where(rand > self.dropout_ratio, 1, 0)  # 1:有効 0:無効
            self.y = x * self.dropout  # ニューロンをランダムに無効化
        else:
            self.y = (1-self.dropout_ratio)*x  # テスト時は出力を下げる
        
    def backward(self, grad_y):
        self.grad_x = grad_y * self.dropout  # 無効なニューロンでは逆伝播しない


logger = logging.getLogger('TfPoseEstimator-WebCam')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

fps_time = 0

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='tf-pose-estimation realtime webcam')
    parser.add_argument('--camera', type=int, default=0)

    parser.add_argument('--resize', type=str, default='0x0',
                        help='if provided, resize images before they are processed. default=0x0, Recommends : 432x368 or 656x368 or 1312x736 ')
    parser.add_argument('--resize-out-ratio', type=float, default=4.0,
                        help='if provided, resize heatmaps before they are post-processed. default=1.0')

    parser.add_argument('--model', type=str, default='mobilenet_thin', help='cmu / mobilenet_thin / mobilenet_v2_large / mobilenet_v2_small')
    parser.add_argument('--show-process', type=bool, default=False,
                        help='for debug purpose, if enabled, speed for inference is dropped.')
    
    parser.add_argument('--tensorrt', type=str, default="False",
                        help='for tensorrt process.')
    args = parser.parse_args()

    logger.debug('initialization %s : %s' % (args.model, get_graph_path(args.model)))
    w, h = model_wh(args.resize)
    if w > 0 and h > 0:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h), trt_bool=str2bool(args.tensorrt))
    else:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(432, 368), trt_bool=str2bool(args.tensorrt))
    logger.debug('cam read+')
    cam = cv2.VideoCapture(args.camera)
    ret_val, image = cam.read()
    logger.info('cam image=%dx%d' % (image.shape[1], image.shape[0]))

    # 状況認識結果のcsv保存用の設定
    dir_here =  os.path.dirname(os.path.abspath(__file__))
    result_dir = dir_here + '/situation_recognition_result/' 
    dt_now = datetime.datetime.now()
    new_dir_path = str(dt_now)[0:16].replace(' ', '-').replace(':', '-')
    save_dir = result_dir + new_dir_path
    os.makedirs(save_dir+'/images/')
    recogniton_result_path =  save_dir + '/recogniton_result.csv'
    f= open(recogniton_result_path, 'w')
    f.close
    pose_path =  save_dir + '/pose.csv'
    f= open(pose_path, 'w')
    f.close

    # 動画ファイル保存用の設定
    camera_w = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
    camera_h = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # print(camera_h, camera_w)
    # fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # video = cv2.VideoWriter(base_dir + '/output.mov', fourcc, 30, (camera_w,camera_h))

    #=============NNの定義==============
    base_dir = '/home/kubotalab-hsr/catkin_ws/src/ros-unity/scripts/' 
    with open(base_dir + '/NN_model/model.json') as f:
        df = json.load(f)
    n_in, n_mid, n_out = df['n_in'], df['n_mid'], df['n_out']
    wb_width, eta = df['wb_width'],  df['eta']
    # -- 各層の初期化 --
    # ml_1 = MiddleLayer(n_in, n_mid)
    # ml_2 = MiddleLayer(n_mid, n_mid)
    # ol = OutputLayer(n_mid, n_out)
    ml_1 = MiddleLayer(n_in, n_mid)
    dp_1 = Dropout(0.5)
    ml_2 = MiddleLayer(n_mid, n_mid)
    dp_2 = Dropout(0.5)
    ol = OutputLayer(n_mid, n_out)
    # -- 順伝播 --
    def fp(x, is_train):
        # ml_1.forward(x)
        # ml_2.forward(ml_1.y)
        # ol.forward(ml_2.y)
        ml_1.forward(x)
        dp_1.forward(ml_1.y, is_train)
        ml_2.forward(dp_1.y)
        dp_2.forward(ml_2.y, is_train)
        ol.forward(dp_2.y)
    #==================================

    #===========学習済みの重みを読み込む============
    # csvを読んで要素をfloatにする
    def str2float(path):
        with open(path) as f:
            reader = csv.reader(f)
            data = np.array([row for row in reader]).astype(np.float64)
        return data
    w_dir = base_dir + '/NN_model/wb/'
    ml_1.b = str2float(w_dir + 'b_ml_1.csv')
    ml_1.w = str2float(w_dir + 'w_ml_1.csv')
    ml_2.w = str2float(w_dir + 'w_ml_2.csv')
    ml_2.b = str2float(w_dir + 'b_ml_2.csv')
    ol.w = str2float(w_dir + 'w_ol.csv')
    ol.b = str2float(w_dir + 'b_ol.csv')
    #============================================

    def detect(pose_info):
        fp(pose_info, is_train=False)

    elasped_time = 0
    count = 0
    second = 1
    while True:
        ret_val, image = cam.read()
        humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=args.resize_out_ratio)
        # ---------------以下、追加部分--------------
        # all_pose_data = []
        if len(humans)!=0:
            for human in humans:
                # print(human)
                pose_data = []
                for part_index in range(18):
                    try:
                        part = human.body_parts[part_index]
                        pose_data.extend([int(part.x*camera_w), int(part.y*camera_h)])
                    except:
                        pose_data.extend([0, 0])
                # print(pose_data)
                detect(pose_data) 
                recognition_result = ol.y
                print('-------situation: '+str(recognition_result))
                # all_pose_data.append(pose_data)
        #         if elasped_time > 1.0:
                # with open(recogniton_result_path, 'a') as csvfile:
                #     writer = csv.writer(csvfile)
                #     writer.writerow(recognition_result)
                # with open(pose_path, 'a') as csvfile:
                #     writer = csv.writer(csvfile)
                #     writer.writerows(all_pose_data)
        #             elasped_time = 0
        #             second += 1
        #         else:
        #             with open(pose_path, 'a') as csvfile:
        #                 writer = csv.writer(csvfile)
        #                 writer.writerows(all_pose_data)
        # else:
        #     all_pose_data = np.zeros(18*3)
        #     with open(pose_path, 'a') as csvfile:
        #         writer = csv.writer(csvfile)
        #         writer.writerow(all_pose_data)

        # -----------------ここまで-------------------
        image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)

        processing_time = (time.time() - fps_time)
        fps = 1.0 / processing_time
        # print('-------fps------' + str(fps))
        # cv2.putText(image,
        #             "FPS: %f" % (1.0 / processing_time),
        #             (10, 10),  cv2.FONT_HERSHEY_SIMPLEX, 0.5,
        #             (0, 255, 0), 2)
        cv2.imshow('tf-pose-estimation result', image)
        # video.write(image) # 1フレームずつ保存する
        cv2.imwrite(save_dir + '/images/' + str(count) + '.png',image)
        fps_time = time.time()
        if cv2.waitKey(1) == 27:
            break
        elasped_time += processing_time
        count += 1
    cam.release()
    cv2.destroyAllWindows()


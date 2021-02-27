# -*- coding: utf-8 -*-

# from tf_pose import common

import numpy as np
import time
from datetime import datetime

import argparse
import logging

# from tf_pose.estimator import TfPoseEstimator
# from tf_pose.networks import get_graph_path, model_wh
import sys
sys.path.remove('/opt/ros/melodic/lib/python2.7/dist-packages')
import cv2
camera = cv2.VideoCapture(2)

# 動画ファイル保存用の設定
fps = int(camera.get(cv2.CAP_PROP_FPS))        
w = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH)) 
h = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
video = cv2.VideoWriter('output.mp4', fourcc, fps, (w,h))

start_time = datetime.now()
record_time = 30
while (datetime.now() - start_time).seconds <= record_time:
    ret, frame = camera.read()
    cv2.imshow('camera', frame)
    video.write(frame)

    # キー操作があればwhileループを抜ける
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 撮影用オブジェクトとウィンドウの解放
camera.release()
cv2.destroyAllWindows()

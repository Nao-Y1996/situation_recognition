import argparse
import logging
import time
import os
import csv
import datetime

import numpy as np

from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh

import sys
# sys.path.remove('/opt/ros/melodic/lib/python2.7/dist-packages')
import cv2
logger = logging.getLogger('TfPoseEstimator-WebCam')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter(
    '[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='tf-pose-estimation realtime webcam')
    parser.add_argument('--camera', type=int, default=0)

    parser.add_argument('--resize', type=str, default='0x0',
                        help='if provided, resize images before they are processed. default=0x0, Recommends : 432x368 or 656x368 or 1312x736 ')
    parser.add_argument('--resize-out-ratio', type=float, default=4.0,
                        help='if provided, resize heatmaps before they are post-processed. default=1.0')

    parser.add_argument('--model', type=str, default='mobilenet_thin',
                        help='cmu / mobilenet_thin / mobilenet_v2_large / mobilenet_v2_small')
    parser.add_argument('--show-process', type=bool, default=False,
                        help='for debug purpose, if enabled, speed for inference is dropped.')

    parser.add_argument('--tensorrt', type=str, default="False",
                        help='for tensorrt process.')
    args = parser.parse_args()

    logger.debug('initialization %s : %s' %
                 (args.model, get_graph_path(args.model)))
    w, h = model_wh(args.resize)
    if w > 0 and h > 0:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(
            w, h), trt_bool=str2bool(args.tensorrt))
    else:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(
            432, 368), trt_bool=str2bool(args.tensorrt))
    logger.debug('cam read+')
    cam = cv2.VideoCapture(args.camera)
    ret_val, image = cam.read()
    logger.info('cam image=%dx%d' % (image.shape[1], image.shape[0]))

    # pose情報のcsv保存用の設定
    dir_here = os.path.dirname(os.path.abspath(__file__))
    # base_dir = '/home/kubotalab-hsr/Desktop/webcamera_pose_data'
    base_dir = dir_here + '/data/'
    dt_now = datetime.datetime.now()
    new_dir_path = str(dt_now)[0:16].replace(' ', '-').replace(':', '-')
    save_dir = base_dir + new_dir_path
    os.makedirs(save_dir+'/images/')
    pose_par_second_path = save_dir + '/index_per_second.csv'
    f = open(pose_par_second_path, 'w')
    f.close
    pose_path = save_dir + '/pose.csv'
    f = open(pose_path, 'w')
    f.close

    # 動画ファイル保存用の設定
    camera_w = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
    camera_h = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # print(camera_h, camera_w)
    # fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # video = cv2.VideoWriter(base_dir + '/output.mov', fourcc, 30, (camera_w,camera_h))

    elasped_time = 0
    frame_num = 0  # 何フレーム目か
    index_pose_par_second = []  # 1秒ごとにその時何フレーム目かを保存する配列
    while True:
        processing_start = time.time()
        ret_val, image = cam.read()
        # 骨格推定を実行
        humans = e.inference(image, resize_to_default=(
            w > 0 and h > 0), upsample_size=args.resize_out_ratio)

        all_pose_data = []
        # 人間がいるとき
        if len(humans) != 0:
            image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)
            # poseを取得
            for human in humans:
                pose_data = []
                for part_index in range(18):
                    try:
                        part = human.body_parts[part_index]
                        pose_data.extend(
                            # [int(part.x*camera_w), int(part.y*camera_h), round(part.score, 4)])
                            [round(part.x, 4), round(part.y,4), round(part.score, 4)])
                    except:
                        pose_data.extend([0.0, 0.0, 0.0])
                all_pose_data.extend(pose_data)
            # 毎フレームposeを記録する
            with open(pose_path, 'a') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(all_pose_data)
        # 人間がいないとき
        else:
            all_pose_data = np.zeros(18*3)
            with open(pose_path, 'a') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(all_pose_data)

        # 1秒ごとに何フレーム目か記録
        if elasped_time > 1.0:
            with open(pose_par_second_path, 'a') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([frame_num])
            elasped_time = 0

        cv2.imwrite(save_dir + '/images/' + str(frame_num) + '.png', image)
        # cv2.putText(image,"frame: : %f" % frame,(5, 5),  cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0, 255, 0), 2)

        # フレームの表示
        cv2.imshow('tf-pose-estimation result', image)

        # image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)
        # cv2.putText(image,"FPS: %f" % (1.0 / processing_time),(10, 10),  cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0, 255, 0), 2)
        # cv2.imshow('tf-pose-estimation result', image)
        # cv2.imwrite(save_dir + '/images/' + str(count) + '.png',image)

        processing_time = time.time() - processing_start
        elasped_time += processing_time
        # fps = 1.0 / processing_time

        frame_num += 1

        if cv2.waitKey(1) == 27:
            break
    cam.release()
    cv2.destroyAllWindows()

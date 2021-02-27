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

from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh
logger = logging.getLogger('TfPoseEstimator-Video')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

fps_time = 0


if __name__ == '__main__':

    # 保存用ディレクトリの設定
    base_dir = '/home/kubotalab-hsr/catkin_ws/src/ros-unity/scripts/openpose_data/' 
    dt_now = datetime.now()
    new_dir_path = str(dt_now)[0:16].replace(' ', '-').replace(':', '-')
    save_dir = base_dir + new_dir_path
    # 各フレームにおけるposeとimageの保存用先
    os.makedirs(save_dir+'/images/')
    os.makedirs(save_dir+'/csv/')
    image_dir = save_dir+'/images/'
    csv_dir = save_dir+'/csv/'
    # キャプチャ動画保の存先
    row_video_path = save_dir + '/row_video.mp4'
    # OpenPose動画の保存先
    pose_video_path = save_dir + '/pose_output.mp4'

    # OpenPose用の設定
    parser = argparse.ArgumentParser(description='tf-pose-estimation Video')
    parser.add_argument('--record_time', type=int, default=3600)
    parser.add_argument('--camera', type=int, default=0)
    parser.add_argument('--video', type=str, default=row_video_path)
    parser.add_argument('--resize', type=str, default='0x0',
                        help='if provided, resize images before they are processed. default=0x0, Recommends : 432x368 or 656x368 or 1312x736 ')
    parser.add_argument('--resize-out-ratio', type=float, default=4.0,
                        help='if provided, resize heatmaps before they are post-processed. default=1.0')
    parser.add_argument('--model', type=str, default='mobilenet_thin', help='cmu / mobilenet_thin')
    parser.add_argument('--show-process', type=bool, default=False,
                        help='for debug purpose, if enabled, speed for inference is dropped.')
    parser.add_argument('--showBG', type=bool, default=True, help='False to show skeleton only.')
    args = parser.parse_args()

    logger.debug('initialization %s : %s' % (args.model, get_graph_path(args.model)))

    w, h = model_wh(args.resize)
    if w > 0 and h > 0:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h))
    else:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(432, 368))
    
    # ---------------WebCameraで録画-------------------
    camera = cv2.VideoCapture(args.camera)
    w = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH)) 
    h = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # カメラが最高で出せるfps 実際に1秒あたり30フレーム保存できるとは限らない
    # 保存する時の1フレームの時間はここで決まるため60秒録画しても60秒の動画になるとは限らない
    fps = int(camera.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    video = cv2.VideoWriter(row_video_path, fourcc, fps, (w,h))

    start_time = datetime.now()
    frame_num = 0
    while (datetime.now() - start_time).seconds <= args.record_time:
        ret, frame = camera.read()
        cv2.imshow('camera', frame)
        video.write(frame)
        frame_num += 1
        
        # キー操作があればwhileループを抜ける
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    print('録画時間：'+str(args.record_time))
    print('フレーム数：'+str(frame_num))
    print('1フレームの秒数：'+str(1/fps))
    camera.release()
    video.release()
    cv2.destroyAllWindows()

    movie_info = {
        'record_time':args.record_time,
        'frame_num':frame_num,
        'fps':fps,
        'movie_length':(1/fps)*frame_num
    }
    with open(save_dir+'/movie_info.json', 'w') as f:
        json.dump(movie_info, f, indent=2, ensure_ascii=False)
    #-------------------------------------------------------

    # openpose動画ファイル保存用の設定
    cap = cv2.VideoCapture(row_video_path) # 動画の読み込み
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if cap.isOpened() is False:
        print("Error opening video stream or file")
    
    count = 0
    while cap.isOpened():
        ret_val, image = cap.read()
        try :
            pose_frame_rate = 1 # 1フレームごとにposeとimageを保存
            if count%pose_frame_rate==0:
                humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=args.resize_out_ratio)
                if not args.showBG:
                    image = np.zeros(image.shape)
                # --------------csvにposeを保存--------------
                all_pose_data = []
                for human in humans:
                    pose_data = []
                    for part_index in range(18):
                        try:
                            part = human.body_parts[part_index]
                            pose_data.extend([int(part.x*w), int(part.y*h),round(part.score,4)])
                        except:
                            pose_data.extend([0, 0, 0.0])
                    all_pose_data.append(pose_data)
                with open(csv_dir + str(count)+ '.csv', 'w') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerows(all_pose_data)

                # ----------------pose画像の保存---------------
                image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)
                cv2.imwrite(image_dir + str(count) + '.png', image)
                if count%100==0:
                    print('Creating OpenPose images ... '+str(count))
            count += 1
            if cv2.waitKey(1) == 27:
                break
        except:
            cv2.destroyAllWindows()
            cap.release()


    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    video = cv2.VideoWriter(pose_video_path,fourcc, fps, (w, h))

    if not video.isOpened():
        print("can't be opened")
        sys.exit()
    print('start createing movie ... ')
    count = 0
    while True:
        try:
            img = cv2.imread(image_dir + str(count) + '.png')

            if img is None:
                print("can't read")
                break
            video.write(img)
            if count%100==0:
                print('image: -- %d' % count )
            count += pose_frame_rate
        except:
            break

    video.release()
    print('written')
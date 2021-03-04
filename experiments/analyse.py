#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import csv
from argparse import ArgumentParser
 
if __name__ == '__main__':
   parser = ArgumentParser()
   parser.add_argument('--dir', type=str, default="",
                       help='directory of the experiment')
   args = parser.parse_args()

   for metric in [ 'cityblock', 'euclidean' ]: # 'cityblock', 'euclidean', 'l1', 'l2', 'manhattan'
        
        for result in ['results', 'results_center']:
            # ディレクトリの設定
            result_path = args.dir+'/' + result + '/' + metric
            clustering_result_path = result_path + '/clustering_result.csv'
            manual_data_path = args.dir + '/correct_data_manual.csv'
            
            clustering_result = []
            with open(clustering_result_path) as f:
                reader = csv.reader(f)
                for row in reader:
                    clustering_result.append(int(row[0]))
            
            manual_data = []
            with open(manual_data_path) as f:
                reader = csv.reader(f)
                for row in reader:
                    manual_data.append(int(row[0]))
            # print(manual_data)
            correct_data_manual = []
            num = 0
            s = 0
            for count, _ in enumerate(range(len(clustering_result))):
                try:
                    if manual_data[2*s] <= count < manual_data[2*s+1]:
                        correct_data_manual.append(num)
                    else:
                        correct_data_manual.append(-1)
                    if count == manual_data[2*s+1]:
                        num += 1
                        s += 1
                except IndexError:
                    correct_data_manual.append(-1)
            
            frame = list(range(len(correct_data_manual)))

            count = 0
            for s1, s2 in zip(clustering_result, correct_data_manual):
                if s1 == s2:
                    count += 1
            correct_answer_rate = (count/len(correct_data_manual))*100

            fig, ax = plt.subplots()
            ax.set_xlabel('frame')  # x軸ラベル
            ax.set_ylabel('situation')  # y軸ラベル
            clustering_result = np.array(clustering_result)-0.05
            correct_data_manual = np.array(correct_data_manual)
            ax.scatter(frame, clustering_result, color="blue",
                        label='clustering result', marker=".")
            ax.scatter(frame, correct_data_manual, color="red",
                        label='correct data', marker=".")
            ax.legend(loc=0)    # 凡例
            ax.set_title(str(correct_answer_rate))
            fig.tight_layout()  # レイアウトの設定
            plt.savefig(result_path + '/' + metric + '_analysis.png') # 画像の保存
            plt.show()
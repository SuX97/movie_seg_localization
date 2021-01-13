import moviepy
import sys
import os.path as osp
from moviepy.editor import VideoFileClip, clips_array, TextClip, CompositeVideoClip
import numpy as np
import glob
from pdb import set_trace as st
import json
import cv2
from multiprocessing import Pool
NUM_PROC = 16


def str2sec(s):
    h, m, s = (s.split(':'))
    return 3600 * int(h) + 60 * int(m) + int(s)


def vis_func(ori_id, shots, video_root):
    clips_list = []
    ori_path = glob.glob(osp.join(video_root, '*', 'original', f"{ori_id}*"))[0]
    with VideoFileClip(ori_path) as clip_ori:
        clip_ori = VideoFileClip(ori_path)
        total_duration = 0
        for shot in shots:
            start_time = shot['start']
            end_time = shot['end']
            # st()
            clips_list.append(clip_ori.subclip(start_time, end_time).resize((255, 255)).set_start(total_duration))

            for i, sub_shot_id in enumerate(shot['shot_id']):
                shot_path = glob.glob(osp.join(video_root, '*', 'shots', f'{sub_shot_id}*'))[0]
                print(f'stack {shot_path}')
                shot_clip = VideoFileClip(shot_path).set_start(total_duration).resize((122, 122))
                clips_list.append(shot_clip)
                # let's just save one shot for clarity and speed
                # break
            total_duration += (str2sec(end_time) - str2sec(start_time))
        stacked_video = CompositeVideoClip(clips_list)
        stacked_video.write_videofile(f'{ori_id}_annotated.gif')


if __name__ == '__main__':
    # visualize(sys.argv[1], sys.argv[2])
    result_path, video_root = sys.argv[1:]
    if isinstance(result_path, str):
        with open(result_path, 'r') as f:
            result_dict = json.load(f)
    
    if NUM_PROC == 0:
        for ori_id, shots in result_dict.items():
            vis_func(ori_id, shots, video_root)
    else:
        with Pool(processes=NUM_PROC) as pool:
            for ori_id, shots in result_dict.items():
                pool.apply_async(vis_func, (ori_id, shots, video_root))
            pool.close()
            pool.join()


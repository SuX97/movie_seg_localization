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

def str2sec(s):
    h, m, s = (s.split(':'))
    return 3600 * int(h) + 60 * int(m) + int(s)

# def visualize(result_path, video_root):
#     if isinstance(result_path, str):
#         with open(result_path, 'r') as f:
#             result_dict = json.load(f)

#     for ori_id, shots in result_dict.items():

def vis_func(ori_id, shots, video_root):
    clips_list = []
    ori_path = glob.glob(osp.join(video_root, '*', 'original', f"{ori_id}*"))[0]
    clip_ori = VideoFileClip(ori_path)
    clips_list.append(clip_ori.resize((255, 255)))
    for shot in shots:
        total_duration = 0
        start_time = str2sec(shot['start'])
        end_time = str2sec(shot['end'])
        # st()
        # txt_clip = TextClip(f"Similarity {shot['score']}",fontsize=70,color='green')
        # txt_clip.set_start(start_time)
        # txt_clip.set_duration(end_time  - start_time)

        # st()
        for i, sub_shot_id in enumerate(shot['shot_id']):
            shot_path = glob.glob(osp.join(video_root, '*', 'shots', f'{sub_shot_id}*'))[0]
            print(f'stack {shot_path}')
            shot_clip = VideoFileClip(shot_path)
            shot_clip.set_start(start_time)
            # shot_clip.set_duration(end_time - start_time)
            total_duration += shot_clip.duration
            clips_list.append(shot_clip.resize((255, 255)))
            # let's just save one shot for clarity and speed
            break
        # print(total_duration, (end_time - start_time))
        # try:
        #     assert (np.allclose(total_duration, (end_time - start_time)))
        # except AssertionError:
        #     print(total_duration, (end_time - start_time))

        # shot_clip.set_start(start_time)
        # clips_list.append()
    
    stacked_video = clips_array([clips_list])
    # stacked_video = CompositeVideoClip([clip_ori, txt_clip])
    stacked_video.write_videofile(f'{ori_id}_annotated.mp4')


# def visualize_cv2(result_path, video_root):
#     if isinstance(result_path, str):
#         with open(result_path, 'r') as f:
#             result_dict = json.load(f)

    # for ori_id, shots in result_dict.items():
    #     clips_list = []
    #     ori_path = glob.glob(osp.join(video_root, '*', 'original', f"{ori_id}*"))[0]
    #     clip_ori = VideoFileClip(ori_path)
    #     clips_list.append(clip_ori)
    #     for shot in shots:
    #         total_duration = 0
    #         start_time = clip_ori.duration * shot['start']
    #         end_time = clip_ori.duration * shot['end']

if __name__ == '__main__':
    # visualize(sys.argv[1], sys.argv[2])
    result_path, video_root = sys.argv[1:]
    if isinstance(result_path, str):
        with open(result_path, 'r') as f:
            result_dict = json.load(f)
    with Pool(processes=16) as pool:
        for ori_id, shots in result_dict.items():
            print(ori_id)
            pool.apply_async(vis_func, (ori_id, shots, video_root))
        pool.close()
        pool.join()


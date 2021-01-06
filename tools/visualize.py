import moviepy
import sys
import os.path as osp
from moviepy.editor import VideoFileClip, clips_array
import numpy as np
import glob
from pdb import set_trace as st
import json

def visualize(result_path, video_root):
    if isinstance(result_path, str):
        with open(result_path, 'r') as f:
            result_dict = json.load(f)

    for ori_id, shots in result_dict.items():
        clips_list = []
        ori_path = glob.glob(osp.join(video_root, '*', 'original', f"{ori_id}*"))[0]
        clip_ori = VideoFileClip(ori_path).margin(10)
        clips_list.append(clip_ori)
        for shot in shots:
            total_duration = 0
            start_time = clip_ori.duration * shot['start']
            end_time = clip_ori.duration * shot['end']
            for i, sub_shot_id in enumerate(shot['shot_it']):
                shot_path = glob.glob(osp.join(video_root, '*', 'shots', f'{sub_shot_id}*'))[0]
                shot_clip = VideoFileClip(shot_path)
                shot_clip.set_start(start_time + total_duration)
                total_duration += shot_clip.duration
                clips_list.append(shot_clip)
            print(total_duration, (end_time - start_time))
            # try:
            #     assert (np.allclose(total_duration, (end_time - start_time)))
            # except AssertionError:
            #     print(total_duration, (end_time - start_time))

            # shot_clip.set_start(start_time)
            # clips_list.append()
        
        stacked_video = clips_array(clips_list)
        stacked_video.write_videofile(f'{ori_id}_annotated.mp4', fps=clip_ori.fps)


if __name__ == '__main__':
    visualize(sys.argv[1], sys.argv[2])


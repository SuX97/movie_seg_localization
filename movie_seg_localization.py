import lmdb
import numpy as np
import torchvision
import torch
import io
import os.path as osp
import glob
import argparse
from tqdm import tqdm
import json
from pdb import set_trace as st
from sklearn.metrics.pairwise import cosine_similarity
from moviepy.editor import VideoFileClip
EPSILON = 1e-5
from datetime import timedelta

def parse_args():
    parser = argparse.ArgumentParser(description='automatic annotation')

    parser.add_argument('db_root', type=str)
    parser.add_argument('video_root', type=str)
    parser.add_argument('--out', type=str, default='results.json')

    args = parser.parse_args()
    return args


def load_lmdb_feature(txn, original_video_ids, shot_video_ids):
    # original_features = [np.reshape(txn.get(i.encode()), (-1, 1000)) for i in original_video_ids]
    # shot_features = [np.reshape(txn.get(i.encode()), (-1, 1000)) for i in shot_video_ids]
    original_features = np.frombuffer(txn.get(original_video_ids.encode()), dtype=np.float32)
    shot_features = np.frombuffer(txn.get(shot_video_ids.encode()), dtype=np.float32)
    return original_features, shot_features

def calculate_similarity(f1, f2):
    f1 = np.reshape(f1, (-1, 1000))
    f2 = np.reshape(f2, (-1, 1000))

    if f1.shape[1] != f2.shape[1]:
        raise ValueError

    cos_sim = cosine_similarity(f1, f2)

    return cos_sim


def write_result_dict(ret_dict, ori_id, sim_map, shot_id, duration):
    ori_length = sim_map.shape[0]
    window_size = sim_map.shape[1]
    start_idx = np.arange(0, ori_length - window_size)
    # n x N -> 1 x N
    sim_map = np.average(sim_map, axis=1)
    scores = [np.average(sim_map[i: (i + window_size)])for i in start_idx]
    best_start_id = np.argmax(scores)
    segment_dict = {'start' : str(timedelta(seconds=int(best_start_id / ori_length * duration))),
                    'end' : str(timedelta(seconds=int((best_start_id + window_size) / ori_length * duration))),
                    'score' : float(scores[best_start_id]),
                    'shot_id' : shot_id
                    }

    if ori_id in ret_dict:
        ret_dict[ori_id].append(segment_dict)
    else:
        ret_dict[ori_id] = list([segment_dict])

def _merge(ret_dict):
    for ori_id in ret_dict.keys():
        merged_result = []
        ret_dict[ori_id] = sorted(ret_dict[ori_id], key=lambda x:x['start'])
        for i, shot_i in enumerate(ret_dict[ori_id]):
            if len(merged_result) == 0 or merged_result[-1]['end'] < shot_i['start']:
                shot_i['shot_id'] = [shot_i['shot_id']]
                merged_result.append(shot_i)
            else:
                merged_result[-1]['end'] = max(merged_result[-1]['end'], shot_i['end'])
                merged_result[-1]['shot_id'].append(shot_i['shot_id'])
                # FIXME: scores is not true
        ret_dict[ori_id] = merged_result  

def parse_file_list(video_root):
    video_id_dirs = glob.glob(osp.join(video_root, '*'))
    video_ids = [osp.split(v_id_dir)[-1] for v_id_dir in video_id_dirs]
    ret_dict = dict()
    for v_id in video_ids:
        original_list = glob.glob(osp.join(video_root, v_id, 'original', '*'))
        shots_list = glob.glob(osp.join(video_root, v_id, 'shots', '*'))
        ret_dict[v_id] = {
            'original': [osp.splitext(osp.basename(i))[0] for i in original_list],
            'shots': [osp.splitext(osp.basename(i))[0] for i in shots_list]
        }
    return ret_dict

def movie_seg_localization():
    env = lmdb.open(args.db_root, map_size=int(1e9))
    original_shot_dict = parse_file_list(args.video_root)
    dict_out = dict()
    with env.begin() as txn:
        for k, v in original_shot_dict.items():
            # looping all TV-series
            for shot_id in v['shots']:
                # looping all shots
                max_sim = -1
                max_sim_ori_id = None
                for ori_id in v['original']:
                    # looping all episode of the series
                    ori_feature, shot_feature = load_lmdb_feature(txn, ori_id, shot_id)
                    sim = calculate_similarity(ori_feature, shot_feature)
                    if np.max(sim) >= max_sim:
                        max_sim = np.max(sim)
                        max_sim_map = sim
                        max_sim_ori_id = ori_id
                        # st()
                        ori_path = glob.glob(osp.join(args.video_root, '*', 'original', f"{ori_id}*"))[0]
                        clip_ori = VideoFileClip(ori_path)
                        ori_duration = clip_ori.duration
                write_result_dict(dict_out, max_sim_ori_id, max_sim_map, shot_id, ori_duration)
    _merge(dict_out)
    for k, v in dict_out.items():
        print(f'Original video {k} was annotated {len(v)} shots')
    with open(args.out, 'w') as f_out:
        json.dump(dict_out, f_out, indent=4)

if __name__ == '__main__':
    args = parse_args()
    movie_seg_localization()
import lmdb
import numpy as np
import torchvision
import torch
import io
import os.path as osp
import glob
import argparse
from tqdm import tqdm

from pdb import set_trace as st
from sklearn.metrics.pairwise import cosine_similarity


def parse_args():
    parser = argparse.ArgumentParser(description='automatic annotation')

    parser.add_argument('db_root', type=str)
    parser.add_argument('video_root', type=str)
    parser.add_argument('--out', type=str, default='results.json')

    args = parser.parse_args()
    return args


def load_lmdb_feature(txn, original_video_ids, shot_video_ids):
    original_features = [np.reshape(txn.get(i), (-1, 1000)) for i in original_video_ids]
    shot_features = [np.reshape(txn.get(i), (-1, 1000)) for i in shot_video_ids]

    return np.array(original_features), np.array(shot_features)

def calculate_similarity(f1, f2):
    
    if f1.shape[1] != f2.shape[1]:
        raise ValueError

    cos_sim = cosine_similarity(f1, f2)
    st()

    return cos_sim


def generate_result_dict(sim_map):
    window_size = sim_map.shape[0]
    start_idx = np.arange(0, sim_map.shape[1] - window_size)
    

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
    st()
    dict_out = dict()
    with env.begin() as txn:
        for k, v in original_shot_dict.items():
            original_features, shot_features = load_lmdb_feature(txn, v['original'], v['shots'])
            similarity_map = calculate_similarity(original_features, shot_features)
            st()



if __name__ == '__main__':
    args = parse_args()
    movie_seg_localization()
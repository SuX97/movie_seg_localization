import lmdb
import numpy as np
import torchvision
import torch
import glob
import argparse
# Read all video files and save their features in lmdb database

def read_video(video_path):

def extract_frame_features(video_data):

def dump_lmdb(video_id, db_path):
    txn = env.begin(write=True)
    txn.put(key = video_id, value = feature_data)
    txn.commit()
    txn.close()


def parse_args():
    parser = argparse.ArgumentParser(description='build feature databse')

    parser.add_argument('videos-root')
    parser.add_argument('result-root')

    args = parser.parse_args()
    return args

def build_feature_databse():
    args = parse_args()
    env = lmdb.open(arg.result_root, map_size=1099511627776) # 1TB
    segment_list = glob.glob(osp.join(args.videos_root, '*', 'segments'))
    origin_video_list = glob.glob(osp.join(args.videos_root, '*', 'original'))
    
    env.close()


if __name__ == '__main__':
    build_feature_databse()

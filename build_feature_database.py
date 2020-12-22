import lmdb
import numpy as np
import torchvision
import torch
import io
import os.path as osp
import glob
import argparse
from tqdm import tqdm
import torchvision.models as models
from pdb import set_trace as st
import torchsnooper
# from torchvideotransforms import video_transforms, volume_transforms
from mmaction.datasets.pipelines import (
                                        Resize,
                                        CenterCrop,
                                        Normalize,
                                        DecordInit,
                                        DecordDecode,
                                        FormatShape)

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)

shufflenet = models.shufflenet_v2_x1_0(pretrained=True)
shufflenet.eval()
shufflenet.cuda()

pipeline = [DecordInit(),
            DecordDecode(),
            Resize((-1, 256)),
            CenterCrop(256),
            Normalize(**img_norm_cfg),
            FormatShape(input_format='NCHW')]


def read_video(video_path):
    results = dict(filename=video_path, modality='RGB')
    for trans in pipeline:
        results = trans(results)
        if isinstance(trans, DecordInit):
            results['frame_inds'] = np.arange(1, results['total_frames'], 30)
    return results['imgs']

@torchsnooper.snoop()
def extract_frame_features(v_path):
    frame_feature = None
    vframes = read_video(v_path)
    vframes = torch.from_numpy(vframes).cuda()
    vframes = torch.split(vframes, 8)
    for i in tqdm(range(len(vframes))):
        test_frames = vframes[i]
        if frame_feature is not None:
            frame_feature = torch.cat((frame_feature, shufflenet(test_frames)), 0)
        else:
            frame_feature = shufflenet(test_frames)
    return np.array(frame_feature.detach().cpu())

def dump_lmdb(env, video_id, db_path, feature_data):
    with env.begin(write=True) as txn:
        txn.put(key = video_id.encode(), value = feature_data)


def parse_args():
    parser = argparse.ArgumentParser(description='build feature databse')

    parser.add_argument('videos_root', type=str)
    parser.add_argument('result_root', type=str)

    args = parser.parse_args()
    return args

def build_feature_databse():
    args = parse_args()
    env = lmdb.open(args.result_root, map_size=109951000) # 1TB
    segment_list = glob.glob(osp.join(args.videos_root, '*', 'segments', '*'))
    origin_video_list = glob.glob(osp.join(args.videos_root, '*', 'original', '*'))
    print(segment_list)
    print(origin_video_list)
    for v_path in segment_list:
        print(f'extracting {v_path}')
        v_feature = extract_frame_features(v_path)
        dump_lmdb(env, osp.basename(v_path), args.result_root, v_feature)

    env.close()


if __name__ == '__main__':
    build_feature_databse()

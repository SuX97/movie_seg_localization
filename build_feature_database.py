import lmdb
import numpy as np
import torchvision
import torch
import os.path as osp
import glob
import argparse
import torchvision.io as io
import torchvision.transforms as T
import torchvision.models as models
from pdb import set_trace as st

normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
transform = T.Compose([T.ToPILImage(),
                       T.Resize(256),
                       T.CenterCrop(224),
                       T.ToTensor(),
                       normalize])
shufflenet = models.shufflenet_v2_x1_0(pretrained=True)
shufflenet.eval()
shufflenet.cuda()


def read_video(video_path):
    vframes, aframes, info = io.read_video(video_path)
    return vframes, aframes, info


def extract_frame_features(v_path):
    frame_feature = None
    vframes, aframes, info = read_video(v_path)
    st()
    vframes = torch.split(vframes, 16)
    for i in range(len(vframes)):
        test_frames = vframes[i].cuda()
        if frame_feature:
            frame_feature = torch.cat((frame_feature, shufflenet(transform(test_frames))), 0)
        else:
            frame_feature = shufflenet(transform(test_frames))

    return np.array(frame_feature.detach().cpu())

def dump_lmdb(env, video_id, db_path, feature_data):
    txn = env.begin(write=True)
    txn.put(key = video_id, value = feature_data)
    txn.commit()
    txn.close()


def parse_args():
    parser = argparse.ArgumentParser(description='build feature databse')

    parser.add_argument('videos_root', type=str)
    parser.add_argument('result_root', type=str)

    args = parser.parse_args()
    return args

def build_feature_databse():
    args = parse_args()
    env = lmdb.open(args.result_root, map_size=109951) # 1TB
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

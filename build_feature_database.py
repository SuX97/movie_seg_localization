# build CNN feature for each videos
import lmdb
import numpy as np
import torchvision
import torch
import cv2
from torch.nn import DataParallel
import io
WITH_CAFFE = True

try:
    import caffe
except ImportError:
    WITH_CAFFE = False
    
import os.path as osp
import glob
import argparse
from tqdm import tqdm
import torchvision.models as models
from pdb import set_trace as st
# import torchsnooper
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
# shufflenet = DataParallel(shufflenet, divice_ids=[0, 1, 2, 3, 4, 5, 6, 7])
class CropBlackBorder:
    count = 0
    def __call__(self, results):
        def crop_image(img):
            h, w = img.shape[:2]
            h_top = int(h // 2 + 0.7 * (w * 9 / 16))
            h_bottom = int(h // 2 - 0.5 * (w * 9 / 16))
            crop = img[h_bottom : h_top, :, :]
            return crop

        # cv2.imwrite('before.jpg', results['imgs'][0])
        results['imgs'] = [crop_image(img) for img in results['imgs']]
        # st()
        # cv2.imwrite('after.jpg', results['imgs'][0])
        # CropBlackBorder.count += 1
        # if CropBlackBorder.count == 100:
        #     assert 1==2
        results['img_shape'] = results['imgs'][0].shape[:2]

        return results




def read_video(video_path, crop=False):
    results = dict(filename=video_path, modality='RGB')
    print(f'decoding {video_path}')
    if crop:
        pipeline = [DecordInit(num_threads=8),
                    DecordDecode(),
                    CropBlackBorder(),
                    Resize((-1, 256)),
                    CenterCrop(256),
                    Normalize(**img_norm_cfg),
                    FormatShape(input_format='NCHW')]
    else:
        pipeline = [DecordInit(num_threads=8),
                    DecordDecode(),
                    # CropBlackBorder(),
                    Resize((-1, 256)),
                    CenterCrop(256),
                    Normalize(**img_norm_cfg),
                    FormatShape(input_format='NCHW')]
    for trans in pipeline:
        results = trans(results)
        if isinstance(trans, DecordInit):
            # extract 1 / 10 frames
            results['frame_inds'] = np.arange(1, results['total_frames'], 10)


    return results['imgs']

# @torchsnooper.snoop()
def extract_frame_features(v_path, crop=False):
    frame_feature = None
    vframes = read_video(v_path, crop=crop)
    vframes = torch.from_numpy(vframes).cuda()
    # batchsize = 8
    vframes = torch.split(vframes, 32)
    for i in tqdm(range(len(vframes))):
        test_frames = vframes[i]
        if frame_feature is not None:
            frame_feature = torch.cat((frame_feature, shufflenet(test_frames).detach().cpu()), 0)
        else:
            frame_feature = shufflenet(test_frames).detach().cpu()
    return np.array(frame_feature)

def dump_lmdb(env, video_id, db_path, feature_data):
    # st()
    with env.begin(write=True) as txn:
        if WITH_CAFFE:
            datum = caffe.proto.caffe_pb2.Datum()
            datum.channels = feature_data.shape[1]
            datum.length = feature_data.shape[0]
            datum.data = feature_data.tobytes()
            txn.put(key = video_id.encode(), value = feature_data.SerializeToString())
        else:
            txn.put(key = video_id.encode(), value = feature_data)


def parse_args():
    parser = argparse.ArgumentParser(description='build feature databse')

    parser.add_argument('videos_root', type=str)
    parser.add_argument('result_root', type=str)

    args = parser.parse_args()
    return args

def build_feature_databse():
    args = parse_args()
    env = lmdb.open(args.result_root, map_size=int(1e9)) # 1GB
    shots_list = glob.glob(osp.join(args.videos_root, '*', 'shots', '*'))
    origin_video_list = glob.glob(osp.join(args.videos_root, '*', 'original', '*'))
    print(shots_list)
    print(origin_video_list)
    for v_path in shots_list:
        # print(f'extracting {v_path}')
        v_feature = extract_frame_features(v_path, crop=True)
        dump_lmdb(env, osp.splitext(osp.basename(v_path))[0], args.result_root, v_feature)

    for v_path in origin_video_list:
        # print(f'extracting {v_path}')
        v_feature = extract_frame_features(v_path)
        dump_lmdb(env, osp.splitext(osp.basename(v_path))[0], args.result_root, v_feature)

    env.close()


if __name__ == '__main__':
    build_feature_databse()

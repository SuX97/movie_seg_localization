# shot segment for movie segment reviews

import matplotlib.pyplot as plt
import sys
from scipy.signal import convolve2d
from skmultiflow.drift_detection import PageHinkley
from sklearn.preprocessing import MinMaxScaler
from scipy.optimize import leastsq
from scipy.optimize import curve_fit
import numpy as np
import math
import time
import cv2
import time
from PIL import Image  
import json
feature_params = dict( maxCorners = 100,
                    qualityLevel = 0.25,
                    minDistance = 7,
                    blockSize = 7)
# params for LK flow calculation
lk_params = dict( winSize  = (15,15),
                maxLevel = 2,
                criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
"""
Helper function
"""
def smooth(x, window_len=13, window='hanning'):
    """
    smooth the 1-D sequence data using a window with requested size.
    """
    #print(len(x), window_len)
    if window_len < 3:
        return x

    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError

    s = np.r_[2 * x[0] - x[window_len:1:-1],
              x, 2 * x[-1] - x[-1:-window_len:-1]]
    #print(len(s))

    if window == 'flat':  # moving average
        w = np.ones(window_len, 'd')
    else:
        w = getattr(np, window)(window_len)
    y = np.convolve(w / w.sum(), s, mode='same')
    return y[window_len - 1:-window_len + 1]

def filter2(x, kernel, mode='same'):
    """
    conv filter on 2d image
    """
    return convolve2d(x, np.rot90(kernel, 2), mode=mode)

def matlab_style_gauss2D(shape=(3,3),sigma=0.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma]), 高斯分布kernel用于SSIM
    """
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h

def detWithEssentialMat(_frame_diffs, _time_lists, _i, _fps, _curr_frame, _prev_frame, _p0 = None):
    """
    EssentialMat reasons the movement of camera, Pa * Eab * Pb = 0 (constrain)
    Eab = (Cab)^T * r, Tba(pose change) = [[Cab, r], [O^T, 1]]
    By using the package from cv2 
    """
    height = _curr_frame.shape[0]
    width = _curr_frame.shape[1]
    if _p0 is None:
        _p0 = cv2.goodFeaturesToTrack(_prev_frame, mask = None, **feature_params)
    else:
        if _i % 2 == 0:
            #print("DEBUG: recalculate features")
            _p0 = cv2.goodFeaturesToTrack(_prev_frame, mask = None, **feature_params)
    p1, st, err = cv2.calcOpticalFlowPyrLK(_prev_frame, _curr_frame, _p0, None, **lk_params)
    if p1 is not None:
        # st means state, if 1 then the corner point exists in the new frame，有效的角点
        good_new = p1[st == 1]
        good_old = _p0[st == 1]
        print("{} feature points detected".format(good_new.size))
        if good_new.size > 5:     
            E_5point, mask1 = cv2.findEssentialMat(good_old, good_new, method=cv2.RANSAC, threshold=0.9)
            if mask1 is not None and E_5point is not None and E_5point.shape[0] == 3:
                #print("DEBUG: size {}".format(E_5point.shape))
                points, R, t, mask2 = cv2.recoverPose(E_5point, good_old, good_new, mask=mask1.copy())
                #print(t)
                _time_lists.append(float(_i / _fps))
                _p0 = good_new.reshape(-1, 1, 2)
                # Translation vector calculate norm
                t_norm = np.linalg.norm(t[2])
                _frame_diffs.append(t_norm)
                return t_norm, _p0
    return 0, _p0


def detWithLUV(_frame_diffs, _time_lists, _i, _fps, _curr_frame, _prev_frame):
    """
        LUV color space can do better to distinguish the difference between pictures
    """
    diff = 0
    diff = cv2.absdiff(_curr_frame, _prev_frame)
    count = np.sum(diff)
    _frame_diffs.append(count)
    _time_lists.append(float(_i / _fps))

    return count

def compute_ssim(im1, im2, k1=0.01, k2=0.03, win_size=11, L=255):
    if not im1.shape == im2.shape:
        raise ValueError("Input Images must have the same dimensions")
    if len(im1.shape) > 2:
        raise ValueError("Please input the images with 1 channel， shape:{}".format(im1.shape))

    M, N = im1.shape
    C1 = (k1*L)**2
    C2 = (k2*L)**2
    window = matlab_style_gauss2D(shape=(win_size,win_size), sigma=1.5)
    window = window/np.sum(np.sum(window))

    if im1.dtype == np.uint8:
        im1 = np.double(im1)
    if im2.dtype == np.uint8:
        im2 = np.double(im2)

    mu1 = filter2(im1, window, 'valid')
    mu2 = filter2(im2, window, 'valid')
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = filter2(im1*im1, window, 'valid') - mu1_sq
    sigma2_sq = filter2(im2*im2, window, 'valid') - mu2_sq
    sigmal2 = filter2(im1*im2, window, 'valid') - mu1_mu2

    ssim_map = ((2*mu1_mu2+C1) * (2*sigmal2+C2)) / ((mu1_sq+mu2_sq+C1) * (sigma1_sq+sigma2_sq+C2))
    #ssim_s = ( sigmal2 + C2/2 ) / (sigma1_sq)
    ret = np.mean(np.mean(ssim_map))
    print("SSIM: {:.4f}".format(ret))
    return ret

def detWithSSIM(_frame_diffs, _time_lists, _i, _fps, _curr_frame, _prev_frame):
    '''
        The SSIM_S denotes the structral similarity between two pic: curr_frame, prev_frame
    '''
    ssim = 0
    ssim = compute_ssim(_curr_frame, _prev_frame)
    _frame_diffs.append(ssim)
    _time_lists.append(float(_i / _fps))

    return ssim

def findCutPointsPairs(sequence):
    '''
    Detect changing edge based on PageHinkley algorithm
    '''
    ret = []
    ph = PageHinkley()
    size = len(sequence)
    data_stream = np.array(sequence)
    # Adding stream elements to the PageHinkley drift detector and verifying if drift occurred
    for i in range(size):
        ph.add_element(data_stream[i])
        if ph.detected_change():
            #print('Change has been detected in data: ' + str(data_stream[i]) + ' - of index: ' + str(i))
            ret.append(i / size)
    return ret



def shot_segment(videofile, target_dir):
    """
    Detect Pan/Tilt/Zoom camera motion
    """
    # display images for debugging/troubleshooting
    visualize = False
    useLUV = True
    useSSIM = False
    useOF = False
    useE = False
    useInt = False
    # frames per second (skip other frames)
    sampling_rate = 1
    cap = cv2.VideoCapture(videofile)
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    fps = cap.get(cv2.CAP_PROP_FPS)
    totFrames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    videoLength = float(totFrames) / float(fps)

    # if unavailable, by default 30.0
    if fps <= 0.0 or math.isnan(fps):
        fps = 30.0
    color = np.random.randint(0,255,(200,3))
    curr_frame = None
    prev_frame = None
    frame_diffs = []
    time_lists = []
    save_lists = []
    ret, frame = cap.read()
    # We may use resize to save time
    #frame = cv2.resize(frame, (400, 400), interpolation=cv2.INTER_CUBIC)
    height = frame.shape[0]
    width = frame.shape[1]
    curr_frame = frame
    count_time_list = []
    i = 1.0
    count = 0
    a = 0
    b= 0
    c= 0
    p0 = None # 初始特征点
    while(ret):
        save_lists.append(curr_frame)
        startTime = time.clock()
        curr_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2LUV)
        curr_frame = cv2.GaussianBlur(frame, (5, 5), 1.5)

        if curr_frame is not None and prev_frame is not None and i % sampling_rate == 0:
            count = detWithLUV(frame_diffs, time_lists, i, fps, curr_frame, prev_frame)
            # if useSSIM:
            #     count = detWithSSIM(frame_diffs, time_lists, i, fps, curr_frame, prev_frame)
        count_time_list.append(time.clock() - startTime)
        prev_frame = curr_frame
        i = i + 1
        ret, frame = cap.read()

    frame_diffs = np.array(frame_diffs)
    frame_diffs = smooth(frame_diffs, window_len=4)
    cutPoints = []
    cutPoints = findCutPointsPairs(frame_diffs)
    #print("Cut points at" )
    '''
    for i in cutPoints:
        print("{:.2f}".format(i * videoLength))
    '''
    cap.release()
    for i, cp in enumerate(cutPoints):
        shot_path = target_dir + 'shot_{i}.mp4' 
        out = cv2.VideoWriter(shot_path, cv2.VideoWriter_fourcc(fourcc), fps, (height, width))
        out.write(save_lists[int(cp * len(save_lists))])
    out.release()
    '''
    for i in range(len(save_lists)):
        out.write(save_lists[i])
    out.release()
    '''    
    print("Time per frame: {}".format(np.mean(count_time_list)))


def main():
    if len(sys.argv) > 1:
        video = sys.argv[1]
        target_dir = sys.argv[2]
    else:
        print("Video file must be specified.")
        sys.exit(-1)
    start = time.time()
    shot_segment(video, target_dir)
    end = time.time()
    print("Running time: {:.4f}".format(end - start))
if __name__ == "__main__":
    main()

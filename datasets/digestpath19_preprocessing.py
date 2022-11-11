# library
import pandas as pd
import os
from tqdm import tqdm
import cv2
import argparse

# local
# from ..utils import create_dir


parser = argparse.ArgumentParser()

parser.add_argument('--size', default=1500, type=int,
                    help='패치 사이즈')
parser.add_argument('--white_space_ratio','--w', default=0.75, type=float,
                    help='patch에서 white space 비율이 얼마 이상이면 버릴건지?')
args = parser.parse_args()


def main():
    # create_dir(dir = 'E:/DigestPath2019/colon_patched')
    neg_list = os.listdir('E:/DigestPath2019/Colonoscopy_tissue_segment_dataset/tissue-train-neg')
    pos_list = []
    for i in os.listdir('E:/DigestPath2019/Colonoscopy_tissue_segment_dataset/tissue-train-pos-v1'):
        if 'mask' not in i:
            pos_list.append(i)
    neg_df = pd.DataFrame(data={'name':neg_list})
    pos_df = pd.DataFrame(data={'name':pos_list})

    # negative
    for name in tqdm(neg_list, desc='Negative Sample...'):
        image = cv2.imread('E:/DigestPath2019/Colonoscopy_tissue_segment_dataset/tissue-train-neg/' + name, cv2.IMREAD_COLOR)
        shape = image.shape
        h, w = shape[0], shape[1]
        assert (h>=args.size or w>=args.size), f"오류 : Patch size {args.size}가 높이 {h}나 너비 {w}보다 큽니다"
        h_mot, h_namage = h//args.size, h%args.size
        w_mot, w_namage = w//args.size, w%args.size
        if h_namage > 100:
            h_overlap = (h_mot+1) * args.size - h
            h_jump = h_overlap//h_mot
        else:
            h_jump = args.size
        if w_namage > 100:
            w_overlap = (w_mot+1) * args.size - w
            w_jump = w_overlap//w_mot
        else:
            w_jump = args.size
        for h_index in range(h_mot):
            for w_index in range(w_mot):
                patch = image[h_index*h_jump:h_index*h_jump+args.size,
                              w_index*w_jump:w_index*w_jump+args.size,
                              ...]
                hsv_patch = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
                # save
    # positive
    for name in tqdm(pos_list, desc='Positive Sample...'):
        break


if __name__ == '__main__':
    main()
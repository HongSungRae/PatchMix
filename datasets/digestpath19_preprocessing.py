# library
import pandas as pd
import os
from tqdm import tqdm
import cv2
import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

# local
dir = os.path.realpath(__file__)
dir = os.path.abspath(os.path.join(dir, os.pardir, os.pardir))
sys.path.append(dir)
from utils import create_dir


parser = argparse.ArgumentParser()

parser.add_argument('--size', default=1024, type=int,
                    help='패치 사이즈')
parser.add_argument('--threshold','--t', default=0.75, type=float,
                    help='patch에서 white space 비율이 얼마 이상이면 버릴건지?')
parser.add_argument('--namage','--n', default=300, type=int,
                    help='patch하고 남은 w,h가 얼마나 되면 겹치는 한이 있어도 살려갈건가?')
args = parser.parse_args()



def break2pieces(image_list, size, namage, threshold, positive=False):
    '''
    = 설명 =
    해당 경로의 이미지를 white space 규칙에 따라서 patch 단위로 자른 뒤, 지정된 경로에 저장함.
    잘려진 이미지의 정보를 알려주는 dataframe을 저장함
    = Input =
    
    = Output = 
    '''
    # 초기 설정
    name_list = []
    num_list = []
    if positive:
        pn = 'pos'
        folder = 'tissue-train-pos-v1'
    else:
        pn = 'neg'
        folder = 'tissue-train-neg' 

    for name in tqdm(image_list, desc=f'{pn} Sample...'):
        ## 이미지 로드 후, 조건 검사
        image = cv2.imread(f'E:/DigestPath2019/Colonoscopy_tissue_segment_dataset/{folder}/' + name, cv2.IMREAD_COLOR)
        h, w, _ = image.shape
        if (h<=size or w<=size):
            print(f'{name}은 size가 {h}x{w}로, 제외되었습니다.')
            continue

        ## 몫과 나머지(몇 개의 패치? 얼마나 점프?)
        h_mot, h_namage = h//size, h%size
        w_mot, w_namage = w//size, w%size

        ## 나머지가 적당히 있다면 겹치더라도 패치를 더 잘라준다
        if h_namage > namage:
            h_overlap = ((h_mot+1) * size - h)//h_mot
            h_mot += 1
        else:
            h_overlap = 1
        
        if w_namage > namage:
            w_overlap = ((w_mot+1) * size - w)//w_mot
            w_mot += 1
        else:
            w_overlap = 1

        ## 규칙에 따라서 자르기
        num = 0
        for h_index in range(h_mot):
            for w_index in range(w_mot):
                patch = image[h_index*(size-h_overlap):h_index*(size-h_overlap)+size,
                              w_index*(size-w_overlap):w_index*(size-w_overlap)+size,
                              ...]
                hsv_patch = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
                white_space = cv2.inRange(hsv_patch,(0, 0, 255*0.9),(360, 255*0.12, 255)) # https://ko.rakko.tools/tools/30/
                if np.sum(white_space/255)/(size**2) < threshold:
                    cv2.imwrite(f'E:/DigestPath2019/colon_patched/{pn}/{name}_{num}.png', patch)
                    name_list.append(name)
                    num_list.append(num)
                    if positive:
                        mask = cv2.imread(f'E:/DigestPath2019/Colonoscopy_tissue_segment_dataset/tissue-train-pos-v1/{name.split(".")[0]}_mask.jpg', cv2.IMREAD_GRAYSCALE)
                        patch = mask[h_index*(size-h_overlap):h_index*(size-h_overlap)+size,
                                     w_index*(size-w_overlap):w_index*(size-w_overlap)+size]
                        cv2.imwrite(f'E:/DigestPath2019/colon_patched/pos/{name}_{num}_mask.png', patch)
                else:
                    cv2.imwrite(f'E:/DigestPath2019/colon_patched/{pn}_white/{name}_{num}.png', patch)
                num += 1
    else:
        df = pd.DataFrame(data={'name':name_list, 'num':num_list})
        df.to_csv(f"E:/DigestPath2019/colon_patched/{pn}.csv", index=False)
        del df, name_list, num_list


def main():
    # 저장될 경로 생성
    create_dir('E:/DigestPath2019/colon_patched')
    create_dir('E:/DigestPath2019/colon_patched/neg')
    create_dir('E:/DigestPath2019/colon_patched/neg_white')
    create_dir('E:/DigestPath2019/colon_patched/pos')
    create_dir('E:/DigestPath2019/colon_patched/pos_white')

    # 이미지 경로를 담은 list 생성
    neg_image_list = os.listdir('E:/DigestPath2019/Colonoscopy_tissue_segment_dataset/tissue-train-neg')
    pos_image_list = []
    pos_mask_list = []
    for directory in os.listdir('E:/DigestPath2019/Colonoscopy_tissue_segment_dataset/tissue-train-pos-v1'):
        if 'mask' in directory:
            pos_mask_list.append(directory)
        else:
            pos_image_list.append(directory)

    # negative 이미지 패치아웃
    # break2pieces(image_list=neg_image_list, 
    #              size=args.size,
    #              namage=args.namage, 
    #              threshold=args.threshold, 
    #              positive=False)
    # positive 이미지 패치아웃
    break2pieces(image_list=pos_image_list, 
                 size=args.size,
                 namage=args.namage, 
                 threshold=args.threshold, 
                 positive=True)


if __name__ == '__main__':
    main()
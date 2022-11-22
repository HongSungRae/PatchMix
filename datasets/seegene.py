import cv2
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch
import random
import os
import sys
import time
from multiprocessing import Manager

dir = os.path.realpath(__file__)
dir = os.path.abspath(os.path.join(dir, os.pardir, os.pardir))
sys.path.append(dir)

# seed
# random.seed(1009)

# local
from augmentations.cp import _cp_poisson, cp_poisson, cp_gaussian, cp_tumor, cp_simple
from augmentations.cutmix import cutmix_dice, cutmix_half, cutmix_random, cutmix_random_tumor, cutmix_random_poisson



class Seegene(Dataset):
    def __init__(self, cache=None, split='train', dataset='NM', augmentation=None, aug_p=0.5, ratio=None, size=256):
        assert split in ['train', 'test', 'validation'], "Seegene split Error"
        assert dataset in ['M','NM'], "Seegene dataset Error"
        assert augmentation in ['cp_naive', 'cp_simple', 'cp_gaussian', 'cp_poisson', 'cp_tumor', 
                                'cutmix_half', 'cutmix_dice', 'cutmix_random', 'cutmix_random_gaussian', 'cutmix_random_simple', 'cutmix_random_tumor', 'cutmix_random_poisson',
                                'image_aug', None], "Seegene augmentation Error"
        self.cache = cache
        self.path = 'D:/seegene/data/segmentation/preprocessed'
        self.split = split
        self.dataset = dataset
        self.augmentation = augmentation
        if ratio != None:
            self.aug_p = 0.0
        else: self.aug_p = aug_p
        self.ratio = ratio
        self.size = size
        self.is_image_M = True

        # define dataset dataframe
        if dataset == 'NM':
            self.df_N = pd.read_csv(self.path + '/N_'+split+'.csv')
            self.df_M = pd.read_csv(self.path + '/M_'+split+'.csv')
            self.df = pd.concat([self.df_M,self.df_N], ignore_index=True) # ignore_index : 행 번호를 0부터 다시 쭈욱
        elif dataset == 'M':
            self.df = pd.read_csv(self.path + '/M_'+split+'.csv')
        self.df = self.df.sample(frac=1).reset_index(drop=True) # shuffle
        self.length = len(self.df)

        # define augmentation
        self.normal_transform = A.Compose([A.Resize(size,size),
                                           ToTensorV2()])
        if augmentation in ['cp_simple', 'cutmix_random_simple']:
            self.transform = A.Compose([A.RandomScale(scale_limit=(-0.9, 1), p=1), # (-1,0)에서 크기가 줄고 (0,2)에서 크기가 원본보다 커진다
                                        A.PadIfNeeded(size, size, border_mode=cv2.BORDER_CONSTANT), # 원래는 0 이었다
                                        A.HorizontalFlip(),
                                        A.RandomCrop(size, size),
                                        ToTensorV2()])
        elif augmentation == 'cp_naive':
            self.transform = self.normal_transform
        elif augmentation in ['cp_tumor', 'cutmix_random_tumor']:
            self.transform = A.Compose([A.Resize(size,size)]) # cp_naive, cp_tumor는 numpy.ndarray를 입력받음
        else: # ['cp_gaussian', 'cp_poisson', 'cutmix_half', 'cutmix_dice', 'cutmix_random', 'image_aug']
            self.transform = A.Compose([# Simple Aug
                                        A.HorizontalFlip(p=0.33),
                                        A.VerticalFlip(p=0.33),
                                        # A.Cutout(max_h_size=16, max_w_size=16, p=0.33),
                                        # A.GaussNoise(p=0.33),
                                        # # Rigid Geometric Aug
                                        # A.GridDistortion(num_steps=4, p=0.33),
                                        # # Color Aug
                                        # A.RandomBrightness(p=0.33),
                                        # A.RandomContrast(p=0.33),
                                        # # Processing
                                        A.Resize(size,size),
                                        ToTensorV2()
                                        ])

    def __len__(self):
        if self.ratio != None:
            return (self.length + int(self.length*self.ratio))
        else:
            return self.length

    def _get_img_mask(self,pat_id,file_id, transform):
        try: # M data
            image = self.cache.get(f'{self.path}/M/{pat_id}_{file_id}.png', None)
            mask = self.cache.get(f"{self.path}/M/{pat_id}_{file_id}_mask.png", None)
            if (image is None) or (mask is None):
                image = cv2.imread(f'{self.path}/M/{pat_id}_{file_id}.png', cv2.IMREAD_COLOR) # (1504,2056,3), np.ndarray
                mask = cv2.imread(f"{self.path}/M/{pat_id}_{file_id}_mask.png", cv2.IMREAD_GRAYSCALE) # (1504,2056), np.ndarray
                resize = A.Resize(self.size,self.size)(image=image, mask=mask) # resize for caching. (1504, 2056) is too big to cache
                self.cache[f'{self.path}/M/{pat_id}_{file_id}.png'] = resize['image']
                self.cache[f"{self.path}/M/{pat_id}_{file_id}_mask.png"] = resize['mask']
            sample = transform(image=image, mask=mask)
            image, mask = sample["image"]/255, sample["mask"]/255 # (3,size,size) (size,size)
            try:
                mask = torch.stack((torch.ones_like(mask) - mask, mask),dim=-1)
            except:
                mask = np.stack((np.ones_like(mask) - mask, mask), axis=-1).astype(np.float64) # (size, size, 2)
        except: # N data
            image = self.cache.get(f"{self.path}/N/{pat_id}_{file_id}.png", None)
            if image is None:
                image = cv2.imread(f"{self.path}/N/{pat_id}_{file_id}.png", cv2.IMREAD_COLOR)
                resize = A.Resize(self.size,self.size)(image=image) # resize for caching. (1504, 2056) is too big to cache
                self.cache[f"{self.path}/N/{pat_id}_{file_id}.png"] = resize['image']
            sample = transform(image=image)
            image = sample["image"]/255 # (3, size, size)
            if self.augmentation in ['cp_tumor', 'cutmix_random_tumor']:
                mask_0 = np.ones((self.size,self.size,1))
                mask_1 = np.zeros((self.size,self.size,1))
                mask = np.concatenate((mask_0,mask_1),axis=-1)
            else:
                mask_0 = torch.ones((self.size,self.size,1))
                mask_1 = torch.zeros((self.size,self.size,1))
                mask = torch.cat((mask_0,mask_1),dim=-1)
            self.is_image_M = False
        return image, mask # (3,size,size) (size,size,2)


    def __getitem__(self, idx):
        if (idx > self.length) and (self.ratio != None):
            self.aug_p == 1.0
            idx = random.randint(0, self.length-1)
        if (random.random() >= self.aug_p) or (self.split in ['test', 'validation']) or (self.augmentation == None):
            pat_id, file_id = self.df.iloc[idx]
            img, mask = self._get_img_mask(pat_id, file_id, self.normal_transform)
        elif self.augmentation == 'image_aug': # & random.random() < self.aug_p
            pat_id, file_id = self.df.iloc[idx]
            img, mask = self._get_img_mask(pat_id, file_id, self.transform)
        else: #img_1 : main(붙임당할 이미지) , img_2 : paste(붙일 이미지)
            pat_id_1, file_id_1 = self.df.iloc[idx]
            if self.dataset == 'NM' and 'cp' in self.augmentation: # cp_NM의 경우, img_2는 무조건 M이여야한다
                pat_id_2, file_id_2 = self.df_M.iloc[random.randint(0, len(self.df_M)-1)]
            else:
                pat_id_2, file_id_2 = self.df.iloc[random.randint(0, self.__len__()-1)]
            img_1, mask_1 = self._get_img_mask(pat_id_1, file_id_1, self.transform)
            img_2, mask_2 = self._get_img_mask(pat_id_2, file_id_2, self.transform)
            if self.augmentation == 'cutmix_half':
                img, mask = cutmix_half(img_1, img_2, mask_1, mask_2)
            elif self.augmentation == 'cutmix_dice':
                img, mask = cutmix_dice(img_1, img_2, mask_1, mask_2)
            elif self.augmentation == 'cutmix_random':
                img, mask = cutmix_random(img_1, img_2, mask_1, mask_2)
            elif self.augmentation == 'cutmix_random_simple':
                img, mask = cutmix_random(img_1, img_2, mask_1, mask_2)
            elif self.augmentation == 'cutmix_random_gaussian':
                img, mask = cutmix_random(img_1, img_2, mask_1, mask_2, gaussian_blur=True)
            elif self.augmentation == 'cutmix_random_tumor':
                img, mask = cutmix_random_tumor(img_1, img_2, mask_1, mask_2, size=self.size)
            elif self.augmentation == 'cutmix_random_poisson':
                img, mask = cutmix_random_poisson(img_1, img_2, mask_1, mask_2)
            elif self.augmentation == 'cp_naive':
                img, mask = cp_simple(img_1, img_2, mask_1, mask_2, gaussian_blur=False)
            elif self.augmentation == 'cp_simple':
                img, mask = cp_simple(img_1, img_2, mask_1, mask_2)
            elif self.augmentation == 'cp_gaussian':
                img, mask = cp_gaussian(img_1, img_2, mask_1, mask_2)
            elif self.augmentation == 'cp_poisson':
                img, mask = cp_poisson(img_1, img_2, mask_1, mask_2)
                # img, mask = _cp_poisson(tar_img_path = f'{self.path}/M/{pat_id_1}_{file_id_1}.png', 
                #                         tar_mask_path = f'{self.path}/M/{pat_id_1}_{file_id_1}_mask.png', 
                #                         src_img_path = f'{self.path}/M/{pat_id_2}_{file_id_2}.png', 
                #                         src_mask_path = f'{self.path}/M/{pat_id_2}_{file_id_2}_mask.png')
            elif  self.augmentation == 'cp_tumor':
                img, mask = cp_tumor(img_1, img_2, mask_1, mask_2, self.size)
        try:
            return img.float(), mask.float()
        except:
            # img = np.einsum('...c->c...', img)
            # return torch.from_numpy(img).float(), torch.from_numpy(mask).float()
            return img.float(), torch.from_numpy(mask).float()



if __name__ == '__main__':
    manager = Manager()
    img_cache = manager.dict()
    for augmentation in ['cp_naive', 'cp_simple', 'cp_gaussian', 'cp_tumor', # 'cp_poisson' 
                         'cutmix_half', 'cutmix_dice', 'cutmix_random', 'cutmix_random_gaussian', 'cutmix_random_simple', 'cutmix_random_tumor', # 'cutmix_random_poisson',
                         'image_aug', None]:
        for d in ['M', 'NM']:
            dataset = Seegene(split='train', cache=img_cache, dataset=d, augmentation=augmentation, aug_p=0.5)
            dataloader = DataLoader(dataset, 16, shuffle=False)
            start = time.time()
            image, mask = next(iter(dataloader))
            print(f'{d}/{augmentation} | Image : {image.shape} | Mask : {mask.shape} | {(time.time()-start):.4f} 초')
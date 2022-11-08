# libraries
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import glob
import cv2
import random
import os
import numpy as np
import re
import sys
dir = os.path.realpath(__file__)
dir = os.path.abspath(os.path.join(dir, os.pardir, os.pardir))
sys.path.append(dir)

# local
from augmentations.cp import _cp_poisson, cp_poisson, cp_gaussian, cp_tumor, cp_simple
from augmentations.cutmix import cutmix_dice, cutmix_half, cutmix_random
from parameters import SEED


class BUSI(Dataset):
    def __init__(self, split='train', dataset=None, augmentation=None, aug_p=0.5, ratio=None, size=256):
        assert split in ['train', 'test', 'validation'], "BUSI split Error"
        assert dataset in ['benign', 'malignant', 'normal'], "BUSI dataset Error"
        assert augmentation in ['cp_simple', 'cp_gaussian', 'cp_poisson', 'cp_tumor', 'cutmix_half', 'cutmix_dice', 'cutmix_random', 'image_aug', None], "BUSI augmentation Error"
        '''
        None : 어떠한 augmentation도 없는 생 데이터
        image_aug : 저수준의 image augmentation만 입한 상태 
        '''
        self.split = split
        self.dataset = dataset
        self.augmentation = augmentation
        self.aug_p = aug_p
        self.ratio = ratio
        self.size = size
        self.path = 'E:/BUSI/' + dataset
        self.img_list = glob.glob(self.path + '/*(*).png')
        random.seed(SEED)
        random.shuffle(self.img_list) # randon.seed에 의해서 고정적
        random.seed(None)
        if self.split == 'train':
            self.img_list = self.img_list[0:int(0.7*len(self.img_list))]
        elif self.split == 'test':
            self.img_list = self.img_list[int(0.7*len(self.img_list)):int(0.85*len(self.img_list))]
        else:
            self.img_list = self.img_list[int(0.85*len(self.img_list)):]
        self.legnth = len(self.img_list)
        
        self.normal_transform = A.Compose([A.Resize(size,size),
                                           ToTensorV2()])
        if augmentation == 'cp_simple':
            self.transform = A.Compose([A.RandomScale(scale_limit=(-0.9, 1), p=1), # (-1,0)에서 크기가 줄고 (0,2)에서 크기가 원본보다 커진다
                                        A.PadIfNeeded(size, size, border_mode=cv2.BORDER_CONSTANT), # 원래는 0 이었다
                                        A.HorizontalFlip(),
                                        A.RandomCrop(size, size),
                                        ToTensorV2()])
        elif augmentation == 'cp_tumor':
            self.transform = A.Compose([A.Resize(size,size)]) # cp_tumor는 numpy.ndarray를 입력받음
        else:
            self.transform = A.Compose([# A.RandomScale(scale_limit=(-0.3, 1), p=1),
                                        # A.PadIfNeeded(size, size, border_mode=0),
                                        # A.RandomResizedCrop(size, size,(0.7,1),p=0.3),
                                        # A.HorizontalFlip(p=0.5),
                                        # A.VerticalFlip(p=0.5),
                                        A.Resize(size,size),
                                        ToTensorV2()
                                        ])

    def __len__(self):
        return self.length

    def _get_img_mask(self, img_path, transform):
        # image Input
        img = cv2.imread(img_path,cv2.IMREAD_COLOR)/255
        # mask Input
        num = re.findall('\d+',img_path)[0]
        mask_list = [i for i in os.listdir(self.path) if re.search(f"\({num}\)_mask_*\d*.png", i)]
        mask = np.zeros(img.shape[0:2], dtype=np.float32)
        for name in mask_list:
            path = os.path.join(self.path, name)
            mask += cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        mask = np.where(mask>0, 255, mask)
        sample = transform(image=img, mask=mask)
        img, mask = sample['image'], sample['mask']
        try:
            mask = torch.where(mask>0, 1, 0)
            mask = torch.stack((torch.ones_like(mask) - mask, mask), dim=-1)
        except:
            mask = np.where(mask>0, 1, 0)
            mask = np.stack((np.ones_like(mask) - mask, mask), axis=-1).astype(np.float64)#.astype(np.uint8)
        del sample
        return img, mask

    def __getitem__(self, idx):
        if self.ratio != None:
            idx = random.random(0, len(self.img_list)-1)
        if (random.random() >= self.aug_p) or (self.split in ['test', 'validation']) or (self.augmentation == None):
            img, mask = self._get_img_mask(self.img_list[idx], self.normal_transform)
        elif self.augmentation == 'image_aug':
            img, mask = self._get_img_mask(self.img_list[idx], self.transform)
        else: #img_1 : main(붙임당할 이미지) , img_2 : paste(붙일 이미지)
            img_1_path, img_2_path = random.sample(self.img_list, 2) # random.seed의 영향없이 무작위
            img_1, mask_1 = self._get_img_mask(os.path.join(img_1_path), self.transform)
            img_2, mask_2 = self._get_img_mask(os.path.join(img_2_path), self.transform)
            # print(torch.any(mask_1[...,1]), torch.any(mask_2[...,1]))
            if self.augmentation == 'cutmix_half':
                img, mask = cutmix_half(img_1, img_2, mask_1, mask_2)
            elif self.augmentation == 'cutmix_dice':
                img, mask = cutmix_dice(img_1, img_2, mask_1, mask_2)
            elif self.augmentation == 'cutmix_random':
                img, mask = cutmix_random(img_1, img_2, mask_1, mask_2)
            elif self.augmentation == 'cp_simple':
                img, mask = cp_simple(img_1, img_2, mask_1, mask_2)
            elif self.augmentation == 'cp_gaussian':
                img, mask = cp_gaussian(img_1, img_2, mask_1, mask_2)
            elif  self.augmentation == 'cp_tumor':
                img, mask = cp_tumor(img_1, img_2, mask_1, mask_2)
            elif self.augmentation == 'cp_poisson':
                img, mask = cp_poisson(img_1, img_2, mask_1, mask_2, self.size, 'import')
        try:
            return img.float(), mask.float()
        except:
            img = np.einsum('...c->c...', img)
            return torch.from_numpy(img).float(), torch.from_numpy(mask).float()


if __name__ == '__main__':
    for augmentation in ['cp_simple', 'cp_gaussian', 'cp_tumor', 'cutmix_half', 'cutmix_dice', 'cutmix_random', 'image_aug', None]: # cp_poisson 없음
        print(f'=== aug : {augmentation} ===')
        busi = BUSI(split='train', dataset='malignant', augmentation=augmentation, aug_p=1)
        dataloader = DataLoader(busi, batch_size=12, shuffle=True)
        img, mask = next(iter(dataloader))
        print(f'image : {img.shape}, dtype : {type(img)}')
        print(f'mask : {mask.shape}, dtype : {type(mask)}')
        
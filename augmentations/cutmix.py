'''
input : img (3, size, size) -> numpy.ndarray or torch.Tensor
        mask (size, size, num_classes) -> numpy.ndarray or torch.Tensor
        size : size of img&mask
output : cutmix augmented img and mask
         img (3, size, size) -> numpy.ndarray or torch.Tensor
         mask (size, size, num_classes) -> numpy.ndarray or torch.Tensor
'''
import random
from skimage.filters import gaussian
import numpy as np
import torch
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import cv2

# local
try:
    from cp import find_channel_to_attach, cp
    import poissonimageediting as pie
except:
    from augmentations.cp import find_channel_to_attach, cp
    import augmentations.poissonimageediting as pie

# seed
random.seed(None)


def cutmix_half(img_1, img_2, mask_1, mask_2):
    size = mask_1.shape[1]
    half = int(size/2)
    if random.random() >= 0.5:
        img_1[:,:,0:half] = img_2[:,:,0:half]
        mask_1[:,0:half,:] = mask_2[:,0:half,:]
    else:
        img_1[:,0:half,:] = img_2[:,0:half,:]
        mask_1[0:half,:,:] = mask_2[0:half,:,:]
    return img_1, mask_1



def cutmix_dice(img_1, img_2, mask_1, mask_2):
    size = mask_1.shape[1]
    half = int(size/2)
    if random.random() >= 0.5:
        img_1[:,0:half:,half:] = img_2[:,0:half:,half:]
        img_1[:,half:,0:half] = img_2[:,half:,0:half]
        mask_1[0:half:,half:,:] = mask_2[0:half:,half:,:]
        mask_1[half:,0:half,:] = mask_2[half:,0:half,:]     
    else:
        img_1[:,0:half,0:half] = img_2[:,0:half,0:half]
        img_1[:,half:,half:] = img_2[:,half:,half:]
        mask_1[0:half,0:half,:] = mask_2[0:half,0:half,:]
        mask_1[half:,half:,:] = mask_2[half:,half:,:]
        
    return img_1, mask_1



def cutmix_random(img_1, img_2, mask_1, mask_2, num_holes:int=3, min_max:list=[0.1,0.4], gaussian_blur=False, sigma=0.4):
    _, h, w = img_1.shape
    if gaussian_blur:
        alpha = np.zeros_like(mask_2)
        for _ in range(num_holes):
            ratio = random.random() * (min_max[1]-min_max[0]) + min_max[0]
            coordinate = (random.randint(0,h-int(h*ratio)-1), 
                          random.randint(0,w-int(w*ratio)-1)) # coordinate of left-top size of the patch
            alpha[coordinate[0]:coordinate[0]+int(h*ratio),
                  coordinate[1]:coordinate[1]+int(w*ratio), :] = 1.0
            mask_1[coordinate[0]:coordinate[0]+int(h*ratio),
                   coordinate[1]:coordinate[1]+int(w*ratio),
                                                           :] = mask_2[coordinate[0]:coordinate[0]+int(h*ratio),
                                                                       coordinate[1]:coordinate[1]+int(w*ratio),
                                                                       :]
        alpha = gaussian(alpha[...,1], sigma)
        alpha = torch.from_numpy(alpha)
        img_1,_ = cp(img_1, img_2, mask_1, mask_2, alpha)
    else:
        for _ in range(num_holes):
            ratio = random.random() * (min_max[1]-min_max[0]) + min_max[0]
            coordinate = (random.randint(0,h-int(h*ratio)-1), 
                          random.randint(0,w-int(w*ratio)-1)) # coordinate of left-top size of the patch
            img_1[:,
                  coordinate[0]:coordinate[0]+int(h*ratio),
                  coordinate[1]:coordinate[1]+int(w*ratio)] = img_2[:,
                                                                    coordinate[0]:coordinate[0]+int(h*ratio),
                                                                    coordinate[1]:coordinate[1]+int(w*ratio)]
            mask_1[coordinate[0]:coordinate[0]+int(h*ratio),
                   coordinate[1]:coordinate[1]+int(w*ratio),
                                                           :] = mask_2[coordinate[0]:coordinate[0]+int(h*ratio),
                                                                       coordinate[1]:coordinate[1]+int(w*ratio),
                                                                       :]
    return img_1, mask_1



def cutmix_random_tumor(img_1, img_2, mask_1, mask_2, num_holes:int=3, min_max:list=[0.1,0.4], size=256, p_cp=1, p_trans=0.5):
    '''
    > TumorCP[Yang et al., MICAAI 2021] for CutMix
    input : img (size, size, 3) -> numpy.ndarray
            mask (size, size, num_classes) -> numpy.ndarray
            size : size of image(width==height)
            p_cp : tumor_cp probability
            p_trans : probability for rigid augmentation
    output : CP-Tumor augmented img and mask
             img (3, size, size) -> torch.LongTensor
             mask (size, size, num_classes) -> torch.LongTensor
    '''
    # step0 : Make alpha
    h, w, _ = img_1.shape
    alpha = np.zeros_like(mask_2)
    for _ in range(num_holes):
        ratio = random.random() * (min_max[1]-min_max[0]) + min_max[0]
        coordinate = (random.randint(0,h-int(h*ratio)-1), 
                      random.randint(0,w-int(w*ratio)-1)) # coordinate of left-top size of the patch
        alpha[coordinate[0]:coordinate[0]+int(h*ratio),
              coordinate[1]:coordinate[1]+int(w*ratio), :] = 1.0
    # step1 : Do TumorCP with P_cp
    if p_cp >= random.random():
        # step1 : Do Object-level augmentation with P_trans
        ## step1-1 : rigid transformation
        transform_rigid = A.Compose([
                                     A.HorizontalFlip(p=p_trans),
                                     A.VerticalFlip(p=p_trans),
                                     A.RandomRotate90(p=p_trans),
                                     A.Rotate(limit=(-180, 180),p=p_trans), # =(-pi, pi)
                                     A.RandomScale(scale_limit=(-0.25,0.25),p=p_trans), # =(0.75,1.25), I scaled from paper's ratio to follow albumentation's range
                                     A.PadIfNeeded(size,size),
                                     A.Resize(size,size)
                                     ]) # a.k.a spatial transformation in the paper
        aug = transform_rigid(image=img_2, mask=alpha)
        img_2 = aug['image'] # (size,size,3) np.ndarray
        alpha = aug['mask'] # (size,size,2) np.ndarray

        ## step1-2 : gamma transformation
        transform_gamma = A.Compose([A.RandomGamma(gamma_limit=(75,150),p=p_trans),
                                     A.Resize(size,size)]) # gamma, I scaled from paper's ratio to follow albumentation's range
        aug = transform_gamma(image=img_2.astype(np.float64), mask=alpha) # A.RandomGamma uses Numpy's function. It requires float64 type.
        img_2 = aug['image'] # (size,size,3) np.ndarray
        alpha = aug['mask'] # (size,size,2) np.ndarray

        ## step1-3 : blurring(gaussian) transformation
        sigma = random.randint(50,100)/100
        alpha = np.where(alpha>0.0001, 1.0, 0.0)
        mask = (1-alpha[...,1:])*mask_1 + alpha[...,1:]*mask_2
        alpha = gaussian(alpha[...,0], sigma=sigma)
        img = (1-alpha[...,None])*img_1 + alpha[...,None]*img_2
    else: # step1 : do nothing with prop (1-P_cp)
        img, mask = img_1, mask_1
    

    # step2 : Image-level Data Augmentation
    transform_img_level = A.Compose([A.Resize(size,size),
                                     ToTensorV2()])
    aug = transform_img_level(image=img, mask=mask)
    img = aug['image']
    mask = aug['mask']
    return img, mask



def cutmix_random_poisson(img_1, img_2, mask_1, mask_2, num_holes:int=3, min_max:list=[0.1,0.4], method='import', p_poisson=1):
    '''
    >P. Perez, M. Gangnet, and A. Blake. Poisson image editing
    >Source code was referenced from [https://github.com/rinsa318/poisson-image-editing].
    input : img (3, size, size) -> torch.FloatTensor
            mask (size, size, num_classes) -> torch.FloatTensor
            size : size of img (width=height) -> int
            p_poisson : poisson augmentation probability -> [0,1)
    output : Poisson Copy-and-Paste(CP) augmented img and mask
             img (3, size, size) -> torch.LongTensor
             mask (size, size, num_classes) -> torch.LongTensor
    '''
    assert method in ['import', 'mix', 'average', 'flatten'], 'CP-Poisson smoothening method Error'
    if p_poisson >= random.random(): # Do CP-Poisson with p_poisson
        # step1 : mask synthesis
        _, h, w = img_1.shape
        alpha = torch.zeros_like(mask_2)
        for _ in range(num_holes):
            ratio = random.random() * (min_max[1]-min_max[0]) + min_max[0]
            coordinate = (random.randint(0,h-int(h*ratio)-1), 
                      random.randint(0,w-int(w*ratio)-1)) # coordinate of left-top size of the patch
            alpha[coordinate[0]:coordinate[0]+int(h*ratio),
                  coordinate[1]:coordinate[1]+int(w*ratio), :] = 1.0
        
        mask = (1-alpha)*mask_1 + alpha*mask_2
        # step2 : image generate
        img_1 = torch.einsum('c...->...c', img_1)
        img_2 = torch.einsum('c...->...c', img_2)
        img_1 = img_1.cpu().detach().numpy().astype(np.float32) # target
        img_2 = img_2.cpu().detach().numpy().astype(np.float32) # source
        alpha = alpha.cpu().detach().numpy().astype(np.uint8) # source mask
        _, alpha = cv2.threshold(alpha[...,0], 0, 255, cv2.THRESH_OTSU)
        blended, _ = pie.poisson_blend(img_2, alpha/255.0, img_1, 'import')
        return torch.einsum('...c->c...', torch.from_numpy(blended/255.0)).float(), mask
    else: # Do nothing with 1-p_poisson
        return img_1, mask_1
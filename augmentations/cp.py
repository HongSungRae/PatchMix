import torch
import numpy as np
import random
from skimage.filters import gaussian
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import cv2
import time


# from local
try:
    import augmentations.poissonimageediting as pie
except:
    import poissonimageediting as pie 


def find_channel_to_attach(alpha):
    '''
    = Input =
    alpha : (size,size,num_classes-1) -> torch.FloatTensor
    = Output =
    alpha : (size,size) -> torch.FloatTensor
    > Input alpha is a binarized mask tensor except background class.
    > This function returns only one channel which cotains at least one object.
    '''
    # channel-wise하게 1(객체)이 있는 것 중에 한 채널만 고르기
    # not_zero_channels = []
    # for channel in range(alpha.shape[-1]):
    #     try: # for torch.Tensor
    #         if torch.any(alpha[...,channel]):
    #             not_zero_channels.append(channel)
    #     except: # for numpy.ndarray
    #         if np.any(alpha[...,channel]):
    #             not_zero_channels.append(channel)
    # return alpha[..., random.sample(not_zero_channels, k=1)[0]]
    return alpha[...,0]



def cp(img_1, img_2, mask_1, mask_2, alpha):
    '''
    input : img (3, size, size) -> torch.FloatTensor
            mask (size, size, num_classes) -> torch.FloatTensor
            alpha (size, size) -> torch.FloatTensor
    output : Copy-and-Paste(CP) augmented img and mask
             img (3, size, size) -> torch.LongTensor
             mask (size, size, num_classes) -> torch.LongTensor
    '''
    alpha = torch.unsqueeze(alpha, -1).float() # (size,size,1)
    alpha = torch.einsum('abc->cab', alpha) # (1,size,size)
    img_1 = img_1 * (1-alpha) + img_2 * alpha # img_1에서 alpha mask만큼 죽이고, img_2에서 alpha mask만큼 복사해옴
    mask_class_1 = torch.unsqueeze(torch.logical_or(mask_1[...,1], mask_2[...,1]),-1).float()
    mask_class_0 = torch.ones(mask_class_1.shape) - mask_class_1
    mask = torch.cat((mask_class_0, mask_class_1), dim=-1)
    return img_1, mask # (3, size, size) (size, size, 2)




def cp_simple(img_1, img_2, mask_1, mask_2, gaussian_blur=False, sigma=0.4):
    '''
    input : img (3, size, size) -> torch.FloatTensor
            mask (size, size, num_classes) -> torch.FloatTensor
            gaussian_blur, sigma : gaussian blur parameters
    output : Simple Copy-and-Paste(CP) augmented img and mask
             img (3, size, size) -> torch.LongTensor
             mask (size, size, num_classes) -> torch.LongTensor
    '''
    alpha = find_channel_to_attach(mask_2[...,1:]) # (size,size)
    if gaussian_blur:
        '''
        Ghiasi et al., (the author of CP-simple) said
        *we also found that simply composing without any (gaussian) blending has similar performance*
        '''
        alpha = gaussian(alpha, sigma)
        alpha = torch.from_numpy(alpha)
    return cp(img_1, img_2, mask_1, mask_2, alpha)




def cp_gaussian(img_1, img_2, mask_1, mask_2, sigma=0.4):
    '''
    input : img (3, size, size) -> torch.FloatTensor
            mask (size, size, num_classes) -> torch.FloatTensor
            sigma : gaussian blur parameter
    output : Gaussian Copy-and-Paste(CP) augmented img and mask
             img (3, size, size) -> torch.LongTensor
             mask (size, size, num_classes) -> torch.LongTensor
    '''
    alpha = find_channel_to_attach(mask_2[...,1:]) # (size,size)
    alpha = gaussian(alpha, sigma)
    alpha = torch.from_numpy(alpha)
    return cp(img_1, img_2, mask_1, mask_2, alpha)





def cp_tumor(img_1, img_2, mask_1, mask_2, size=256, p_cp=1, p_trans=0.5):
    '''
    > Implementation of TumorCP[Yang et al., MICAAI 2021]
    input : img (size, size, 3) -> numpy.ndarray
            mask (size, size, num_classes) -> numpy.ndarray
            size : size of image(width==height)
            p_cp : tumor_cp probability
            p_trans : probability for rigid augmentation
    output : CP-Tumor augmented img and mask
             img (3, size, size) -> torch.LongTensor
             mask (size, size, num_classes) -> torch.LongTensor
    '''
    # step1 : Do TumorCP with P_cp
    if p_cp >= random.random():
        # convert img_1, mask_1 to torch.FloatTensor
        transform_pytorch_tensor = A.Compose([A.Resize(size,size),
                                              ToTensorV2()])
        aug = transform_pytorch_tensor(image=img_1, mask=mask_1)
        img_1 = aug['image'] # (3,size,size)
        mask_1 = aug['mask'] # (size,size,2)

        # step2 : Randomly select a tumor to copy. img_2 and mask_2 are np.ndarray yet
        alpha = find_channel_to_attach(mask_2[...,1:]) # (size,size)
        alpha = np.expand_dims(alpha, -1) # (size,size,1)
        img_2 = img_2 * alpha # image with tumor only # (size,size,3)

        # step3 : Do Object-level augmentation with P_trans
        ## step3-1 : rigid transformation
        transform_rigid = A.Compose([
                                     A.HorizontalFlip(p=p_trans),
                                     A.VerticalFlip(p=p_trans),
                                     A.RandomRotate90(p=p_trans),
                                     A.Rotate(limit=(-180, 180),p=p_trans), # =(-pi, pi)
                                     A.RandomScale(scale_limit=(-0.25,0.25),p=p_trans), # =(0.75,1.25), I scaled from paper's ratio to follow albumentation's range
                                     A.PadIfNeeded(size,size),
                                     A.Resize(size,size)
                                     ]) # a.k.a spatial transformation in the paper
        aug = transform_rigid(image=img_2, mask=mask_2)
        img_2 = aug['image'] # (size,size,3) np.ndarray
        mask_2 = aug['mask'] # (size,size,2) np.ndarray

        ## step3-2 : gamma transformation
        transform_gamma = A.Compose([A.RandomGamma(gamma_limit=(75,150),p=p_trans),
                                     A.Resize(size,size)]) # gamma, I scaled from paper's ratio to follow albumentation's range
        aug = transform_gamma(image=img_2.astype(np.float64), mask=mask_2) # A.RandomGamma uses Numpy's function. It requires float64 type.
        img_2 = aug['image'] # (size,size,3) np.ndarray
        mask_2 = aug['mask'] # (size,size,2) np.ndarray

        ## step3-3 : blurring(gaussian) transformation
        sigma = random.randint(50,100)/100
        alpha = mask_2[...,1] # (size,size)
        alpha = gaussian(alpha, sigma=sigma) # (size,size)
        alpha = torch.from_numpy(alpha)

        # step4 : Randomly select a place to paste onto
        aug = transform_pytorch_tensor(image=img_2, mask=mask_2)
        img_2 = aug['image']
        mask_2 = aug['mask']
        img, mask = cp(img_1, img_2, mask_1, mask_2, alpha)
        
        # conver img, mask to numpy.ndarray
        img = img.cpu().detach().numpy()
        mask = mask.cpu().detach().numpy()
        img = np.einsum('chw->hwc',img)
    else: # step1 : do nothing with prop (1-P_cp)
        img, mask = img_1, mask_1
    

    # step5 : Image-level Data Augmentation
    transform_img_level = A.Compose([A.Resize(size,size),
                                     ToTensorV2()])
    aug = transform_img_level(image=img, mask=mask)
    img = aug['image']
    mask = aug['mask']
    return img, mask



# https://github.com/rinsa318/poisson-image-editing
# https://github.com/PPPW/poisson-image-editing
def cp_poisson(img_1, img_2, mask_1, mask_2, size=256, method='import', p_poisson=1):
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
        mask_class_1 = torch.unsqueeze(torch.logical_or(mask_1[...,1], mask_2[...,1]),-1).float()
        mask_class_0 = torch.ones(mask_class_1.shape) - mask_class_1
        mask = torch.cat((mask_class_0, mask_class_1), dim=-1)
         
        # step2 : image generate
        img_1 = torch.einsum('c...->...c', img_1)
        img_2 = torch.einsum('c...->...c', img_2)
        img_1 = img_1.cpu().detach().numpy().astype(np.float32) # target
        img_2 = img_2.cpu().detach().numpy().astype(np.float32) # source
        mask_2 = mask_2.cpu().detach().numpy().astype(np.uint8) # source mask
        mask_2 = find_channel_to_attach(mask_2[...,1:])
        _, mask_2 = cv2.threshold(mask_2, 0, 255, cv2.THRESH_OTSU)
        blended, _ = pie.poisson_blend(img_2, mask_2/255.0, img_1, 'import')
        return torch.einsum('...c->c...', torch.from_numpy(blended/255.0)).float(), mask
    else: # Do nothing with 1-p_poisson
        return img_1, mask_1




def _cp_poisson(tar_img_path, tar_mask_path, src_img_path, src_mask_path, size=256, method='import', p_poisson=1):
    '''
    >P. Perez, M. Gangnet, and A. Blake. Poisson image editing
    >Source code was referenced from [https://github.com/rinsa318/poisson-image-editing].
    input : img, mask path -> Note that img_1 is target and img_2 is source.
            size : size of img (width=height) -> int
            p_poisson : poisson augmentation probability -> [0,1)
    output : Poisson Copy-and-Paste(CP) augmented img and mask
             img (3, size, size) -> torch.LongTensor
             mask (size, size, num_classes) -> torch.LongTensor
    '''
    assert method in ['import', 'mix', 'average', 'flatten'], 'CP-Poisson smoothening method Error'
    resize = A.Compose([A.Resize(size,size)])
    # step1 : load images and masks
    tar = np.array(cv2.imread(tar_img_path, 1)/255.0, dtype=np.float32) # 붙임을 당할 사진
    tar_mask = cv2.imread(tar_mask_path, 0).astype(np.uint8) # 붙임을 당할 사진의 mask
    aug = resize(image=tar, mask=tar_mask)
    tar, tar_mask = aug['image'], aug['mask']
    src = np.array(cv2.imread(src_img_path, 1)/255.0, dtype=np.float32) # 붙일 사진
    src_mask = np.array(cv2.imread(src_mask_path, 0), dtype=np.uint8) # 붙일 사진의 mask
    aug = resize(image=src, mask=src_mask)
    src, src_mask = aug['image'], aug['mask']
    if p_poisson >= random.random():
        # step2 : mask
        mask_1 = np.logical_or(tar_mask, src_mask) # (size, size)
        mask = np.stack((~mask_1, mask_1),axis=-1) # (size, size, 2)
        mask = torch.from_numpy(mask)
        # step3 : image
        _, src_mask = cv2.threshold(src_mask, 0, 255, cv2.THRESH_OTSU)
        img, _ = pie.poisson_blend(src, src_mask/255.0, tar, method)
        img = torch.from_numpy(img)
        return img, mask
    else:
        tar = torch.from_numpy(tar)
        tar_mask = torch.from_numpy(tar_mask)
        return tar, tar_mask





if __name__ == '__main__':
    num_classes = 8 # The num of classes with background class
    img_1 = torch.abs(torch.randn((3,256,256)))
    img_2 = torch.abs(torch.randn((3,256,256)))
    mask_1 = torch.ones((256,256,num_classes))
    mask_2 = torch.ones((256,256,num_classes))

    # cp_simple
    start = time.time()
    img, mask = cp_simple(img_1, img_2, mask_1, mask_2)
    print(f'CP Simple : {img.shape} | {mask.shape} | {(time.time()-start):.4f} Seconds')

    # cp_gaussian
    start = time.time()
    img, mask = cp_gaussian(img_1, img_2, mask_1, mask_2)
    print(f'CP Gaussian : {img.shape} | {mask.shape} | {(time.time()-start):.4f} Seconds')

    # cp_poisson
    m_1 = torch.zeros((256,256,1))
    m_1[150:180,15:200,0] = 1
    mask_2 = torch.cat((torch.ones((256,256,1)) - m_1,m_1),dim=-1)
    start = time.time()
    img, mask = cp_poisson(img_1, img_2, mask_1, mask_2)
    print(f'CP Poisson : {img.shape} | {mask.shape} | {(time.time()-start):.4f} Seconds')

    # cp_tumor
    img_1 = np.random.randn(256,256,3)
    img_2 = np.random.randn(256,256,3)
    mask_1 = np.ones((256,256,num_classes))
    mask_2 = np.ones((256,256,num_classes))
    start = time.time()
    img, mask = cp_tumor(img_1, img_2, mask_1, mask_2, 256)
    print(f'CP Tumor : {img.shape} | {mask.shape} | {(time.time()-start):.4f} Seconds')
    print('* 참고 : 가끔 cp_tumor에서 np.power 오류 나는데, dummy 사용시만 경고 나고 실제 이미지에서는 문제 없음.')
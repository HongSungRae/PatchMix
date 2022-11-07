'''
input : img (3, size, size) -> numpy.ndarray or torch.Tensor
        mask (size, size, num_classes) -> numpy.ndarray or torch.Tensor
        size : size of img&mask
output : cutmix augmented img and mask
         img (3, size, size) -> numpy.ndarray or torch.Tensor
         mask (size, size, num_classes) -> numpy.ndarray or torch.Tensor
'''
import random

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



def cutmix_random(img_1, img_2, mask_1, mask_2, num_holes:int=3, min_max:list=[0.1,0.4]):
    _, h, w = img_1.shape
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
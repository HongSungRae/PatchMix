import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn

# from local




class DigestPath19(nn.Module):
    def __init__(self, split='train', dataset='NM', augmentation=None, aug_p=0.5, size=256):
        super().__init__()
        assert split in ['train', 'test', 'validation'], "BCNB split Error"
        assert dataset in ['M','NM'], "BCNB dataset Error"
        assert augmentation in ['cp_simple', 'cp_gaussian', 'cp_poisson', 'cp_tumor', 'cutmix_half', 'cutmix_dice', 'cutmix_random', 'image_aug', None], "BCNB augmentation Error"


    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass




if __name__ == '__main__':
    pass
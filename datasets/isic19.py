import os
import random
from pathlib import Path

import albumentations
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader

class Isic19(Dataset):

    def __init__(self, client_id, root_path:str, split:str = 'train', transforms=None):
        self.client_id   = client_id
        self.root_path   = root_path
        self.meta_data   = pd.read_csv(os.path.join(root_path, 'fl_metadata.csv'))
        self.meta_data   = self.meta_data[(self.meta_data['client_id'] == client_id) & 
                                                (self.meta_data['split'] == split)]
        self.root_path   = os.path.join(self.root_path, 'ISIC_2019_Training_Input_preprocessed')
        self.transforms  = transforms

    def __len__(self):
        return len(self.meta_data)

    def __getitem__(self,idx):
        y = self.meta_data.iloc[idx]['label']
        filename = self.meta_data.iloc[idx]['image'] + '.jpg'
        file_path = os.path.join(self.root_path, filename)
        image = np.array(Image.open(file_path))
        if self.transforms:
            augmented = self.transforms(image=image)
            image = augmented['image']
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)
        
        return (
            torch.tensor(image, dtype=torch.float32),
            torch.tensor(y, dtype=torch.int64),
        )

def get_augmentations(split='train'):
    
    sz = 200

    train_augmentations = albumentations.Compose(
                [
                    albumentations.RandomScale(0.07),
                    albumentations.Rotate(50),
                    albumentations.RandomBrightnessContrast(0.15, 0.1),
                    #albumentations.Flip(p=0.5),
                    albumentations.Affine(shear=0.1),
                    albumentations.RandomCrop(sz, sz),
                    #albumentations.CoarseDropout(random.randint(1, 8), (16, 16), (16, 16)),
                    albumentations.Normalize(),
                ]
            )
    
    val_augmentations = albumentations.Compose(
                    [
                        albumentations.Affine(shear=0.1),
                        albumentations.RandomCrop(sz, sz),
                        albumentations.Normalize(),
                    ]
                )
    if split == 'train':
        return train_augmentations
    else:
        return val_augmentations


def isic19_loading(client_id, root_dir='/scratch/htc/mkonstantin/data/P2P/Isic19/', batch_size=32):

    ds_train  = Isic19(client_id, root_dir, split = 'train', transforms=get_augmentations('train'))
    ds_eval   = Isic19(client_id, root_dir, split = 'val', transforms=get_augmentations('val'))
    ds_test   = Isic19(client_id, root_dir, split = 'test', transforms=get_augmentations('test'))
    
    ldr_train = DataLoader(ds_train, batch_size = batch_size, shuffle=True)
    ldr_eval  = DataLoader(ds_eval, batch_size = batch_size, shuffle=True)
    ldr_test  = DataLoader(ds_test, batch_size = batch_size, shuffle=True)

    return ldr_train, ldr_eval, ldr_test


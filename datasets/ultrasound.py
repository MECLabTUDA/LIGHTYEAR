import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
from PIL import Image
import os
import torch.nn.functional as F

class FetalAbdominal(Dataset):
    def __init__(self, client_id, root_path, split='train'):
        meta = pd.read_csv(os.path.join(root_path, 'fl_metadata.csv'))
        self.meta = meta[(meta['client_id'] == client_id) & (meta['split'] == split)]
        self.root_path = root_path
        
        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=0.5,std=0.5) #normalize to [-1, 1]
        ])

    def __len__(self):
        return len(self.meta)
    
    def __getitem__(self, idx):
        img_name = self.meta.iloc[idx]['filename']
        img = np.load(os.path.join(self.root_path, "ARRAY_FORMAT", img_name), allow_pickle=True)
        img = img.item()
        segmentation = img['structures']
        img = img['image']
        img = Image.fromarray(img)
        img = img.crop((100, 0, 900, img.height))
        img = self.transform(img)
        img = img[0:1]

        seg_liver = segmentation['liver']
        label = torch.from_numpy(seg_liver[:, 100:900])
        label = F.interpolate(label[None, None, :, :].float(), size=(64, 64), mode='nearest')[0, 0]
        #label = einops.rearrange(label, 'h w -> 1 h w')
        label = label.unsqueeze(0)
        
        return img, label

def ultrasound_loading(client_id, root_dir = '/scratch/htc/mkonstantin/data/P2P/ultrasound', batch_size=8):

    ds_train = FetalAbdominal(client_id, root_dir, split='train')
    ds_eval  = FetalAbdominal(client_id, root_dir, split='val')
    ds_test  = FetalAbdominal(client_id, root_dir, split='test')

    ldr_train = DataLoader(ds_train, batch_size = batch_size, shuffle=True, drop_last=True)
    ldr_eval  = DataLoader(ds_eval,  batch_size = 1, shuffle=True, drop_last=True)
    ldr_test  = DataLoader(ds_test, batch_size = 1, shuffle=True, drop_last=True)

    return ldr_train, ldr_eval, ldr_test

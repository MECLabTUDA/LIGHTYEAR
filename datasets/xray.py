import pandas as pd
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F

class XrayMimic(Dataset):
    def __init__(self, client_id, root_path, split='train'):
        meta = pd.read_csv(os.path.join(root_path, 'fl_metadata.csv'))
        self.root_path = root_path
        self.meta = meta[(meta['client_id'] == client_id) & (meta['split'] == split)]
        self.transforms = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=0.5,std=0.5) #normalize to [-1, 1]
        ])

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        filename = self.meta.iloc[idx]['filename']
        img_file = os.path.join(self.root_path, 'images', filename)
        lbl_file = os.path.join(self.root_path, 'labels', filename)

        img = Image.open(img_file).convert('L')
        lbl = Image.open(lbl_file)
        img = self.transforms(img)

        lbl = np.array(lbl)
        lbl = lbl[:, :, :3]  # Keep only the first three channels
        lbl[:,:,2] = 0
        lbl = lbl.max(axis=2)  # Convert to grayscale
        lbl = torch.from_numpy(lbl)
        lbl = F.interpolate(lbl[None, None, :, :].float(), size=(64, 64), mode='nearest')[0, 0]
        lbl = lbl.bool().float()

        lbl = lbl.unsqueeze(0)

        return img, lbl

def xray_loading(client_id, root_dir = '/scratch/htc/mkonstantin/data/P2P/xray', batch_size=4):
    ds_train = XrayMimic(client_id, root_dir, split='train')
    ds_eval  = XrayMimic(client_id, root_dir, split='val')
    ds_test  = XrayMimic(client_id, root_dir, split='test')

    ldr_train = DataLoader(ds_train, batch_size = batch_size, shuffle=True, drop_last=True)
    ldr_eval  = DataLoader(ds_eval,  batch_size = 1, shuffle=True, drop_last=True)
    ldr_test  = DataLoader(ds_test, batch_size = 1, shuffle=True, drop_last=True)

    return ldr_train, ldr_eval, ldr_test

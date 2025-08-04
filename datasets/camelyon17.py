from torch.utils.data import Dataset, DataLoader
import os
from torchvision import transforms
from PIL import Image
import pandas as pd
import random
from pathlib import Path
import torchvision.transforms as T


class Camelyon17(Dataset):

    def __init__(self, client_name, root_path:str, split:str = 'train'):
        
        self.client_name = client_name
        self.root_path   = root_path
        self.meta_data   = pd.read_csv(os.path.join(root_path, 'p2pfl_metadata.csv'))
        self.meta_data   = self.meta_data[(self.meta_data['center'] == client_name) & 
                                                (self.meta_data['split'] == split)]
        self.root_path   = os.path.join(self.root_path, 'patches')
        self.transform   = T.Compose([T.ToTensor(),])

    def __len__(self):
        return len(self.meta_data)

    
    def _get_sample(self, idx):
        y = self.meta_data.iloc[idx]['tumor']
        patient = self.meta_data.iloc[idx]['patient']
        node    = self.meta_data.iloc[idx]['node']
        x_coord = self.meta_data.iloc[idx]['x_coord']
        y_coord = self.meta_data.iloc[idx]['y_coord']
        filename = f'patient_{patient:03d}_node_{node}/patch_patient_{patient:03d}_node_{node}_x_{x_coord}_y_{y_coord}.png'
        img = Image.open(os.path.join(self.root_path, filename)).convert('RGB')
        return img, y

    def __getitem__(self, idx):
        
        img, y = self._get_sample(idx)
        img = self.transform(img)
        return img, y


def camelyon17_loading(client_id, root_dir='/path/to/camelyon17_v1.0', batch_size=32):

    ds_train  = Camelyon17(client_id, root_dir, split = 'train')
    ds_eval   = Camelyon17(client_id, root_dir, split = 'val')
    ds_test   = Camelyon17(client_id, root_dir, split = 'test')
    
    ldr_train = DataLoader(ds_train, batch_size = batch_size, shuffle=True)
    ldr_eval  = DataLoader(ds_eval, batch_size = batch_size, shuffle=True)
    ldr_test  = DataLoader(ds_test, batch_size = batch_size, shuffle=True)

    return ldr_train, ldr_eval, ldr_test


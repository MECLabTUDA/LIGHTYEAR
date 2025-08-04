import os
import torch
import re
import numpy as np

from torch.utils.data import Dataset, DataLoader

class FEMNIST(Dataset):
    def __init__(self, client_id, root_dir, split='train'):
        self.samples = self.sample(os.path.join(root_dir, f'client_{client_id}', split))

    def sample(self, root_dir):
        samples = []
        images  = os.listdir(root_dir)
        for img in images:
            match = re.search(r'label_(\d+)\.npy$', img)
            label = int(match.group(1))
            samples.append((os.path.join(root_dir, img), label))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        (img, label) = self.samples[idx]
        img = np.load(img)
        img = torch.Tensor(img).unsqueeze(0)
        return img, label

def femnist_loading(client_id, root_dir = '/path/to/femnist/leaf/data/femnist/data/clients', batch_size=32):

    ds_train = FEMNIST(client_id, root_dir, split='train')
    ds_eval  = FEMNIST(client_id, root_dir, split='val')
    ds_test  = FEMNIST(client_id, root_dir, split='test')

    ldr_train = DataLoader(ds_train, batch_size = batch_size, shuffle=True)
    ldr_eval  = DataLoader(ds_eval,  batch_size = batch_size, shuffle=True)
    ldr_test  = DataLoader(ds_test, batch_size = batch_size, shuffle=True)

    return ldr_train, ldr_eval, ldr_test

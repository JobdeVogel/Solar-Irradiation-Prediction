from __future__ import print_function
import torch.utils.data as data
import os
import os.path
import torch
import numpy as np
import sys
from tqdm import tqdm 
import json
from plyfile import PlyData, PlyElement
import time
import functools

class IrradianceDataset(data.Dataset):
    def __init__(self,
                 root,
                 npoints=10000,
                 split='train',
                 normals=True,
                 dtype=np.float32,
                 seed=78789
                 ):
        self.root = root
        self.npoints = npoints
        self.split = split
        self.normals = normals
        self.dtype = dtype
        self.files = []
        self.seed=seed
        
        np.random.seed(self.seed)
        
        for file_path in os.listdir(self.root):
            if os.path.isfile(os.path.join(self.root, file_path)):
                self.files.append(file_path)
        
    def __getitem__(self, index):
        file = self.files[index]
        
        with open(os.path.join(self.root, file), 'rb') as f:
            data = np.load(f)
    
        nan_mask = np.isnan(data).any(axis=1)
        filtered_data = data[~nan_mask]
        
        choice = np.random.choice(len(filtered_data), self.npoints, replace=True)
        sampled_data = filtered_data[choice, :]
        
        points = sampled_data[:, :6] if self.normals else sampled_data[:, :3]
        irr = sampled_data[:, -1]
        
        points = torch.from_numpy(points.astype(self.dtype))
        irr = torch.from_numpy(irr.astype(self.dtype))
        
        return points, irr
    
    def __len__(self):
        return len(self.files)

if __name__ == '__main__':
    dataset = IrradianceDataset(
        root="C:\\Users\\Job de Vogel\\OneDrive\\Documenten\\TU Delft\Master Thesis\\Dataset_pipeline\\pointnet_testing\\pointnet.pytorch\\raw",
    dtype=np.float32,
    normals=False
    )
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4,
        pin_memory=True
        )
    
    data = iter(dataloader)
    batch = next(data)
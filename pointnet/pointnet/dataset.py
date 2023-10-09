from __future__ import print_function
import torch.utils.data as data
import os
import os.path
import torch
import numpy as np
import sys
import random
from tqdm import tqdm 
import json
from plyfile import PlyData, PlyElement
import time
import functools

torch.set_printoptions(precision=5)
torch.set_printoptions(threshold=10)

def get_segmentation_classes(root):
    catfile = os.path.join(root, 'synsetoffset2category.txt')
    cat = {}
    meta = {}

    with open(catfile, 'r') as f:
        for line in f:
            ls = line.strip().split()
            cat[ls[0]] = ls[1]

    for item in cat:
        dir_seg = os.path.join(root, cat[item], 'points_label')
        dir_point = os.path.join(root, cat[item], 'points')
        fns = sorted(os.listdir(dir_point))
        meta[item] = []
        for fn in fns:
            token = (os.path.splitext(os.path.basename(fn))[0])
            meta[item].append((os.path.join(dir_point, token + '.pts'), os.path.join(dir_seg, token + '.seg')))
    
    with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../misc/num_seg_classes.txt'), 'w') as f:
        for item in cat:
            datapath = []
            num_seg_classes = 0
            for fn in meta[item]:
                datapath.append((item, fn[0], fn[1]))

            for i in tqdm(range(len(datapath))):
                l = len(np.unique(np.loadtxt(datapath[i][-1]).astype(np.uint8)))
                if l > num_seg_classes:
                    num_seg_classes = l

            print("category {} num segmentation classes {}".format(item, num_seg_classes))
            f.write("{}\t{}\n".format(item, num_seg_classes))

def gen_modelnet_id(root):
    classes = []
    with open(os.path.join(root, 'train.txt'), 'r') as f:
        for line in f:
            classes.append(line.strip().split('/')[0])
    classes = np.unique(classes)
    with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../misc/modelnet_id.txt'), 'w') as f:
        for i in range(len(classes)):
            f.write('{}\t{}\n'.format(classes[i], i))

class ShapeNetDataset(data.Dataset):
    def __init__(self,
                 root,
                 npoints=2500,
                 classification=False,
                 class_choice=None,
                 split='train',
                 data_augmentation=True):
        self.npoints = npoints
        self.root = root
        self.catfile = os.path.join(self.root, 'synsetoffset2category.txt')
        self.cat = {}
        self.data_augmentation = data_augmentation
        self.classification = classification
        self.seg_classes = {}
        
        with open(self.catfile, 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.cat[ls[0]] = ls[1]
        #print(self.cat)
        if not class_choice is None:
            self.cat = {k: v for k, v in self.cat.items() if k in class_choice}

        self.id2cat = {v: k for k, v in self.cat.items()}

        self.meta = {}
        splitfile = os.path.join(self.root, 'train_test_split', 'shuffled_{}_file_list.json'.format(split))
        #from IPython import embed; embed()
        filelist = json.load(open(splitfile, 'r'))
        for item in self.cat:
            self.meta[item] = []

        for file in filelist:
            _, category, uuid = file.split('/')
            if category in self.cat.values():
                self.meta[self.id2cat[category]].append((os.path.join(self.root, category, 'points', uuid+'.pts'),
                                        os.path.join(self.root, category, 'points_label', uuid+'.seg')))

        self.datapath = []
        for item in self.cat:
            for fn in self.meta[item]:
                self.datapath.append((item, fn[0], fn[1]))

        self.classes = dict(zip(sorted(self.cat), range(len(self.cat))))
        # print(self.classes)
        with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../misc/num_seg_classes.txt'), 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.seg_classes[ls[0]] = int(ls[1])
        self.num_seg_classes = self.seg_classes[list(self.cat.keys())[0]]
        # print(self.seg_classes, self.num_seg_classes)

    def __getitem__(self, index):
        fn = self.datapath[index]
        # print(fn)
        cls = self.classes[self.datapath[index][0]]
        point_set = np.loadtxt(fn[1]).astype(np.float32)
        seg = np.loadtxt(fn[2]).astype(np.int64)
        #print(point_set.shape, seg.shape)

        choice = np.random.choice(len(seg), self.npoints, replace=True)
        #resample
        point_set = point_set[choice, :]

        point_set = point_set - np.expand_dims(np.mean(point_set, axis = 0), 0) # center
        dist = np.max(np.sqrt(np.sum(point_set ** 2, axis = 1)),0)
        point_set = point_set / dist #scale

        if self.data_augmentation:
            theta = np.random.uniform(0,np.pi*2)
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
            point_set[:,[0,2]] = point_set[:,[0,2]].dot(rotation_matrix) # random rotation
            point_set += np.random.normal(0, 0.02, size=point_set.shape) # random jitter

        seg = seg[choice]
        point_set = torch.from_numpy(point_set)
        seg = torch.from_numpy(seg)
        cls = torch.from_numpy(np.array([cls]).astype(np.int64))

        if self.classification:
            return point_set, cls
        else:
            return point_set, seg

    def __len__(self):
        return len(self.datapath)

class ModelNetDataset(data.Dataset):
    def __init__(self,
                 root,
                 npoints=2500,
                 split='train',
                 data_augmentation=True):
        self.npoints = npoints
        self.root = root
        self.split = split
        self.data_augmentation = data_augmentation
        self.fns = []
        with open(os.path.join(root, '{}.txt'.format(self.split)), 'r') as f:
            for line in f:
                self.fns.append(line.strip())

        self.cat = {}
        with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../misc/modelnet_id.txt'), 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.cat[ls[0]] = int(ls[1])

        # print(self.cat)
        self.classes = list(self.cat.keys())

    def __getitem__(self, index):
        fn = self.fns[index]
        cls = self.cat[fn.split('/')[0]]
        with open(os.path.join(self.root, fn), 'rb') as f:
            plydata = PlyData.read(f)
        pts = np.vstack([plydata['vertex']['x'], plydata['vertex']['y'], plydata['vertex']['z']]).T
        choice = np.random.choice(len(pts), self.npoints, replace=True)
        point_set = pts[choice, :]

        point_set = point_set - np.expand_dims(np.mean(point_set, axis=0), 0)  # center
        dist = np.max(np.sqrt(np.sum(point_set ** 2, axis=1)), 0)
        point_set = point_set / dist  # scale

        if self.data_augmentation:
            theta = np.random.uniform(0, np.pi * 2)
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
            point_set[:, [0, 2]] = point_set[:, [0, 2]].dot(rotation_matrix)  # random rotation
            point_set += np.random.normal(0, 0.02, size=point_set.shape)  # random jitter

        point_set = torch.from_numpy(point_set.astype(np.float32))
        cls = torch.from_numpy(np.array([cls]).astype(np.int64))
        return point_set, cls


    def __len__(self):
        return len(self.fns)

def traverse_root(root):
    res = []
    for (dir_path, _, file_names) in os.walk(root):
        for file in file_names:
            res.append(os.path.join(dir_path, file))

    return res
   
class IrradianceDataset(data.Dataset):
    def __init__(self,
                 root,
                 npoints=10000,
                 split='train',
                 split_size=0.8,
                 normals=True,
                 dtype=np.float32,
                 shuffle=True,
                 seed=78789
                 ):
        self.root = root
        self.npoints = npoints
        self.split = split
        self.split_size = split_size
        self.normals = normals
        self.dtype = dtype
        self.files = []
        self.shuffle = shuffle
        self.seed=seed
        
        np.random.seed(self.seed)
        random.seed(self.seed)
        
        files = traverse_root(self.root)
        split_index = int(len(files) * self.split_size)
        
        if self.shuffle:
            random.shuffle(files)
            
        if split == 'train':
            for file_path in files[:split_index]:
                self.files.append(file_path)
        elif split == 'test':
            for file_path in files[split_index:]:
                self.files.append(file_path)       
    
    def transform_features(self, sample: torch.tensor, min=-50, max=50) -> torch.tensor:
        # TODO: Get out off dataset class and use in preprocessing phase
        # Clip and normalize tensor with pointcloud
        # Return absolute values to avoid negative 0.0 values
        
        columns_to_normalize = slice(0, 3)
        
        clip_min = torch.tensor([min, min, 0])
        clip_max = torch.tensor([max, max, max*2])
        min_values = torch.tensor([min, min, 0])
        max_values = torch.tensor([max, max, max*2])
        
        normalized_tensor = sample.clone()
        
        normalized_tensor[:, columns_to_normalize] = torch.clamp(normalized_tensor[:, columns_to_normalize], clip_min, clip_max)
        
        normalized_tensor[:, columns_to_normalize] -= min_values
        normalized_tensor[:, columns_to_normalize] /= (max_values - min_values)
        
        # From [0, 1] to [-1, 1]
        normalized_tensor = 2 * normalized_tensor - 1
        
        return normalized_tensor
        
    def transform_outputs(self, outputs: torch.tensor) -> torch.tensor:
        outputs = torch.clamp(outputs, 0, 1000)
        
        min = 0
        max = 1000
        
        outputs -= min
        outputs /= (max - min)
        
        return outputs
     
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
        
        points = self.transform_features(points)
        
        irr = self.transform_outputs(irr)
        
        return points, irr
    
    def __len__(self):
        return len(self.files)

if __name__ == '__main__':
    
    path = "C:\\Users\\Job de Vogel\\OneDrive\\Documenten\\TU Delft\\Master Thesis\\Dataset_pipeline\\dataset\\data\\raw"
    
    train_dataset = IrradianceDataset(
        root=path,
        npoints=2500,
        dtype=np.float32,
        normals=False
    )
    
    test_dataset = IrradianceDataset(
        root=path,
        npoints=2500,
        split='test',
        dtype=np.float32,
        normals=False
    )
    
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4,
        pin_memory=True
        )
    
    test_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4,
        pin_memory=True
        )
    
    
    dataset = 'shapenet'
    datapath = 'D:\\Master Thesis Data\\Shapenet\\nonormal'

    dataset = ShapeNetDataset(
        root=datapath,
        npoints=2500,
        classification=False,
        class_choice=['Chair', 'Guitar'])
    
    print(dataset[0])
    print(dataset[0][0].shape)
    print(train_dataset[0])
    print(train_dataset[0][0].shape)
    '''
    # dataloader = torch.utils.data.DataLoader(
    #     dataset,
    #     batch_size=32,
    #     shuffle=True,
    #     num_workers=int(4))

    # test_dataset = ShapeNetDataset(
    #     root=datapath,
    #     npoints=2500,
    #     classification=False,
    #     class_choice=['Chair', 'Guitar'],
    #     split='test',
    #     data_augmentation=False)
    # testdataloader = torch.utils.data.DataLoader(
    #     test_dataset,
    #     batch_size=32,
    #     shuffle=True,
    #     num_workers=int(4))
    '''
import os
import pickle
import logging
import numpy as np
from tqdm import tqdm
import random
from datetime import datetime
import torch
from torch.utils.data import Dataset
from ..data_util import crop_pc, voxelize
from ..build import DATASETS
import sys
import matplotlib.pyplot as plt
import matplotlib

def traverse_root(root):
    res = []
    for (dir_path, _, file_names) in os.walk(root):
        for file in file_names:
            # Make sure you are not using the .pkl file in the processed folder
            if '.npy' in file:
                res.append(os.path.join(dir_path, file))

    return res

@DATASETS.register_module()
class IRRADIANCE(Dataset):
    '''
    classes = ['ceiling',
               'floor',
               'wall',
               'beam',
               'column',
               'window',
               'door',
               'chair',
               'table',
               'bookcase',
               'sofa',
               'board',
               'clutter']
    '''
    num_classes = 1
    '''
    num_per_class = np.array([3370714, 2856755, 4919229, 318158, 375640, 478001, 974733,
                              650464, 791496, 88727, 1284130, 229758, 2272837], dtype=np.int32)
    class2color = {'ceiling':     [0, 255, 0],
                   'floor':       [0, 0, 255],
                   'wall':        [0, 255, 255],
                   'beam':        [255, 255, 0],
                   'column':      [255, 0, 255],
                   'window':      [100, 100, 255],
                   'door':        [200, 200, 100],
                   'table':       [170, 120, 200],
                   'chair':       [255, 0, 0],
                   'sofa':        [200, 100, 100],
                   'bookcase':    [10, 200, 100],
                   'board':       [200, 200, 200],
                   'clutter':     [50, 50, 50]}    
    cmap = [*class2color.values()]
    gravity_dim = 2
    '''
    """S3DIS dataset, loading the subsampled entire room as input without block/sphere subsampling.
    number of points per room in average, median, and std: (794855.5, 1005913.0147058824, 939501.4733064277)
    Args:
        data_root (str, optional): Defaults to 'data/S3DIS/s3disfull'.
        test_area (int, optional): Defaults to 5.
        voxel_size (float, optional): the voxel size for donwampling. Defaults to 0.04.
        voxel_max (_type_, optional): subsample the max number of point per point cloud. Set None to use all points.  Defaults to None.
        split (str, optional): Defaults to 'train'.
        transform (_type_, optional): Defaults to None.
        loop (int, optional): split loops for each epoch. Defaults to 1.
        presample (bool, optional): wheter to downsample each point cloud before training. Set to False to downsample on-the-fly. Defaults to False.
        variable (bool, optional): where to use the original number of points. The number of point per point cloud is variable. Defaults to False.
    """
    def __init__(self,
                 data_root: str = 'D:/Master Thesis Data',
                 test_area: int = 5,
                 voxel_size: float = 0.04,
                 voxel_max=None,
                 split: str = 'train',
                 split_ratio = 0.8,
                 transform=None,
                 loop: int = 1,
                 presample: bool = True,
                 variable: bool = False,
                 shuffle: bool = True,
                 bins=10,
                 compute_hist=False,
                 show=False,
                 _overwrite_dset_size=0
                 ):

        super().__init__()
        self.split, self.voxel_size, self.transform, self.voxel_max, self.loop = \
            split, voxel_size, transform, voxel_max, loop
        self.presample = presample
        self.variable = variable
        self.shuffle = shuffle
        self.bins=bins
        self.hist=None
        self.compute_hist=compute_hist
        self.show_hist=show
        self._overwrite_dset_size=_overwrite_dset_size

        #data_root = 'D:/Master Thesis Data/3SDIS/data/S3DIS/s3disfull'
        
        #raw_root = os.path.join(data_root, 'raw')
        raw_root = data_root
        
        self.raw_root = raw_root
        '''data_list = sorted(os.listdir(raw_root))'''
        data_list = traverse_root(raw_root)
        
        if self._overwrite_dset_size > 0:
            data_list = data_list[:_overwrite_dset_size]
        
        # TODO: include        
        split_index = int(len(data_list) * split_ratio)
        
        if len(data_list) == 0:
            print(f'WARNING: number of available samples in {self.raw_root} is 0, it is not possible to generate a dataset')
            sys.exit()
        elif len(data_list) == 1:
            print(f'WARNING: number of available samples in {self.raw_root} is 1, only a train dataset can be generated, test will be skipped')
        
        data_list = [item[:-4] for item in data_list]
        '''data_list = [item[:-4] for item in data_list if 'Area_' in item] '''
        
        '''        
        if split == 'train':
            self.data_list = [
                item for item in data_list if not 'Area_{}'.format(test_area) in item]
        else:
            self.data_list = [
                item for item in data_list if 'Area_{}'.format(test_area) in item]
        '''
        
        # Make sure the shuffling is similar for training and evaluation
        random.seed(1)
        if self.shuffle:
            random.shuffle(data_list)
        
        from_date = "{:%Y_%m_%d_%H_%M_%S}".format(datetime.now())
        
        if split == 'train':
            self.data_list = [
                item for item in data_list[:split_index+1]
            ]
            
            with open(f'.\data\{from_date}_train_samples.txt', 'w') as file:
                for item in self.data_list:
                    file.write(item + '\n')
        elif split == 'val':
            self.data_list = [
                item for item in data_list[split_index+1:]
            ]

            with open(f'.\data\{from_date}_evaluation_samples.txt', 'w') as file:
                for item in self.data_list:
                    file.write(item + '\n')
        elif split == 'test':
            self.data_list = data_list

        processed_root = os.path.join(data_root, 'processed_random_100')
        # processed_root = 'D:\Master Thesis Data\IrradianceNet\\100_regular'
        
        filename = os.path.join(
            processed_root, f'irradiance_{split}_{voxel_size:.3f}_{str(voxel_max)}_{str(self.bins)}.pkl')
                
        self.bin_edges = torch.linspace(0, 1000, steps=self.bins+1).unsqueeze(0)        
        
        if presample and not os.path.exists(filename):
            np.random.seed(0)
            self.data = []

            print(f"Preampling {len(self.data_list)} samples from {raw_root}")
            for item in tqdm(self.data_list, desc=f'Loading irradiance dataset {split} split'):
                data_path = os.path.join(raw_root, item + '.npy')

                cdata = np.load(data_path).astype(np.float32)
                '''
                Sample is extracted, most likely needs to be preprocessed here
                '''
                # Remove the None values (points that should not be included)
                nan_mask = np.isnan(cdata).any(axis=1)
                cdata = cdata[~nan_mask]

                '''cdata[:, :3] -= np.min(cdata[:, :3], 0)'''
                if voxel_size:
                    coord, feat, label = cdata[:,0:3], cdata[:, 3:-1], cdata[:, -1]

                    uniq_idx = voxelize(coord, voxel_size)

                    label = np.expand_dims(label, 1)

                    coord, feat, label = coord[uniq_idx], feat[uniq_idx], label[uniq_idx]
                    cdata = np.hstack((coord, feat, label))

                labels = cdata[:, -1]
                self.compute_bins(torch.tensor(labels))

                cdata = np.hstack((cdata, self.bin_idxs))

                self.data.append(cdata)
                
            npoints = np.array([len(data) for data in self.data])
            logging.info('split: %s, median npoints %.1f, avg num points %.1f, std %.1f' % (
                self.split, np.median(npoints), np.average(npoints), np.std(npoints)))
            os.makedirs(processed_root, exist_ok=True)
            
            with open(filename, 'wb') as f:
                logging.info(f"Saving file {filename}...")
                pickle.dump(self.data, f)
                print(f"{filename} saved successfully")
            
        elif presample:
            logging.info(f"Loading file {filename}...")
            with open(filename, 'rb') as f:
                self.data = pickle.load(f)
                print(f"{filename} load successfully")
        
        if self.compute_hist:
            histograms = torch.zeros((len(self.data_list) * self.loop, self.bins)).long()
        
        if presample:
            if self.compute_hist:
                logging.info("Generating histogram...")
                if _overwrite_dset_size > 0:
                    self.data = self.data[:_overwrite_dset_size]

                for i, sample in enumerate(self.data):
                    targets = torch.tensor(np.split(sample, [3, 6, 7], axis=1)[2])
                    hist = torch.histc(targets, bins=self.bins).long()
                    
                    histograms[i, :] = hist

                self.hist = histograms.sum(axis=0)

                print(f"Histogram: {self.hist.tolist()}")
            
                if self.show_hist:
                    edges = self.bin_edges.view(-1)
                    names = [str(int(edges[i].item())) + '-' + str(int(edges[i+1].item())) for i in range(len(edges[:-1]))]
                    matplotlib.use('TkAgg')
                    
                    # Plotting the histogram
                    plt.bar(names, self.hist)
                    plt.xlabel('Irradiance [kWh/m2]')
                    plt.ylabel('Frequency of irradiance bin')
                    plt.title('Histogram of Irradiance Data')
                    plt.show()
                    
                    matplotlib.use('Agg')
        else:
            for sample in self.data_list:
                data_path = os.path.join(
                    self.raw_root, sample + '.npy')
                
                cdata = np.load(data_path).astype(np.float32)
                
                # Remove the None values (points that should not be included)
                nan_mask = np.isnan(cdata).any(axis=1)
                cdata = cdata[~nan_mask]
                
                labels = cdata[:, -1]
                
                if self.hist == None:
                    self.hist = torch.histc(torch.tensor(labels), bins=self.bins).long()
                else:
                    self.hist += torch.histc(torch.tensor(labels), bins=self.bins).long()

        self.data_idx = np.arange(len(self.data_list))
        assert len(self.data_idx) > 0
        
        # ! commented
        # logging.info(f"\nTotally {len(self.data_idx)} samples in {split} set")

    def compute_bins(self, values):
        diff = values.unsqueeze(1) - self.bin_edges
        cumsum = torch.cumsum(diff >= 0, dim=1)
        bin_idxs = torch.argmax(cumsum, dim=1)
        
        self.bin_idxs = np.expand_dims(bin_idxs.numpy(), 1)

    @staticmethod
    def statistics(sample):
        print(sample['pos'])
        print('\n')
        
        print('x:')
        print('min :' + str(torch.min(sample['pos'][:, :, 0])))
        print('max: ' + str(torch.max(sample['pos'][:, :, 0])) + '\n')
        
        print('y:')
        print('min :' + str(torch.min(sample['pos'][:, :, 1])))
        print('max: ' + str(torch.max(sample['pos'][:, :, 1])) + '\n')
        
        print('z:')
        print('min :' + str(torch.min(sample['pos'][:, :, 2])))
        print('max: ' + str(torch.max(sample['pos'][:, :, 2])) + '\n')
        
        print('u:')
        print('min :' + str(torch.min(sample['normals'][:, :, 0])))
        print('max: ' + str(torch.max(sample['normals'][:, :, 0])) + '\n')
        
        print('v:')
        print('min :' + str(torch.min(sample['normals'][:, :, 1])))
        print('max: ' + str(torch.max(sample['normals'][:, :, 1])) + '\n')
        
        print('w:')
        print('min :' + str(torch.min(sample['normals'][:, :, 2])))
        print('max: ' + str(torch.max(sample['normals'][:, :, 2])) + '\n')

    def __getitem__(self, idx):      
        data_idx = self.data_idx[idx % len(self.data_idx)]
        file = self.data_list[idx]        

        # If I have turned presample off, then there is no self.data
        # else there is self.data        
        if self.presample:
            try:
                coord, feat, label, bins = np.split(self.data[data_idx], [3, 6, 7], axis=1)    
            except:
                logging.warning(f"Failed to extract data from sample {data_idx}")
            feat = np.hstack((feat, bins))
            
            # Scale from [-50, 50] to [0, 100]
            coord, feat, label = crop_pc(
                coord, feat, label, self.split, self.voxel_size, self.voxel_max,
                downsample=False, variable=self.variable, shuffle=self.shuffle)     
            
            feat, bins = np.split(feat, [3], axis=1)    
        else:          
            data_path = os.path.join(
                self.raw_root, self.data_list[data_idx] + '.npy')
                     
            cdata = np.load(data_path).astype(np.float32)
            
            # Remove the None values (points that should not be included)
            nan_mask = np.isnan(cdata).any(axis=1)
            cdata = cdata[~nan_mask]
            
            labels = cdata[:, -1]
            self.compute_bins(torch.tensor(labels))
                
            cdata = np.hstack((cdata, self.bin_idxs))
            
            '''cdata[:, :3] -= np.min(cdata[:, :3], 0)'''
            try:
                coord, feat, label, bins = np.split(cdata, [3, 6, 7], axis=1)       
            except:
                logging.warning(f"Failed to extract data from sample {data_idx}")
                print(self.data[data_idx])
            
            feat = np.hstack((feat, bins))
            
            # TODO: Upsample or downsample here
            coord, feat, label = crop_pc(
                coord, feat, label, self.split, self.voxel_size, self.voxel_max,
                downsample=False, variable=self.variable, shuffle=self.shuffle)
            
            feat, bins = np.split(feat, [3], axis=1)    
                
            '''# TODO: do we need to -np.min in cropped data?'''

        '''label = label.squeeze(-1).astype(np.long)'''
        if label.ndim == 2:
            label = label.squeeze(-1)

        data = {'pos': coord, 'x': feat, 'y': label}

        # TODO: figure out if the transforms are necessary
        '''
        I SHOULD PROBABLY BUILD MY OWN TRANSFORM THAT ARE SIMILAR TO THE ORIGINAL
        POINTNEXT PAPER, BUT ALIGN WITH MY OWN DATASET.
        
        * validation: [PointsToTensor, PointCloudXYZAlign, ChromaticNormalize]
            * PointsToTensor: Transform the numpy arrays to torch tensors
            * PointCloudXYZAlign: Center the data to the center xy plane
            * ChromaticNormalize: Seems to be related to color
        
        * training: [ChromaticAutoContrast, PointsToTensor, PointCloudScaling, PointCloudXYZAlign, PointCloudRotation, PointCloudJitter, ChromaticDropGPU, ChromaticNormalize]
            * REMOVE: ChromaticAutoContrast: Seems to be related to color
            * PointsToTensor: Transform the numpy arrays to torch tensors
            * PointCloudXYZAlign: Center the data to the center xy plane
            * REMOVE: PointCloudRotation: Rotate in a 3D direction
            * REMOVE?: PointCloudJitter: Add noise to the pos
            * REMOVE: ChromaticDropGPU: Remove colors?
            * REMOVE: ChromaticNormalize: Seems to be related to color
        '''       
        # pre-process. 
        # Currently  pointstotensor, normalize and centering included
        if self.transform is not None:
            data = self.transform(data)

        if 'normals' not in data.keys():
            data['normals'] =  torch.from_numpy(feat.astype(np.float32))
        
        data['bins'] =  torch.from_numpy(np.squeeze(bins)).long()
        
        # ! Consider adding height
        
        # Remove negative zero values in normals
        data['normals'] += 0.
        data['x'] += 0.

        data['idx'] = idx
        # data['file'] = file
        
        return data

    def __len__(self):
        return len(self.data_idx) * self.loop
        # return 1   # debug

"""debug
from openpoints.dataset import vis_multi_points
import copy
old_data = copy.deepcopy(data)
if self.transform is not None:
    data = self.transform(data)
vis_multi_points([old_data['pos'][:, :3], data['pos'][:, :3].numpy()], colors=[old_data['x'][:, :3]/255.,data['x'][:, :3].numpy()])
"""

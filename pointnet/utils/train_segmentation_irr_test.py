from __future__ import print_function

import sys
sys.path.append("../")

import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from pointnet.dataset import ShapeNetDataset, IrradianceDataset
from pointnet.irr_model_test import PointNetDenseCls, feature_transform_regularizer
from torch.utils.tensorboard import SummaryWriter

from torch.multiprocessing import freeze_support
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import sys
import time

import open3d as o3d
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize


'''
CHANGES FOR IRRADIANCE PREDICTION

The traditional PointNet predicts an output value for each class with the likelihood
of being that class. Since we are using regression to predict irradiance. The number
of classes is set to 1, log_softmax in the network is removed and loss function is
changed to mse_loss.
'''

parser = argparse.ArgumentParser()
parser.add_argument(
    '--batchSize', type=int, default=32, help='input batch size')
parser.add_argument(
    '--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument(
    '--nepoch', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--outf', type=str, default='seg', help='output folder')
parser.add_argument('--model', type=str, default='', help='model path')
parser.add_argument('--dataset', type=str, required=False, help="dataset path")
parser.add_argument('--class_choice', type=str, default='Chair', help="class_choice")
parser.add_argument('--feature_transform', action='store_true', help="use feature transform")

opt = parser.parse_args()

def visualize_pointcloud(pointcloud: np.array, color_values: np.array):
    print([i for i in color_values[:25]])
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pointcloud)
    
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, -1])
    
    color_map = plt.get_cmap('coolwarm')
    
    norm = Normalize(vmin=min(color_values), vmax=max(color_values))

    # Map normalized values to colors
    colors = color_map(norm(color_values))[:, :3]       
    
    pcd.colors = o3d.utility.Vector3dVector(colors)    

    o3d.visualization.draw_geometries([pcd, coord_frame],
                                        zoom=1,
                                        front=[1, 1, 1],
                                        lookat=[0, 0, -1],
                                        up=[0, 0, 1])
    
    # camera = visualizer.get_view_control()
    
    # camera.set_constant_z_near(0.001)
    # camera.set_constant_z_far(100.0)
    # camera.set_front([0, -np.sin(np.radians(45)), -np.cos(np.radians(45))])
    # camera.set_lookat([0, 0, 0])
    # camera.set_up([0, 0, 1])
    
    # visualizer.run()
    
    # o3d.visualization.draw_geometries([pcd],
    #                                     zoom=1,
    #                                     front=[0.4257, -0.2125, -0.8795],
    #                                     lookat=[2.6172, 2.0475, 1.532],
    #                                     up=[0, 0, -1])
    # sys.exit()

def fit(opt, lr, name):    
    writer = SummaryWriter(f'runs/shapenet/{name}') 
    
    print(f'Fitting model {name} with learning_rate {lr} and batchSize {opt.batchSize}')
        
    # Initialize dataset
    dataset = ShapeNetDataset(
        root=opt.dataset,
        npoints=2500,
        classification=False,
        class_choice=[opt.class_choice])
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batchSize,
        shuffle=True,
        num_workers=int(opt.workers))

    test_dataset = ShapeNetDataset(
        root=opt.dataset,
        npoints=2500,
        classification=False,
        class_choice=[opt.class_choice],
        split='test',
        data_augmentation=False)
    
    testdataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=opt.batchSize,
        shuffle=True,
        num_workers=int(opt.workers))
    
    
    # ! Overwrite for irradiance prediction
    num_classes = 1

    blue = lambda x: '\033[94m' + x + '\033[0m'

    classifier = PointNetDenseCls(k=num_classes, feature_transform=opt.feature_transform)

    if opt.model != '':
        classifier.load_state_dict(torch.load(opt.model))

    classifier.cuda()    
    
    optimizer = optim.Adam(classifier.parameters(), lr=lr, betas=(0.9, 0.999))
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    classifier.cuda()
    
    num_batch = len(dataset) / opt.batchSize

    losses = []

    step = 0
    for epoch in range(opt.nepoch):
        for i, data in enumerate(dataloader, 0):
            points, target = data
            
            points = points.transpose(2, 1)
            points, target = points.cuda(), target.cuda()
            
            optimizer.zero_grad()
            
            classifier = classifier.train()
            
            pred, trans, trans_feat = classifier(points)
            
            # Flatten the data
            pred = pred.view(-1)
            
            target = target.view(-1, 1)[:, 0] - 1
                        
            target = target.float()

            loss = F.mse_loss(pred, target)
            
            losses.append(loss)
            
            if opt.feature_transform:
                 loss += feature_transform_regularizer(trans_feat) * 0.001
            
            loss.backward()
            
            optimizer.step()
            
            avg_loss = sum(losses) / len(losses)
            
            writer.add_scalar('Training loss', loss, global_step=step)
            writer.add_scalar('Average training loss', avg_loss, global_step=step)
            step += 1

            print('[%d: %d/%d] train loss: %f, avg_loss: %f' % (epoch, i, num_batch, loss.item(), avg_loss))

            if i % 10 == 0:
                j, data = next(enumerate(testdataloader, 0))
                points, target = data
                
                points = points.transpose(2, 1)
                points, target = points.cuda(), target.cuda()
                
                classifier = classifier.eval()
                
                pred, _, _ = classifier(points)
                pred = pred.view(-1, 1)
                
                pred_regress = pred.data.squeeze()
                
                target = target.view(-1, 1)[:, 0] - 1
                target = target.float()
                
                loss = F.mse_loss(pred_regress, target) 
        
                writer.add_scalar('Average test loss', loss, global_step=step)
        
                print('[%d: %d/%d] %s loss: %f' % (epoch, i, num_batch, blue('test'), loss.item()))
          
          
        writer.add_hparams({'lr': lr, 'bs': opt.batchSize}, {'avg_train_loss': avg_loss})
        scheduler.step()

        print('[%d: %d/%d] Saving dict state...' % (epoch, i, num_batch))
        torch.save(classifier.state_dict(), '%s/seg_model_%s_%d.pth' % (opt.outf, opt.class_choice, epoch))

def main(opt):
    
    opt.dataset = 'C:\\Users\\Job de Vogel\\OneDrive\\Documenten\\TU Delft\\Master Thesis\\Dataset_pipeline\\dataset\\data\\raw'
    
    dataset = IrradianceDataset(
        root=opt.dataset,
        dtype=np.float32,
        normals=False,
        npoints=30000,
        transform=True,
        shuffle=False
    )
    
    idx = 15
    visualize_pointcloud(dataset[idx][0].numpy(), dataset[idx][1].numpy())
    
    '''
    if not opt.dataset:
        opt.dataset = 'D:\\Master Thesis Data\\Shapenet\\nonormal'
    
    # Set seed
    opt.manualSeed = random.randint(1, 10000)  # fix seed
    opt.manualSeed = 0
    
    try:
        os.makedirs(opt.outf)
    except OSError:
        pass
    
    learning_rates = [0.1, 0.01, 0.001, 0.0001]
    batch_sizes = [16, 32, 64, 128]
    learning_rates = [0.001]
    batch_sizes = [32]
    num_epochs = 100
    
    opt.nepoch = num_epochs
    
    for bs in batch_sizes:
        for lr in learning_rates:
            opt.batchSize = bs
            name = f'model_lr_{lr}_bs_{bs}'
            
            fit(opt, lr, name)
    '''

if __name__ == '__main__':      
    main(opt)
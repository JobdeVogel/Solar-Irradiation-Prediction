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

if not opt.dataset:
    # Temporarily assign dataset
    opt.dataset = "D:\\Master Thesis Data\\Shapenet\\nonormal"
    #opt.dataset = "C:\\Users\\Job de Vogel\\OneDrive\\Documenten\\TU Delft\Master Thesis\\Dataset_pipeline\\dataset\\pointnet\\raw"

def main():
    # Set seed
    opt.manualSeed = random.randint(1, 10000)  # fix seed
    opt.manualSeed = 0
    
    print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)

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
    
    '''
    dataset = IrradianceDataset(
        root=opt.dataset,
        dtype=np.float32,
        normals=False,
        npoints=2500
    )
    
    test_dataset = IrradianceDataset(
        root=opt.dataset,
        split='test',
        dtype=np.float32,
        normals=False,
        npoints=2500
    )
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4,
        pin_memory=True
        )
    
    testdataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4,
        pin_memory=True
        )
    '''
    
    # ! Overwrite for irradiance prediction
    num_classes = 1

    try:
        os.makedirs(opt.outf)
    except OSError:
        pass

    blue = lambda x: '\033[94m' + x + '\033[0m'

    classifier = PointNetDenseCls(k=num_classes, feature_transform=opt.feature_transform)

    if opt.model != '':
        classifier.load_state_dict(torch.load(opt.model))

    classifier.cuda()
    
    '''
    # Test a specific sample from the dataset
    criterion = nn.MSELoss()
    
    idx = 0
    points, target = dataset[idx]
    points = points.transpose(1, 0) 
    points = points.unsqueeze(0).cuda()
    points, target = points.cuda(), target.cuda()
    
    
    
    classifier.eval()  # Set the classifier to evaluation mode
    with torch.no_grad():
        pred, _, _ = classifier(points)
    
    pred_regress = pred.view(-1, 1)
    target = target.view(-1, 1)[:, 0] - 1
    tartget = target.float()
    
    loss = criterion(pred_regress, target)
    '''
    
    optimizer = optim.Adam(classifier.parameters(), lr=0.0001, betas=(0.9, 0.999))
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    classifier.cuda()

    num_batch = len(dataset) / opt.batchSize

    losses = []

    for epoch in range(opt.nepoch):
        for i, data in enumerate(dataloader, 0):
            points, target = data
            
            points = points.transpose(2, 1)
            points, target = points.cuda(), target.cuda()
            
            optimizer.zero_grad()
            
            classifier = classifier.train()
            
            pred, trans, trans_feat = classifier(points)
            
            # pred_choice = pred.data.max(1)[1] # Only used for segmentation
            # pred_regress = pred.data.squeeze() #! FIXED THE ISSUE, COMPUTATIONAL GRAPH BREAKS HERE!!!
            
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

            print('[%d: %d/%d] train loss: %f, avg_loss: %f' % (epoch, i, num_batch, loss.item(), sum(losses) / len(losses)))

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
        
                print('[%d: %d/%d] %s loss: %f' % (epoch, i, num_batch, blue('test'), loss.item()))
            
        scheduler.step()

        print('[%d: %d/%d] Saving dict state...' % (epoch, i, num_batch))
        torch.save(classifier.state_dict(), '%s/seg_model_%s_%d.pth' % (opt.outf, opt.class_choice, epoch))

if __name__ == '__main__':
    main()
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
from pointnet.irradiancemodel import PointNetDenseCls, feature_transform_regularizer
from torch.utils.tensorboard import SummaryWriter

from torch.multiprocessing import freeze_support
import torch.nn.functional as F
import numpy as np
import sys

# from eval import eval_image, visualize_pointcloud

from skorch import NeuralNetRegressor
from sklearn.model_selection import GridSearchCV

'''
CHANGES FOR IRRADIANCE PREDICTION

The traditional PointNet predicts an output value for each class with the likelihood
of being that class. Since we are using regression to predict irradiance. The number
of classes is set to 1, log_softmax in the network is removed and loss function is
changed to mse_loss.
'''

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchSize', type=int, default=32, help='input batch size')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
    parser.add_argument('--nepoch', type=int, default=25, help='number of epochs to train for')
    parser.add_argument('--outf', type=str, default='seg', help='output folder')
    parser.add_argument('--model', type=str, default='', help='model path')
    parser.add_argument('--dataset', type=str, required=False, help="dataset path")
    parser.add_argument('--class_choice', type=str, default='Chair', help="class_choice")
    parser.add_argument('--feature_transform', action='store_true', help="use feature transform")
    parser.add_argument('--fc1_out', type=int, default=2048, required=False)
    parser.add_argument('--fc2_out', type=int, default=512, required=False)
    parser.add_argument('--fc3_out', type=int, default=-1, required=False)
    parser.add_argument('--fc4_out', type=int, default=-1, required=False)
    parser.add_argument('--fc5_out', type=int, default=-1, required=False)
    parser.add_argument('--relu1', action='store_true', help="use feature transform")
    parser.add_argument('--relu2', action='store_true', help="use feature transform")
    parser.add_argument('--relu3', action='store_true', help="use feature transform")
    parser.add_argument('--relu4', action='store_true', help="use feature transform")
    parser.add_argument('--relu5', action='store_true', help="use feature transform")

    opt = parser.parse_args()
    
    return opt
    
def fit(opt, lr, name):    
    writer = SummaryWriter(f'runs/irradiancenet/{name}') 
    
    print(f'Fitting model {name} with learning_rate {lr} and batchSize {opt.batchSize}')
    
    dataset = IrradianceDataset(
        root=opt.dataset,
        dtype=np.float32,
        normals=False,
        npoints=2500,
        transform=True,
        resample=True
    )
    
    test_dataset = IrradianceDataset(
        root=opt.dataset,
        split='test',
        dtype=np.float32,
        normals=False,
        npoints=2500,
        transform=True,
        resample=True
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

    num_classes = 1

    blue = lambda x: '\033[94m' + x + '\033[0m'

    classifier = PointNetDenseCls(k=dataset.npoints, feature_transform=opt.feature_transform)

    if opt.model != '':
        classifier.load_state_dict(torch.load(opt.model))

    classifier.cuda()    

    test_interval = 25
    eval_interval = 25
    
    optimizer = optim.Adam(classifier.parameters(), lr=lr, betas=(0.9, 0.999))
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    
    num_batch = len(dataset) / opt.batchSize

    losses = []

    step = 0   
   
    eval_points, _ = test_dataset[0]

    # Goal [32, 3, 2500]
   
    for epoch in range(opt.nepoch):
       
        # Train
        for i, data in enumerate(dataloader, 0):
            points, target = data
            
            points = points.transpose(2, 1)
            points, target = points.cuda(), target.cuda()
            
            optimizer.zero_grad()
            
            pred, trans, trans_feat = classifier(points)
            
            target = target.view(-1, 1)[:, 0] - 1
            target = target.float()

            loss = F.mse_loss(pred, target)
            
            losses.append(loss)
            
            if opt.feature_transform:
                 loss += feature_transform_regularizer(trans_feat) * 0.001
            
            loss.backward()
            
            optimizer.step()
            
            # Compute average loss
            avg_loss = sum(losses) / len(losses)
            
            # Add data to Tensorboard
            writer.add_scalar('Training loss', loss, global_step=step)
            writer.add_scalar('Average training loss', avg_loss, global_step=step)
            step += 1

            print('[%d: %d/%d] train loss: %f, avg_loss: %f' % (epoch, i, num_batch, loss.item(), avg_loss))
            
            # Test
            if i % test_interval == 0:
                j, data = next(enumerate(testdataloader, 0))
                points, target = data
                
                points = points.transpose(2, 1)
                points, target = points.cuda(), target.cuda()
                
                classifier = classifier.eval()
                
                with torch.no_grad():
                    pred, _, _ = classifier(points)
                    
                    target = target.view(-1, 1)[:, 0] - 1
                    target = target.float()
                    
                    loss = F.mse_loss(pred, target) 
            
                    writer.add_scalar('Average test loss', loss, global_step=step)
                
                print('[%d: %d/%d] %s loss: %f' % (epoch, i, num_batch, blue('test'), loss.item()))
        
            # Evaluation
            if i % eval_interval == 0:
                eval_name = f'evaluation_epoch_{str(epoch)}_it_{str(i)}.png'
                eval_image(eval_points, classifier, writer, eval_name)
        
        # writer.add_hparams({'lr': lr, 'bs': opt.batchSize}, {'avg_train_loss': avg_loss})
        scheduler.step()

        print('[%d: %d/%d] Saving dict state...' % (epoch, i, num_batch))
        torch.save(classifier.state_dict(), '%s/irr_model_epoch_%d.pth' % (opt.outf, epoch))

def gridsearch(opt, lr, name):   
    print(f'Fitting model {name} with learning_rate {lr} and batchSize {opt.batchSize}')
    
    dataset = IrradianceDataset(
        root=opt.dataset,
        dtype=np.float32,
        normals=False,
        split_size=1.0,
        npoints=2500,
        transform=True,
        resample=True
    )
    
    points = torch.stack([sample[0] for sample in dataset]).permute(0, 2, 1).type(torch.float32)
    values = torch.stack([sample[1] for sample in dataset]).type(torch.float32)
    
    model = NeuralNetRegressor(
        module=PointNetDenseCls,
        criterion=nn.MSELoss,
        optimizer=optim.Adam,
        max_epochs=5,
        lr=0.001,
        batch_size=32,
        device='cuda',
        verbose=3,
        module__k=2500,
        module__feature_transform=True,
        module__gridsearch=True
    )
    
    param_grid = {
        'optimizer': [optim.SGD, optim.RMSprop, optim.Adagrad, optim.Adadelta, optim.Adam, optim.Adamax, optim.NAdam],
        'batch_size': [8, 16, 32],
        'optimizer__lr': [0.001, 0.01, 0.1]
    }
    
    grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=1, cv=3)
    grid_result = grid.fit(points, values)
    
    # summarize results
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))
    
    '''
    Test if classifier works
    '''
    '''
    batch_points = points[:32, :, :].cuda()
    batch_target = values[:32, :].cuda()
    
    print(batch_points.shape)
    
    classifier = PointNetDenseCls(k=dataset.npoints, feature_transform=opt.feature_transform, gridsearch=True)
    classifier.cuda()
    batch_pred = classifier(batch_points)
       
    print(model.initialize())
    
    print(nn.MSELoss()(batch_pred, batch_target))
    '''
   
def example(opt):
    opt.dataset = 'T:\\student-homes\\v\\jobdevogeldevo\\My Documents\\Data'
    dataset = IrradianceDataset(
        root=opt.dataset,
        split='train',
        dtype=np.float32,
        normals=False,
        npoints=2500,
        transform=True,
        resample=False
    )
    
    testdataset = IrradianceDataset(
        root=opt.dataset,
        split='test',
        dtype=np.float32,
        normals=False,
        npoints=2500,
        transform=True,
        resample=False
    )
    
    print(f'Dataset size: {len(dataset) + len(testdataset)}')
    
    for i in range(100):
        idx = i
        points, values = dataset[idx]
    
        img = visualize_pointcloud(points.numpy(), values.numpy(), 'C:\\Users\\Job de Vogel\\Desktop\\test.png', visualize=True)
    sys.exit()
    
def main(opt):    
    if not opt.dataset:
        opt.dataset = 'D:\\Master Thesis Data\\raw'
    
    # Set seed
    opt.manualSeed = 0
    
    try:
        os.makedirs(opt.outf)
    except OSError:
        pass
    
    num_epochs = 100
    opt.nepoch = num_epochs
    
    '''
    lr = 0.001
    name = 'gridsearch'
    gridsearch(opt, lr, name)
    '''
    ''''''
    lr = 0.001
    name = 'irradiance_fit'
    fit(opt, lr, name)
    ''''''

if __name__ == '__main__':
    opt = parse_args()
    # example(opt)
    main(opt)
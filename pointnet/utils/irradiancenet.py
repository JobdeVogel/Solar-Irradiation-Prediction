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
from torchvision.io import read_image
from torchvision.utils import make_grid
from pointnet.dataset import ShapeNetDataset, IrradianceDataset
from pointnet.irradiancemodel import PointNetDenseCls, feature_transform_regularizer
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

from skorch import NeuralNetRegressor
from sklearn.model_selection import GridSearchCV

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

def visualize_pointcloud(pointcloud: np.array, color_values: np.array, path: str, visualize=False):   
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pointcloud)
    
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.25, origin=[0, 0, 0])
    
    color_map = plt.get_cmap('coolwarm')
    
    norm = Normalize(vmin=min(color_values), vmax=max(color_values))

    # Map normalized values to colors
    colors = color_map(norm(color_values))[:, :3]       
    
    pcd.colors = o3d.utility.Vector3dVector(colors)    
    
    # Create a visualizer and set view parameters
    visualizer = o3d.visualization.Visualizer()
    visualizer.create_window()
    visualizer.add_geometry(pcd)
    visualizer.add_geometry(coord_frame)

    # Set the view parameters
    view_control = visualizer.get_view_control()
    view_control.set_zoom(0.8)  # Adjust zoom (e.g., zoom_factor > 1 for zoom-in)
    view_control.set_front([1, 1, 1])  # Set the front direction as a 3D vector
    view_control.set_lookat([0, 0, 0])  # Set the point to look at as a 3D vector
    view_control.set_up([0, 0, 1])  # Set the up direction as a 3D vector
    view_control.change_field_of_view(-10)

    # Update and render
    visualizer.poll_events()
    visualizer.update_renderer()
    
    visualizer.capture_screen_image(path)
    
    if visualize:
        visualizer.run()
    
    return read_image(path)

def eval_image(points, classifier, writer, name, path=None):
    if path == None:
        path = './images'
    
    if not os.path.exists(path):
        os.makedirs(path)
    
    eval_points = torch.unsqueeze(points, dim=0)

    eval_points = eval_points.transpose(2, 1)
    
    eval_points_cuda = eval_points.cuda()
    
    with torch.no_grad():        
        classifier = classifier.eval()
        
        pred, _, _ = classifier(eval_points_cuda)
        pred = pred.to('cpu')
    
    start = time.perf_counter()

    path = os.path.join(path, name)
    pixel_array = visualize_pointcloud(points.numpy(), pred.detach().numpy(), path)
    
    grid = make_grid(pixel_array)
    writer.add_image('images', grid)
    print(f'Saved evaluation image in {round(time.perf_counter() - start, 2)}s')
    
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
    classifier.cuda()
    
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
            
            # ! What is this actually doing??
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

def main(opt):
    '''
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
    '''
    
    if not opt.dataset:
        opt.dataset = 'D:\\Master Thesis Data\\raw'
    
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
    
    ##############################################
    dataset = IrradianceDataset(
        root=opt.dataset,
        split='test',
        dtype=np.float32,
        normals=False,
        npoints=2500,
        transform=True,
        resample=True
    )
    
    x = torch.stack([sample[0] for sample in dataset]).permute(0, 2, 1)
    y = torch.stack([sample[1] for sample in dataset])
    
    print('Dataset x an y finished loading...')
    
    model = NeuralNetRegressor(
        module=PointNetDenseCls,
        criterion=nn.MSELoss,
        optimizer=optim.Adam,
        max_epochs=2,
        verbose=3,
        module__k=2500,
        module__feature_transform=True        
    )
    
    param_grid = {
        'batch_size': [16, 32]
        }
    
    grid = GridSearchCV(estimator=model,
                        param_grid=param_grid,
                        n_jobs=-1,
                        cv=3,
                        scoring='neg_mean_squared_error'
                        )

    grid_result = grid.fit(x, y)
    
    # Print results
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))
    

    sys.exit()
    
    
    for bs in batch_sizes:
        for lr in learning_rates:
            opt.batchSize = bs
            name = f'model_lr_{lr}_bs_{bs}'
            
            fit(opt, lr, name)
    

if __name__ == '__main__':      
    main(opt)
import sys
sys.path.append("../")

import os
import random
import math

import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision.utils import make_grid
import torchvision.transforms as transforms
from tqdm.auto import tqdm

from pointnet.dataset import ShapeNetDataset, IrradianceDataset
from pointnet.irradiancemodel import PointNetDenseCls, feature_transform_regularizer

from eval import eval_image, visualize_pointcloud

import argparse

import warnings

import onnx

import pprint

import wandb

# Ensure deterministic behavior
torch.backends.cudnn.deterministic = True
random.seed(hash("setting random seeds") % 2**32 - 1)
np.random.seed(hash("improves reproducibility") % 2**32 - 1)
torch.manual_seed(hash("by removing stochasticity") % 2**32 - 1)
torch.cuda.manual_seed_all(hash("so runs are repeatable") % 2**32 - 1)

# Device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

wandb.login()

def model_pipeline(config=None):
    # tell wandb to get started
    with wandb.init(mode="online", project="wandb_single-run", config=config):
        # access all HPs through wandb.config, so logging matches execution!
        config = wandb.config

        # make the model, data, and optimization problem
        model, train_loader, test_loader, criterion, optimizer, scheduler = make(config)
        
        pipeline_loop(model, train_loader, test_loader, criterion, optimizer, scheduler, config)
        
        '''
        # USE THESE LINES TO RUN TRAIN AND TEST LOOP INDIVIDUALLY
        
        # and use them to train the model
        train(model, train_loader, criterion, optimizer, scheduler, config)
        '''
        '''
        # and test its final performance
        num_test_samples = int(math.floor(len(train_loader) / config.test_interval)) * config.epochs
        test(model, test_loader, criterion, 0, num_test_samples, config)
        '''
        
    return model

def build_criterion(config):
    if config.criterion == 'mse':
        criterion = nn.MSELoss()
    
    return criterion

def build_optimizer(config, model):
    if config.optimizer == 'sgd':
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=config.learning_rate, 
            momentum=0.9)
    if config.optimizer == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=config.learning_rate, 
            betas=(0.9, 0.999)
            )
    
    return optimizer

def build_scheduler(config, optimizer):
    if config.scheduler == 'StepLR':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, 
            step_size=20, 
            gamma=0.5
            )

    return scheduler

def build_model(config):
    if config.architecture == 'PointNet':
        model = PointNetDenseCls(k=config.npoints, feature_transform=config.feature_transform, single_output=config.single_output).to(device)
    
    return model

def make(config):
    # Make the data
    train, test = get_data(config, slice=config.train_slice, train=True), get_data(config, slice=config.test_slice, train=False)
    
    train_loader = make_loader(train, batch_size=config.batch_size)
    test_loader = make_loader(test, batch_size=config.batch_size)
    
    model = build_model(config) 
    
    # Make the loss, optimizer and scheduler
    criterion = build_criterion(config)
    optimizer = build_optimizer(config, model)
    scheduler = build_scheduler(config, optimizer)
    
    return model, train_loader, test_loader, criterion, optimizer, scheduler

def get_data(config, slice=None, train=True):
    if train:
        split = 'train'
    else:
        split = 'test'
    
    dataset = IrradianceDataset(
        root=config.dataset,
        split=split,
        dtype=np.float32,
        slice=slice,
        normals=False,
        npoints=config.npoints,
        transform=True,
        resample=True
    )
    
    return dataset

def make_loader(dataset, batch_size):
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=1,
        pin_memory=True
        )
    
    return dataloader

def pipeline_loop(model, train_loader, test_loader, criterion, optimizer, scheduler, config):
    # Tell wandb to watch what the model gets up to: gradients, weights, and more!
    wandb.watch(model, criterion, log="all", log_freq=10)

    num_train_batches = len(train_loader) * config.epochs
    num_test_batches = int(math.floor(len(train_loader) / config.test_interval)) * config.epochs
    
    # Prepare evaluation sample
    eval_sample = next(iter(test_loader))
    eval_points = eval_sample[0][config.eval_sample, :, :]
    eval_targets = eval_sample[1][config.eval_sample, :]
    
    # Base prediction
    eval_name = f'base_evaluation.png'
    
    path = './images'
    
    if not os.path.exists(path):
        os.makedirs(path)
    
    path = os.path.join(path, eval_name)
    pixel_array = visualize_pointcloud(eval_points.numpy(), eval_targets.numpy(), path)
    
    grid = make_grid(pixel_array)
    
    image = wandb.Image(
        grid,
        caption=f"Base Evaluation"
    )
    
    wandb.log({"Base Irradiance Predicion": image}, step=0)
    
    step = 0

    # Loop
    for epoch in range(config.epochs):
        model.train()
        
        for i, data in enumerate(train_loader, 0):
            points, targets = data
            
            points = points.transpose(2, 1)
            targets = targets.view(-1, 1)[:, 0] - 1
            
            loss, model, optimizer = train_batch(points, targets, model, optimizer, criterion, config)
            
            # Report metrics every nth batch
            if step % config.train_metrics_interval == 0:
                train_log(loss, epoch, step, num_train_batches)

            
            # Test the model
            if step % config.test_interval == 0:
                model.eval()
                
                # Run the model on some test examples
                with torch.no_grad():
                    j, data = next(enumerate(test_loader, 0))
                    points, targets = data
                        
                    points = points.transpose(2, 1)
                    targets = targets.view(-1, 1)[:, 0] - 1
                    
                    loss = test_batch(points, targets, model, criterion, config)
                    
                    test_log(loss, epoch, step, num_train_batches)
            
            # Evaluation
            if step % config.eval_interval == 0:
                eval_name = f'Evaluation_epoch_{str(epoch)}_it_{str(i)}_step_{step}.png'
                grid = eval_image(eval_points, model, eval_name)
                
                image = wandb.Image(
                    grid,
                    caption=f"Irradiance Predication Epoch: {epoch}, it {step}"
                )
                
                wandb.log({"Evaluation Irradiance Predications": image}, step=step)
            
            step += 1
            
        scheduler.step()
        
        print('[%d] Saving dict state...' % (epoch))
        torch.save(model.state_dict(), '%s/irr_model_%s_epoch_%d.pth' % (config.model_outf, wandb.run.name, epoch))
    
    export_model(model, points, 'IrradianceNet', r'models')
    
    return

def train(model, loader, criterion, optimizer, scheduler, config):
    # Tell wandb to watch what the model gets up to: gradients, weights, and more!
    wandb.watch(model, criterion, log="all", log_freq=10)

    model.train()

    # Run training and track with wandb
    total_batches = len(loader) * config.epochs
    num_epoch_batches = len(loader)
    step = 0  # number of examples seen
    batch_ct = 0
    
    for epoch in tqdm(range(config.epochs)):
        for _, data in enumerate(loader, 0):
            points, targets = data
            
            points = points.transpose(2, 1)
            targets = targets.view(-1, 1)[:, 0] - 1
            
            loss, _, _ = train_batch(points, targets, model, optimizer, criterion, config)
            step +=  1
            batch_ct += 1
            
            # Report metrics every 25th batch
            if ((batch_ct + 1) % config.train_metrics_interval) == 0:
                train_log(loss, epoch, step, num_epoch_batches)
            
        scheduler.step()

def train_batch(points, targets, model, optimizer, criterion, config):
    points, targets = points.to(device), targets.to(device)
    
    optimizer.zero_grad()
    
    # Forward pass ➡
    outputs = model(points)
    
    if isinstance(outputs, tuple):
        pred, trans, trans_feat = outputs
    else:
        pred = outputs
       
    loss = criterion(pred, targets)
    
    if config.feature_transform:
        loss += feature_transform_regularizer(trans_feat) * 0.001
    
    # Backward pass ⬅
    loss.backward()
    
    # Step with optimizer
    optimizer.step()

    return loss, model, optimizer

def train_log(loss, epoch, step, num_batch):
    #? Does step work in wandb or should examplect used?
    yellow = lambda x: '\033[93m' + x + '\033[0m'
    
    # Where the magic happens
    wandb.log({"epoch": epoch, "train_loss": loss}, step=step)
    
    print('[Epoch %d: it %s/%s] %s loss: %f' % (epoch, str(step).zfill(3), str(num_batch).zfill(3), yellow('train'), loss.item()))
    
def test(model, loader, criterion, epoch, num_test_samples, config):
    model.eval()

    # Run the model on some test examples
    with torch.no_grad():
        j, data = next(enumerate(loader, 0))
        
        points, targets = data
            
        points = points.transpose(2, 1)
        targets = targets.view(-1, 1)[:, 0] - 1
        
        loss = test_batch(points, targets, model, criterion, config)
        
        step = 0
        test_log(loss, epoch, step, num_test_samples)
 
def test_batch(points, targets, model, criterion, config):
    points, targets = points.to(device), targets.to(device)
    
    # Forward pass ➡
    outputs = model(points)
    
    if isinstance(outputs, tuple):
        pred, trans, trans_feat = outputs
    else:
        pred = outputs
    
    loss = criterion(pred, targets)
    
    if config.feature_transform:
        loss += feature_transform_regularizer(trans_feat) * 0.001
    
    return loss

def test_log(loss, epoch, step, num_batch):
    blue = lambda x: '\033[94m' + x + '\033[0m'
    
    # Where the magic happens
    wandb.log({"epoch": epoch, "test_loss": loss}, step=step)
    
    print('[Epoch %d: it %s/%s] %s loss: %f' % (epoch, str(step).zfill(3), str(num_batch).zfill(3), blue('test'), loss.item()))

def export_model(model, points, name, directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    path = os.path.join(directory, name) + '.onnx'
    
    # Save the model in the exchangeable ONNX format
    points = points.to(device)
        
    input_names = ['points']
    output_names = ["output", "trans", "trans_feat"]
        
    torch.onnx.disable_log()
    torch.onnx.export(model, points, path, input_names=input_names, output_names=output_names)
        
    wandb.save("model.onnx")

def main():
    config = dict(
        epochs=5,
        batch_size=8,
        learning_rate=0.001,
        dataset="D:\\Master Thesis Data\\raw",
        model_outf="seg",
        wandb_outf='C:\\Users\\Job de Vogel\\Desktop\\wandb',
        train_slice=None,
        test_slice=None,
        architecture="PointNet",
        feature_transform=True,
        single_output=False,
        npoints=10000,
        test_interval=10,
        eval_interval=25,
        train_metrics_interval=5,
        eval_sample=0
        )
    
    # Configuration
    sweep_config = {
        'method': 'random'
        }
    
    # Metric
    metric = {
        'name': 'train_loss',
        'goal': 'minimize'   
        }

    sweep_config['metric'] = metric
    
    # Sweep parameters
    parameters_dict = {
        'optimizer': {
            'values': ['adam', 'sgd']
            }
        }
    
    parameters_dict.update({
        'learning_rate': {
            # a flat distribution between 0 and 0.1
            'distribution': 'log_uniform_values',
            'min': 1e-5,
            'max': 1e-2
          },
        'batch_size': {
            # integers between 32 and 256
            # with evenly-distributed logarithms 
            'distribution': 'q_log_uniform_values',
            'q': 8,
            'min': 8,
            'max': 32,
          }
        })
    
    # Non-changing parameters
    parameters_dict.update({
        'dataset': {'value': "D:\\Master Thesis Data\\raw"},
        'epochs': {'value': 1},
        'criterion': {'value': 'mse'},
        'scheduler': {'value': 'StepLR'},
        'model_outf': {'value': "seg"},
        'wandb_outf': {'value': "C:\\Users\\Job de Vogel\\Desktop\\wandb"},
        'train_slice': {'value': None},
        'test_slice': {'value': None},
        'architecture': {'value': "PointNet"},
        'feature_transform': {'value': True},
        'single_output': {'value': False},
        'npoints': {'value': 2500},
        'test_interval': {'value': 10},
        'eval_interval': {'value': 25},
        'train_metrics_interval': {'value': 5},
        'eval_sample': {'value': 0}
        })
    
    sweep_config['parameters'] = parameters_dict
    
    try:
        os.makedirs(config['model_outf'])
    except OSError:
        pass
    
    # Build, train and analyze the model with the pipeline
    # model = model_pipeline(config)   
    sweep_id = wandb.sweep(sweep_config, project="wandb_sweep")
    
    wandb.agent(sweep_id, model_pipeline, count=3)  

if __name__ == '__main__':
    main()    
import sys
sys.path.append("../")

import os
import random
import math

import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm

from pointnet.dataset import ShapeNetDataset, IrradianceDataset
from pointnet.irradiancemodel import PointNetDenseCls, feature_transform_regularizer

from eval import get_im_data, plot, compute_errors

import argparse

import wandb

parser = argparse.ArgumentParser()
parser.add_argument('--cpu', action='store_true', help="run on cpu")
parser.add_argument('--feature_transform', action='store_true', help="feature transform 2")
opt = parser.parse_args()

# Ensure deterministic behavior
torch.backends.cudnn.deterministic = True
random.seed(hash("setting random seeds") % 2**32 - 1)
np.random.seed(hash("improves reproducibility") % 2**32 - 1)
torch.manual_seed(hash("by removing stochasticity") % 2**32 - 1)
torch.cuda.manual_seed_all(hash("so runs are repeatable") % 2**32 - 1)

# Device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if opt.cpu:
    device = "cpu"

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
        normals=config.meta,
        npoints=config.npoints,
        transform=True,
        resample=config.resample,
        preload=config.preload_data
    )
    
    return dataset

def make_loader(dataset, batch_size):
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True
        )
    
    return dataloader

def build_criterion(config):
    if config.criterion == 'mse':
        criterion = nn.MSELoss()
    else:
        print(f'Criterion {config.criterion} is not available')
        sys.exit()
    
    return criterion

def build_optimizer(config, model):
    if config.optimizer == 'sgd':
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=config.learning_rate, 
            momentum=0.9)
    elif config.optimizer == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=config.learning_rate, 
            betas=(0.9, 0.999)
            )
    elif config.optimizer == 'adagrad':
        optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=config.learning_rate, 
            betas=(0.9, 0.999)
            )
        optimizer = torch.optim.Adagrad(
            model.parameters(),
            lr=config.learning_rate
        )
    else:
        print(f'Optimizer {config.optimizer} is not available')
        sys.exit()
    
    return optimizer

def build_scheduler(config, optimizer):
    if config.scheduler == 'StepLR':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, 
            step_size=1, 
            gamma=0.5,
            verbose=True
            )
    else:
        print(f'Scheduler {config.scheduler} is not available')
        sys.exit()

    return scheduler

def build_model(config, m):
    if config.architecture == 'PointNet':
        if config.meta:
            model = PointNetDenseCls(k=config.npoints, m=m, feature_transform=config.feature_transform, single_output=config.single_output).to(device)
        else:
            model = PointNetDenseCls(k=config.npoints, feature_transform=config.feature_transform, single_output=config.single_output).to(device)
    else:
        print(f'Model {config.architecture} is not available')
        sys.exit()
    
    return model

def make(config):
    # Make the data
    train, test = get_data(config, slice=config.train_slice, train=True), get_data(config, slice=config.test_slice, train=False)
    
    train_loader = make_loader(train, batch_size=config.batch_size)
    test_loader = make_loader(test, batch_size=config.batch_size)
    
    m = train[0][0].shape[1] - 3
    model = build_model(config, m)
    
    # Make the loss, optimizer and scheduler
    criterion = build_criterion(config)
    optimizer = build_optimizer(config, model)
    scheduler = build_scheduler(config, optimizer)
    
    return model, train_loader, test_loader, test, criterion, optimizer, scheduler

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
        losses = []
        
        for i, data in enumerate(loader, 0):
            points, targets = data
            points = points.transpose(2, 1)

            targets = targets.view(-1, 1)[:, 0] - 1
            
            loss, _, _ = train_batch(points, targets, model, optimizer, criterion, config)
            
            step +=  1
            batch_ct += 1
            
            losses.append(loss)
            
            # Report metrics every 25th batch
            if ((batch_ct + 1) % config.train_metrics_interval) == 0:
                train_log(loss, epoch, step, num_epoch_batches)
            
        print(f'Avg. loss: {sum(losses) / len(losses)}')
        # scheduler.step()

def train_batch(data, targets, model, optimizer, criterion, config):
    data, targets = data.to(device), targets.to(device)
    
    optimizer.zero_grad()
    
    # ! here we transform the data to points and meta
    if config.meta:
        # There is meta data available
        points = data[:, :3, :]
        meta = data[:, 3:, :]
    else:
        points = data
        meta = None
    
    # Forward pass ➡
    outputs = model(points, meta=meta)

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

def test_batch(data, targets, model, criterion, config):
    data, targets = data.to(device), targets.to(device)
    
    # ! here we transform the data to points and meta
    if config.meta:
        # There is meta data available
        points = data[:, :3, :]
        meta = data[:, 3:, :]
    else:
        points = data
        meta=None
      
    # Forward pass ➡
    outputs = model(points, meta=meta)
    
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

def pipeline_loop(model, train_loader, test_loader, eval_dataset, criterion, optimizer, scheduler, config):        
    # Tell wandb to watch what the model gets up to: gradients, weights, and more!
    wandb.watch(model, criterion, log="all", log_freq=10)
    
    num_train_batches = len(train_loader) * config.epochs
    num_test_batches = int(math.floor(len(train_loader) / config.test_interval)) * config.epochs
    
    ''' --- EVALUATION --- '''
    path = f'./images/{wandb.run.id}/'   
    
    if not os.path.exists(path):
        os.makedirs(path)   
    
    eval_points_plot, eval_vectors_plot, eval_targets_plot = get_im_data(eval_dataset, config.eval_sample)
    
    plot(eval_points_plot, eval_vectors_plot, eval_targets_plot, save=True, save_name='base_idx_0', save_path=path)
    
    wandb.log({"Base Target Irradiance": wandb.Image(os.path.join(path, 'base_idx_0') + '.png')}, step=0)
    
    eval_points = torch.from_numpy(eval_points_plot).reshape(1, 3, -1)
    if len(eval_vectors_plot) > 0:
        eval_vectors = torch.from_numpy(eval_vectors_plot).reshape(1, 3, -1)
    eval_targets = torch.from_numpy(eval_targets_plot)
    ''''''

    step = 0

    # Loop
    for epoch in range(config.epochs):
        model.train()
        
        for i, data in enumerate(train_loader, 0):
            data, targets = data
            
            data = data.transpose(2, 1)
            targets = targets.view(-1, 1)[:, 0] - 1
            
            loss, model, optimizer = train_batch(data, targets, model, optimizer, criterion, config)
            
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
                print(f'[Epoch: {epoch}: it {str(step).zfill(3)}/{num_train_batches}] Evaluating model for epoch {epoch} it {i} step {step} and saving in {path}')
                
                model.eval()
                with torch.no_grad():                   
                    eval_points = eval_points.to(device)
                    
                    if len(eval_vectors_plot) > 0:
                        eval_vectors = eval_vectors.to(device)
                        
                        # Forward pass ➡
                        outputs, _, _ = model(eval_points, meta=eval_vectors)
                        eval_vectors = eval_vectors.to('cpu')
                    else:
                        outputs, _, _ = model(eval_points)         

                outputs = outputs.to('cpu')
                
                save_name = f'Evaluation_epoch_{epoch}_it_{i}_step_{step}'
                plot(eval_points_plot, eval_vectors_plot, outputs, save=True, save_name=save_name, save_path=path)
                wandb.log({f"Evaluation Irradiance Predictions": wandb.Image(os.path.join(path, save_name) + '.png')}, step=step)
                
                save_name = f'Evaluation_epoch_{epoch}_it_{i}_step_{step}_errors'
                errors = compute_errors(eval_targets_plot, outputs)               
                plot(eval_points_plot, eval_vectors_plot, errors, save=True, save_name=save_name, save_path=path, error=True)
                wandb.log({"Evaluation Irradiance Errors": wandb.Image(os.path.join(path, save_name) + '.png')}, step=step)
            
            step += 1
            
        # scheduler.step()
        
        print('[Epoch %d] Saving dict state...' % (epoch))
        torch.save(model.state_dict(), '%s/irr_model_%s_epoch_%d.pth' % (config.model_outf, wandb.run.name, epoch))
    
    export_model(model, points, 'IrradianceNet', r'models')
    
    return

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

def model_pipeline(config=None):        
    # tell wandb to get started
    with wandb.init(mode="online", project="wandb_single-run", config=config):
        # access all HPs through wandb.config, so logging matches execution!
        config = wandb.config

        # # make the model, data, and optimization problem
        model, train_loader, test_loader, eval_dataset, criterion, optimizer, scheduler = make(config)
        
        pipeline_loop(model, train_loader, test_loader, eval_dataset, criterion, optimizer, scheduler, config)

        '''
        # USE THESE LINES TO RUN TRAIN AND TEST LOOP INDIVIDUALLY
        
        # and use them to train the model
        train(model, train_loader, criterion, optimizer, scheduler, config
        # and test its final performance
        num_test_samples = int(math.floor(len(train_loader) / config.test_interval)) * config.epochs
        test(model, test_loader, criterion, 0, num_test_samples, config)
        '''

def main(opt):    
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
            'values': ['adam', 'sgd', 'adagrad']
            },
        'feature_transform': {
            'values': [True, False]
            },
        'meta': {
            'values': [True, False]
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
            'min': 16,
            'max': 64,
          }
        })
    
    # Non-changing parameters
    parameters_dict.update({
        'dataset': {'value': "C:\\Users\\Job de Vogel\\OneDrive\\Documenten\\TU Delft\\Master Thesis\\Code\\IrradianceNet\\data\\BEEST_data\\raw"}, 
        'meta': {'value': True},
        'epochs': {'value': 2},
        'criterion': {'value': 'mse'},
        'scheduler': {'value': 'StepLR'},
        'model_outf': {'value': "seg"},
        'wandb_outf': {'value': "C:\\Users\\Job de Vogel\\Desktop\\wandb"},
        'architecture': {'value': "PointNet"},
        'single_output': {'value': False},
        'npoints': {'value': 2500},
        'test_interval': {'value': 25},
        'eval_interval': {'value': 25},
        'train_metrics_interval': {'value': 1},
        'train_slice': {'value': None},
        'test_slice': {'value': None},
        'resample': {'value': True},
        'eval_sample': {'value': 0},
        'preload_data': {'value': False}
        })
    
    sweep_config['parameters'] = parameters_dict
    
    try:
        os.makedirs(parameters_dict['model_outf']['value'])
    except OSError:
        pass
    
    config = dict(
        epochs=5,
        batch_size=32,
        learning_rate=0.001,
        dataset="C:\\Users\\Job de Vogel\\OneDrive\\Documenten\\TU Delft\\Master Thesis\\Code\\IrradianceNet\\data\\BEEST_data\\raw",
        model_outf="seg",
        wandb_outf='C:\\Users\\Job de Vogel\\Desktop\\wandb',
        train_slice=200,
        test_slice=None,
        architecture="PointNet",
        optimizer='adam',
        criterion='mse',
        scheduler='StepLR',
        feature_transform=opt.feature_transform,
        npoints=2500,
        single_output=False,
        test_interval=25,
        eval_interval=25,
        train_metrics_interval=1,
        eval_sample=0,
        resample=True,
        meta=True,
        preload_data=False
        )
    
    try:
        os.makedirs(config['model_outf'])
    except OSError:
        pass
    
    # Build, train and analyze the model with the pipeline
    model = model_pipeline(config)   
    # sweep_id = wandb.sweep(sweep_config, project="wandb_sweep-normals")
    # wandb.agent(sweep_id, model_pipeline, count=1)

if __name__ == '__main__':
    wandb.login()
    
    main(opt)


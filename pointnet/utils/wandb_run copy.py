import sys
sys.path.append("../")

import os
import time

import numpy as np
import torch
import torch.nn as nn

from pointnet.dataset import IrradianceDataset
from pointnet.irradiancemodel import PointNetDenseCls, feature_transform_regularizer, init_weights

from eval import get_im_data, plot, compute_errors

import argparse

import wandb
import itertools

parser = argparse.ArgumentParser()
parser.add_argument('--cpu', action='store_true', help="run on cpu")
parser.add_argument('--gpu', type=int, nargs='?', default=0, help="cuda device idx, defaults to 0, or 'parallel'")
parser.add_argument('--feature_transform', action='store_true', help="feature transform 2")
opt = parser.parse_args()

# Device configuration
device = torch.device(f"cuda:{opt.gpu}" if torch.cuda.is_available() else "cpu")

if opt.cpu:
    device = "cpu"

if opt.gpu == 'parallel':
    device = "cuda"

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
        preload=config.preload_data,
        lex_sort=config.lex_sort
    )

    return dataset

def make_loader(dataset, batch_size, workers=0):
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
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
            step_size=30, 
            gamma=0.5,
            verbose=True
            )
    elif config.scheduler == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.1,
            patience=1,
            verbose=True
        )
    else:
        print(f'Scheduler {config.scheduler} is not available')
        sys.exit()

    return scheduler

def build_model(config, m, parallel=False):
    if config.k == 0:
        n = 64
    else:
        n = config.k

    if config.architecture == 'PointNet':
        if config.meta:
            model = PointNetDenseCls(k=config.npoints, m=m, n=n, config=config, device=device, feature_transform=config.feature_transform).to(device)
        else:
            model = PointNetDenseCls(k=config.npoints, n=n, config=config, device=device, feature_transform=config.feature_transform).to(device)
    else:
        print(f'Model {config.architecture} is not available')
        sys.exit()
    
    if parallel:
        model = nn.DataParallel(model)

    return model

def make(config):
    # Make the data
    train, test = get_data(config, slice=config.train_slice, train=True), get_data(config, slice=config.test_slice, train=False)
    
    train_loader = make_loader(train, batch_size=config.batch_size, workers=0)
    test_loader = make_loader(test, batch_size=config.batch_size, workers=0)
    
    m = train[0][0].shape[1] - 3
    model = build_model(config, m)

    model.apply(lambda x: init_weights(x, config=config))
    
    # Make the loss, optimizer and scheduler
    criterion = build_criterion(config)
    optimizer = build_optimizer(config, model)
    scheduler = build_scheduler(config, optimizer)
    
    return model, train_loader, test_loader, test, criterion, optimizer, scheduler

def train_batch(inputs, targets, model, optimizer, criterion, config):
    inputs = inputs.transpose(2, 1)

    points = inputs[:, :3, :]
    meta = inputs[:, 3:, :]

    # Move data to the GPU
    points = points.to(device)
    targets = targets.to(device)

    if config.meta:
        meta = meta.to(device)

    optimizer.zero_grad()

    if config.meta:
        # Forward pass
        outputs, trans, trans_feat = model(points, meta=meta)
    else:
        outputs, trans, trans_feat = model(points, meta=None)

    loss = criterion(outputs, targets)

    if config.feature_transform:
        loss += feature_transform_regularizer(trans_feat, device=device) * 0.001

    loss.backward()

    optimizer.step()

    return loss

def test_batch(inputs, targets, model, criterion, config):
    inputs = inputs.transpose(2, 1)

    points = inputs[:, :3, :]
    meta = inputs[:, 3:, :]

    # Move data to the GPU
    points = points.to(device)
    targets = targets.to(device)

    if config.meta:
        meta = meta.to(device)

    if config.meta:
        # Forward pass
        outputs, trans, trans_feat = model(points, meta=meta)
    else:
        outputs, trans, trans_feat = model(points, meta=None)

    loss = criterion(outputs, targets)

    if config.feature_transform:
        loss += feature_transform_regularizer(trans_feat, device=device) * 0.001

    return loss

def evaluate(inputs, targets, model, epoch, i, max_i, path, config):
    eval_points = inputs[:, :3]
    eval_meta = inputs[:, 3:]
    
    inputs = inputs.unsqueeze(dim=0).transpose(2, 1)

    points = inputs[:, :3, :]
    meta = inputs[:, 3:, :]

    # Move data to the GPU
    points = points.to(device)

    if config.meta:
        meta = meta.to(device)

    if config.meta:
        # Forward pass
        outputs, _, _ = model(points, meta=meta)
    else:
        outputs, _, _ = model(points, meta=None)
    
    outputs = outputs.to('cpu').view(-1)

    step = i + epoch * max_i

    try:
        save_name = f'Evaluation_epoch_{epoch}_it_{i}_step_{step}'
        
        plot(eval_points, eval_meta, outputs, save=True, save_name=save_name, save_path=path)
        wandb.log({f"Evaluation Irradiance Predictions": wandb.Image(os.path.join(path, save_name) + '.png')}, step=step)

        save_name = f'Evaluation_epoch_{epoch}_it_{i}_step_{step}_errors'
        errors = compute_errors(targets, outputs)               
        
        plot(eval_points, eval_meta, errors, save=True, save_name=save_name, save_path=path, error=True)
        wandb.log({"Evaluation Irradiance Errors": wandb.Image(os.path.join(path, save_name) + '.png')}, step=step)
    except Exception as e:
        print(e)
        print('WARNING: failed to generate image in run')
        pass

def pipeline_loop(model, train_loader, test_loader, eval_dataset, criterion, optimizer, scheduler, config):        
    # Tell wandb to watch what the model gets up to: gradients, weights, and more!
    wandb.watch(model, criterion, log="all", log_freq=10)
    
    ''' --- EVALUATION --- '''
    path = f'./images/{wandb.run.id}/'   
    
    if not os.path.exists(path):
        os.makedirs(path)   
    
    eval_points_plot, eval_meta_plot, eval_targets_plot = get_im_data(eval_dataset, config.eval_sample)
    
    plot(eval_points_plot, eval_meta_plot, eval_targets_plot, save=True, save_name='Base_Target', save_path=path)
    
    wandb.log({"Evaluation Irradiance Predictions": wandb.Image(os.path.join(path, 'Base_Target') + '.png')}, step=0)
    
    eval_inputs = torch.from_numpy(eval_points_plot)
    if config.meta:
        eval_meta = torch.from_numpy(eval_meta_plot)
        eval_inputs = torch.cat((eval_inputs, eval_meta), dim=1)
    
    eval_targets = torch.from_numpy(eval_targets_plot)
    ''''''

    print(f'Loading train samples with dataloader on {train_loader.num_workers} cpus (test dataloader is using workers={test_loader.num_workers})')
    test_loader = itertools.cycle(test_loader)

    model.to(device)
    model = model.train()

    export_model(model, eval_inputs, 'IrradianceNet', r'models')

    step = 0
    # Loop
    for epoch in range(config.epochs):
        losses = []

        for i, data in enumerate(train_loader, 0):
            if i == 0  and epoch == 0:
                start = time.time()

            inputs, targets, idxs = data

            loss = train_batch(inputs, targets, model, optimizer, criterion, config)
            losses.append(loss)

            if i % config.train_metrics_interval == 0 and i != 0:
                avg_loss = sum(losses) / config.train_metrics_interval
                
                #? Does step work in wandb or should examplect used?
                yellow = lambda x: '\033[93m' + x + '\033[0m'

                # Calculate the step
                step = i + epoch * len(train_loader)

                # Where the magic happens
                wandb.log({"Epoch": epoch, "avg_train_loss": avg_loss}, step=step)
                
                print('[Epoch %d: it %s/%s] %s loss: %f' % (epoch, str(i).zfill(3), str(len(train_loader)).zfill(3), yellow('train'), avg_loss))

                losses = []
            elif i == 0:
                losses = []

            # Test the model
            if step % config.test_metrics_interval == 0:
                model = model.eval()
                
                # Run the model on some test examples
                with torch.no_grad():
                    inputs, targets, idxs = next(test_loader)

                    loss = test_batch(inputs, targets, model, criterion, config)
                    
                    blue = lambda x: '\033[94m' + x + '\033[0m'
                    print('[Epoch %d: it %s/%s] %s loss: %f' % (epoch, str(i).zfill(3), str(len(train_loader)).zfill(3), blue('test '), loss))

                    # Calculate the step
                    step = i + epoch * len(train_loader)

                    # Where the magic happens
                    wandb.log({"Epoch": epoch, "test_loss": loss}, step=step)
                                    
                model = model.train()

            try:
                # Evaluation
                if step % config.eval_interval == 0 and (i != 0 and epoch != 0):                    
                    model = model.eval()
                    
                    print(f'Evaluating [Epoch {epoch}: it. {i}] on test sample {config.eval_sample} ...')
                    start = time.perf_counter()
                    with torch.no_grad():
                        evaluate(eval_inputs, eval_targets, model, epoch, i, len(train_loader), path, config)
                    print(f'... Finished evaluating [Epoch {epoch}: it. {i}] in {round(time.perf_counter() - start, 2)}s')

                    model = model.train()
            except:
                model = model.train()
                pass
            step += 1

        scheduler.step()
        
        print('[Epoch %d] Saving dict state...' % (epoch))
        seg_path = os.path.join(config.model_outf, wandb.run.id)

        if not os.path.exists(seg_path):
            os.makedirs(seg_path)

        torch.save(model.state_dict(), '%s/irr_model_%s_epoch_%d.pth' % (seg_path, wandb.run.name, epoch))
    
    return

def export_model(model, points, name, directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    path = os.path.join(directory, name) + '.onnx'
    
    x = points.unsqueeze(dim=0).transpose(2, 1)

    meta = x[:, 3:, :].to(device)
    x = x[:, :3, :].to(device)
    
    inputs = (x, meta)
    
    # Save the model in the exchangeable ONNX format        
    input_names = ['points', 'meta']
    output_names = ["output", "trans"]
        
    torch.onnx.disable_log()
    torch.onnx.export(model, inputs, path, input_names=input_names, output_names=output_names)
        
    wandb.save("model.onnx")

def model_pipeline(config=None):        
    # tell wandb to get started
    with wandb.init(mode="online", project="v8", config=config, allow_val_change=True):        
        # access all HPs through wandb.config, so logging matches execution!
        config = wandb.config

        if config.k == 0:
            config.update({'feature_transform': False})
        else:
            config.update({'feature_transform': True})

        # # make the model, data, and optimization problem
        model, train_loader, test_loader, eval_dataset, criterion, optimizer, scheduler = make(config)

        # print(model)
        
        pipeline_loop(model, train_loader, test_loader, eval_dataset, criterion, optimizer, scheduler, config)        

def main(opt, config=None):    
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
    
    early_terminate = {
        'type': 'hyperband',
        'min_iter': 100
    }
    
    sweep_config['early_terminate'] = early_terminate
    
    # Sweep parameters
    parameters_dict = {
        'optimizer': {
            'values': ['adam']
            },
        'meta': {
            'values': [True],
            'probabilities': [1]
            },
        'initialization': {
            'values': [None]
            },
        'k': {
            'values': [0],
            'probabilities': [1]
            },
        'batch_size': {
            'values': [32]
            },
        'scheduler': {
            'values': ['StepLR']
        }
    }
    
    parameters_dict.update({
        'learning_rate': {
            # a flat distribution between 0 and 0.1
            'distribution': 'log_uniform_values',
            'min': 1e-5,
            'max': 1e-4
            }
        })

    # Non-changing parameters
    for hp in config:
        if str(hp) not in parameters_dict.keys():
            parameters_dict.update({
                str(hp): {'value': config[str(hp)]}
            })

    sweep_config['parameters'] = parameters_dict

    try:
        os.makedirs(parameters_dict['model_outf']['value'])
    except OSError:
        pass
    
    # Build, train and analyze the model with the pipeline
    model = model_pipeline(config)

    # project = "v8"
    # sweep_id = wandb.sweep(sweep_config, project=project)
    # # #sweep_id = '85x58gz3
    # wandb.agent(sweep_id, model_pipeline, count=5, project=project)

if __name__ == '__main__':
    wandb.login()
    
    config = dict(
        epochs=5,
        batch_size=32,
        learning_rate=0.001,
        dataset="C:\\Users\\Job de Vogel\\OneDrive\\Documenten\\TU Delft\\Master Thesis\\Code\\IrradianceNet\\data\\BEEST_data",
        model_outf="seg",
        wandb_outf='\\tudelft.net\student-homes\V\jobdevogeldevo\Desktop\\wandb',
        train_slice=7680, #7584,
        test_slice=None,
        architecture="PointNet",
        optimizer='adam',
        criterion='mse',
        scheduler='StepLR',
        npoints=2500,
        eval_interval=250,
        train_metrics_interval=10,
        test_metrics_interval=100,
        eval_sample=0,
        resample=True,
        meta=True,
        preload_data=False,
        hidden_layers=[],
        initialization=None,
        lex_sort=False,
        k=0
        )
    
    if opt.feature_transform:
        print(f'Running script on {device} WITH feature transform')
    else:
        print(f'Running script on {device} WITHOUT feature transform')
    
    main(opt, config=config)


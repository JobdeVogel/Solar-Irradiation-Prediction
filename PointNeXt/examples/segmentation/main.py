"""
(Distributed) training script for scene segmentation
This file currently supports training and testing on S3DIS
If more than 1 GPU is provided, will launch multi processing distributed training by default
if you only wana use 1 GPU, set `CUDA_VISIBLE_DEVICES` accordingly
"""
import sys
import time
from datetime import datetime

import __init__
import argparse, yaml, os, logging, numpy as np, csv, wandb, glob
from tqdm import tqdm
import torch, torch.nn as nn
from torch import distributed as dist, multiprocessing as mp
from openpoints.utils import set_random_seed, save_checkpoint, load_checkpoint, resume_checkpoint, setup_logger_dist, \
    cal_model_parm_nums, Wandb, generate_exp_directory, resume_exp_directory, EasyConfig, dist_utils, find_free_port
from openpoints.utils import AverageMeter
from openpoints.dataset import build_dataloader_from_cfg, get_features_by_keys, get_class_weights
from openpoints.dataset.data_util import voxelize
from openpoints.dataset.semantic_kitti.semantickitti import load_label_kitti, load_pc_kitti, remap_lut_read, remap_lut_write, get_semantickitti_file_list
from openpoints.transforms import build_transforms_from_cfg
from openpoints.optim import build_optimizer_from_cfg
from openpoints.scheduler import build_scheduler_from_cfg
from openpoints.loss import build_criterion_from_cfg
from openpoints.models import build_model_from_cfg
import warnings
import shutil

from visualize import from_sample, plot, binned_cm

warnings.simplefilter(action='ignore', category=FutureWarning)

def main(gpu, cfg):
    # if cfg.distributed:
    #     if cfg.mp:
    #         cfg.rank = gpu
    #     dist.init_process_group(backend=cfg.dist_backend,
    #                             init_method=cfg.dist_url,
    #                             world_size=cfg.world_size,
    #                             rank=cfg.rank)
    #     dist.barrier()

    if cfg.criterion_args.NAME.lower() == 'weightedmse':
        cfg.criterion_args.bins = 5
        cfg.criterion_args.min = -1
        cfg.criterion_args.max = 1
        cfg.criterion_args.weights = [1,1,1,1,0.25]
    
    if cfg.criterion_args.NAME.lower() == 'deltaloss':
        cfg.criterion_args.delta = 0.6
        cfg.criterion_args.power = 2
    
    if cfg.criterion_args.NAME.lower() == 'reductionloss':
        cfg.criterion_args.bins = 5
        cfg.criterion_args.min = -1
        cfg.criterion_args.max = 1
        cfg.criterion_args.reduction = 1


    # logger
    setup_logger_dist(cfg.log_path, cfg.rank, name=cfg.dataset.common.NAME)
    
    # if cfg.rank == 0:
    #     #if not cfg.wandb.sweep:        
    #     #Wandb.launch(cfg, cfg.wandb.use_wandb)
    #     writer = SummaryWriter(log_dir=cfg.run_dir) if cfg.is_training else None
    # else:
    #     writer = None
    set_random_seed(cfg.seed, deterministic=cfg.deterministic)
    torch.backends.cudnn.enabled = False
    
    # ! Commented
    #logging.info(cfg)

    if cfg.model.get('in_channels', None) is None:
        cfg.model.in_channels = cfg.model.encoder_args.in_channels
    
    model = build_model_from_cfg(cfg.model).to(cfg.rank)
    model_size = cal_model_parm_nums(model)
    
    logging.info(f'Cfg parameters:')
    logging.info(cfg)
    
    logging.info(model)
    logging.info('Number of params: %.4f M' % (model_size / 1e6))

    if cfg.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        
        # ! commented
        # logging.info('Using Synchronized BatchNorm ...')
    if cfg.distributed:
        torch.cuda.set_device(gpu)
        # model = nn.parallel.DistributedDataParallel(model.cuda(), device_ids=[cfg.rank], output_device=cfg.rank)
        logging.info(f"Model is using {cfg.world_size} gpus in DatalParallel mode!")
        model = nn.parallel.DataParallel(model.cuda())
        
        # ! commented
        # logging.info('Using Distributed Data parallel ...')

    
    # optimizer & scheduler
    optimizer = build_optimizer_from_cfg(model, lr=cfg.lr, **cfg.optimizer)    
    scheduler = build_scheduler_from_cfg(cfg, optimizer)  
   
    
    # build dataset
    val_loader, val_histogram = build_dataloader_from_cfg(cfg.get('val_batch_size', cfg.batch_size),
                                            cfg.dataset,
                                            cfg.dataloader,
                                            datatransforms_cfg=cfg.datatransforms,
                                            split='val',
                                            distributed=False
                                            )      

    # ! commented
    # logging.info(f"length of validation dataset: {len(val_loader.dataset)}")
    num_classes = val_loader.dataset.num_classes if hasattr(val_loader.dataset, 'num_classes') else None
    
    if num_classes is not None:
        assert cfg.num_classes == num_classes
    
    # ! commented
    # logging.info(f"number of classes of the dataset: {num_classes}")
    cfg.classes = val_loader.dataset.classes if hasattr(val_loader.dataset, 'classes') else np.arange(num_classes)
    
    cfg.cmap = np.array(val_loader.dataset.cmap) if hasattr(val_loader.dataset, 'cmap') else None
    validate_fn = validate
    
    """
    # import matplotlib
    # import matplotlib.pyplot as plt
    # matplotlib.use('TkAgg')
    # tensors = []
    
    # data = iter(val_loader)
    # for _ in range(len(data)):
    #     tensors.append(next(data)['y'].squeeze(0))
    
    # tensor = torch.cat(tensors)
    # tensor = ((tensor + 1) / 2) * 1000
    
    # bins = 8
    # # Creating histogram
    # histogram = torch.histc(tensor, bins=bins)
    
    # # Calculate bin edges
    # bin_edges = torch.linspace(tensor.min(), tensor.max(), bins+1)
    # names = [str(int(bin_edges[i].item())) + '-' + str(int(bin_edges[i+1].item())) for i in range(len(bin_edges[:-1]))]

    # # Plotting the histogram
    # plt.bar(names, histogram)
    # plt.xlabel('Solar Irradiance [kWh/m2]')
    # plt.ylabel('Point frequency')
    # plt.title('Solar irradiance distribution over points')
    
    # plt.show()
    # sys.exit()
    """
    
    # optionally resume from a checkpoint
    model_module = model.module if hasattr(model, 'module') else model
    
    if cfg.pretrained_path is not None:
        if cfg.mode == 'resume':
            resume_checkpoint(cfg, model, optimizer, scheduler, pretrained_path=cfg.pretrained_path)
    else:
        logging.info('Training from scratch')

    if 'freeze_blocks' in cfg.mode:
        for p in model_module.encoder.blocks.parameters():
            p.requires_grad = False

    
    train_loader, train_histogram = build_dataloader_from_cfg(cfg.batch_size,
                                                cfg.dataset,
                                                cfg.dataloader,
                                                datatransforms_cfg=cfg.datatransforms,
                                                split='train',
                                                distributed=False,
                                                )
    
    logging.info(f"length of training dataset: {len(train_loader.dataset)}")



    if not cfg.regression:
        cfg.criterion_args.weight = None
        if cfg.get('cls_weighed_loss', False):
            if hasattr(train_loader.dataset, 'num_per_class'):
                cfg.criterion_args.weight = get_class_weights(train_loader.dataset.num_per_class, normalize=True)
            else:
                logging.info('`num_per_class` attribute is not founded in dataset')

    if cfg.criterion_args.NAME.lower() == 'reductionloss':
        cfg.criterion_args.histogram = train_histogram
        
        if train_histogram == None:
            print("Reduction loss requires a valid train histogram, which is only available when dataset preprocessing is enabled.")
            raise RuntimeError
    
    
    criterion = build_criterion_from_cfg(cfg.criterion_args).cuda()
    
    mse_criterion = torch.nn.MSELoss().cuda()
    
    
    # ===> start training
    if cfg.use_amp:
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None
    best_val, best_epoch = float('inf'), 0
    test_array = iter(val_loader)
    
    evaluation_test_array_0 = next(test_array)
    evaluation_test_array_1 = next(test_array)
    evaluation_test_array_2 = next(test_array)
    evaluation_test_array_3 = next(test_array)
    evaluation_test_array_4 = next(test_array)
    evaluation_train_array = next(iter(train_loader))
    
    total_iter = 0
    
    if cfg.wandb.use_wandb:
        wandb.watch(model, criterion, log="parameters", log_freq=1000)
        
    from_date = "{:%Y_%m_%d_%H_%M_%S}".format(datetime.now())
    image_dir = f'.\\data\\images\\{cfg.cfg_basename}\\{from_date}\\'
    
    if not os.path.exists(image_dir + '\\evaluation'):
        os.makedirs(image_dir + '\\evaluation')
    
    if not os.path.exists(image_dir + '\\training'):
        os.makedirs(image_dir + '\\training')
    
    logging.info('Logging initial images...')
    max_images = min([5, cfg.batch_size])
    max_evaluation_images = 5
    
    for idx in range(max_images):
        if idx == 0:
            image_path_0 = eval_image(model, evaluation_test_array_0, idx, f'Epoch base test 0 sample {idx}', image_dir + '\\evaluation')
            image_path_1 = eval_image(model, evaluation_test_array_1, idx, f'Epoch base test 1 sample {idx}', image_dir + '\\evaluation')
            image_path_2 = eval_image(model, evaluation_test_array_2, idx, f'Epoch base test 2 sample {idx}', image_dir + '\\evaluation')
            image_path_3 = eval_image(model, evaluation_test_array_3, idx, f'Epoch base test 3 sample {idx}', image_dir + '\\evaluation')
            image_path_4 = eval_image(model, evaluation_test_array_4, idx, f'Epoch base test 4 sample {idx}', image_dir + '\\evaluation')
            
            if cfg.wandb.use_wandb:
                wandb.log({f"Evaluation Irradiance Predictions 0": wandb.Image(image_path_0 + '.png')}, step=0)
                wandb.log({f"Evaluation Irradiance Predictions 1": wandb.Image(image_path_1 + '.png')}, step=0)
                wandb.log({f"Evaluation Irradiance Predictions 2": wandb.Image(image_path_2 + '.png')}, step=0)
                wandb.log({f"Evaluation Irradiance Predictions 3": wandb.Image(image_path_3 + '.png')}, step=0)
                wandb.log({f"Evaluation Irradiance Predictions 4": wandb.Image(image_path_4 + '.png')}, step=0)
        image_path = eval_image(model, evaluation_train_array, idx, f'Epoch 0 train sample {idx}', image_dir + '\\training')
        
        if cfg.wandb.use_wandb:
            wandb.log({f"Train Irradiance Predictions {idx}": wandb.Image(image_path + '.png')}, step=0)
    if cfg.wandb.use_wandb:
        wandb.log({'crit': str(cfg.criterion_args.NAME)}, step=0)
        wandb.log({'model': str(cfg.cfg_basename)}, step=0)
        wandb.log({'optim': str(cfg.optimizer.NAME)}, step=0)
        wandb.log({'sched': str(cfg.sched)}, step=0)
        wandb.log({'batchsize': cfg.batch_size}, step=0)
     
    logging.info(f'Started training {cfg.cfg_basename} with criterion {cfg.criterion_args.NAME}, voxelsize {cfg.dataset.train.voxel_max}, batchsize {cfg.batch_size}...')
    for epoch in range(cfg.start_epoch, cfg.epochs + 1):
        # # ! Only important for distributed gpu
        # if cfg.distributed:
        #     train_loader.sampler.set_epoch(epoch)
        # if hasattr(train_loader.dataset, 'epoch'):  # some dataset sets the dataset length as a fixed steps.
        #     train_loader.dataset.epoch = epoch - 1
        
        train_loss, train_rmse, total_iter = \
            train_one_epoch(model, train_loader, criterion, mse_criterion, optimizer, scheduler, scaler, epoch, total_iter, cfg)
        
        # ! Log the results from the epoch step
        is_best = False
        
        logging.info(f"Started evalution epoch {epoch}")
        if epoch % cfg.val_freq == 0:
            eval_loss, eval_rmse = validate_fn(model, val_loader, criterion, mse_criterion, cfg, epoch=epoch, total_iter=total_iter, image_dir=image_dir)
            
            if eval_loss < best_val:
                logging.info("Found new best model!")
                is_best = True
                best_val = eval_loss
        
        # ! Log to the writer
        lr = optimizer.param_groups[0]['lr']
        
        logging.info(f'Logging images for epoch {epoch}')
        
        max_images = min([5, cfg.batch_size])
        for idx in range(max_images):
            if idx == 0:
                image_path_0 = eval_image(model, evaluation_test_array_0, idx, f'Epoch {epoch} test 0 sample {idx}', image_dir + '\\evaluation')
                image_path_1 = eval_image(model, evaluation_test_array_1, idx, f'Epoch {epoch} test 1 sample {idx}', image_dir + '\\evaluation')
                image_path_2 = eval_image(model, evaluation_test_array_2, idx, f'Epoch {epoch} test 2 sample {idx}', image_dir + '\\evaluation')
                image_path_3 = eval_image(model, evaluation_test_array_3, idx, f'Epoch {epoch} test 3 sample {idx}', image_dir + '\\evaluation')
                image_path_4 = eval_image(model, evaluation_test_array_4, idx, f'Epoch {epoch} test 4 sample {idx}', image_dir + '\\evaluation')
                
                if cfg.wandb.use_wandb:
                    wandb.log({f"Evaluation Irradiance Predictions 0": wandb.Image(image_path_0 + '.png')})
                    wandb.log({f"Evaluation Irradiance Predictions 1": wandb.Image(image_path_1 + '.png')})
                    wandb.log({f"Evaluation Irradiance Predictions 2": wandb.Image(image_path_2 + '.png')})
                    wandb.log({f"Evaluation Irradiance Predictions 3": wandb.Image(image_path_3 + '.png')})
                    wandb.log({f"Evaluation Irradiance Predictions 4": wandb.Image(image_path_4 + '.png')})
            image_path = eval_image(model, evaluation_train_array, idx, f'Epoch {epoch} train sample {idx}', image_dir + '\\training')
            
            if cfg.wandb.use_wandb:
                wandb.log({f"Train Irradiance Predictions {idx}": wandb.Image(image_path + '.png')})
    
        if epoch % cfg.val_freq == 0:
            logging.info(f'Epoch {epoch} LR {lr:.6f} '
                     f'train loss {train_loss:.2f}, eval loss {eval_loss:.2f}')
        else:
            logging.info(f'Epoch {epoch} LR {lr:.6f} '
                     f'train loss {train_loss:.2f}')
                  
        if epoch % cfg.val_freq == 0:
            wandb.log({'Evaluation Loss (mse)': eval_loss})
            wandb.log({'Evaluation Loss (rmse) [kWh/m2]': eval_rmse})
        
        # writer.add_scalar('train_loss', train_loss, epoch)
        # writer.add_scalar('RMSE per train step [kWh/m2]', train_rmse, epoch)
        wandb.log({'Learning Rate': lr})
         # ! Update the optimizer with scheduler
        if cfg.sched_on_epoch:
            if cfg.sched.lower() != 'plateau':
                scheduler.step(epoch)
            else:
                scheduler.step(epoch, metric=train_loss)
         # ! Save model parameters to file
        if cfg.rank == 0:
            save_checkpoint(cfg, model, epoch, optimizer, scheduler,
                            additioanl_dict={'best_val': best_val},
                            is_best=is_best,
                            post_fix=f'ckpt_epoch_{epoch}'
                            )
            is_best = False
        
        if epoch == cfg.max_epoch:
            logging.info('Early finish!')
            break
    
    # Test the model using the test dataset
    test(cfg, model, cfg.dataset.test.data_root)
    
    wandb.finish(exit_code=True)
    
    # do not save file to wandb to save wandb space
    # Wandb.add_file(os.path.join(cfg.ckpt_dir, f'{cfg.run_name}_ckpt_best.pth'))
    # Wandb.add_file(os.path.join(cfg.ckpt_dir, f'{cfg.logname}_ckpt_latest.pth'))

    # # validate
    # with np.printoptions(precision=2, suppress=True):
    #     if epoch % cfg.val_freq == 0:
    #         logging.info(
    #             f'Best ckpt @E{best_epoch},  eval loss {eval_loss:.2f}, train loss {train_loss:.2f}')
    #     else:
    #         logging.info(
    
    # ! Removed testing with sphere_validation

    # dist.destroy_process_group() # comment this line due to https://github.com/guochengqian/PointNeXt/issues/95
    wandb.finish(exit_code=True)

def train_one_epoch(model, train_loader, criterion, mse_criterion, optimizer, scheduler, scaler, epoch, total_iter, cfg):
    loss_meter = AverageMeter()
    rmse_meter = AverageMeter()
     
    model.train()  # set model to training mode
    pbar = tqdm(enumerate(train_loader), total=train_loader.__len__())

    '''
    individual_criterion = nn.MSELoss(reduction='none')
    '''
    
    num_iter = 0
    loss = torch.Tensor([0.0])
    rmse = torch.Tensor([0.0])
    mse_loss = torch.Tensor([0.0])
    
    for idx, data in pbar:
        pbar.set_description(f"Average loss: {format(round(loss_meter.avg, 4), '.4f')}, Average RMSE: {format(round(rmse_meter.avg, 4), '.4f')} [kWh/m2], Loss: {round(loss.item(), 4)}, MSE: {round(mse_loss.item(), 4)}, RMSE: {round(rmse.item(), 4)} [kWh/m2]")
        pbar.refresh()
        
        keys = data.keys() if callable(data.keys) else data.keys
        
        # ! Send all data to GPU
        for key in keys:
            data[key] = data[key].cuda(non_blocking=True)

        num_iter += 1
        
        target = data['y'].squeeze(-1)

        """ debug
        from openpoints.dataset import vis_points
        vis_points(data['pos'].cpu().numpy()[0], labels=data['y'].cpu().numpy()[0])
        vis_points(data['pos'].cpu().numpy()[0], data['x'][0, :3, :].transpose(1, 0))
        end of debug """
        # ! Combine all the feature, so pos and heigts are combined to shape (batchsize, channels, points) (32, 4, 24000)
        data['x'] = get_features_by_keys(data, cfg.feature_keys)
        
        # ! Overwrite data['x'] with number of model channgels
        data['x'] = data['x'][:,:cfg.model.in_channels,:]
        
        # ! Set the epoch in the data
        data['epoch'] = epoch
        
        total_iter += 1 
        
        # ! Set the iteration number in the data
        data['iter'] = total_iter 
              
        with torch.cuda.amp.autocast(enabled=cfg.use_amp):
            # ! Cast all the data to the model
            logits = model(data)
            
            logits = logits.squeeze(1)
            
            '''
            loss is used for backwards pass
            mse_loss is used for performance comparison
            '''
            
            if cfg.criterion_args.NAME.lower() == 'weightedmse' or cfg.criterion_args.NAME.lower() == 'reductionloss':
                loss = criterion(logits, target, bins=data['bins'])
            elif 'mask' in cfg.criterion_args.NAME.lower():
                loss = criterion(logits, target, data['mask'])
            else:
                loss = criterion(logits, target)
                    
            mse_loss = mse_criterion(logits, target)
            
            wandb.log({'Train Loss (non-MSE)': loss})
            wandb.log({'Train Loss MSE': mse_loss})
            
            if cfg.regression:
                rmse = torch.sqrt(mse_criterion(((logits + 1) / 2) * 1000, ((target + 1) / 2) * 1000))
                rmse_meter.update(rmse.item())
                
                wandb.log({'Train Loss RMSE [kWh/m2]': rmse})
                '''
                individual_losses = torch.mean(individual_criterion(logits, target), 1)
                
                for j, (ind, index) in enumerate(zip(individual_losses, data['idx'])):
                    writer.add_scalar('mse_per_train_sample', ind, total_iter * cfg.batch_size + j)
                    writer.add_scalar('sample_idx', index, total_iter * cfg.batch_size + j)
                '''
        if cfg.use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # optimize
        if num_iter == cfg.step_per_update:
            if cfg.get('grad_norm_clip') is not None and cfg.grad_norm_clip > 0.:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_norm_clip, norm_type=2)
            num_iter = 0

            if cfg.use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            optimizer.zero_grad()
            if not cfg.sched_on_epoch:
                if cfg.sched != 'plateau':
                    scheduler.step(epoch)
                else:
                    scheduler.step(epoch, metric=loss)
            
            # mem = torch.cuda.max_memory_allocated() / 1024. / 1024.
            # print(f"Memory after backward is {mem}")
        
        loss_meter.update(loss.item())      
    
    return loss_meter.avg, rmse_meter.avg, total_iter

def cm(targets, predictions):
    pass

@torch.no_grad()
def validate(model, val_loader, criterion, mse_criterion, cfg, num_votes=1, data_transform=None, epoch=-1, total_iter=-1, image_dir=''):   
    model.eval()  # set model to eval mode
    
    loss = torch.Tensor([0.0])
    rmse = torch.Tensor([0.0])
    
    loss_meter = AverageMeter()
    rmse_meter = AverageMeter()
    all_targets = torch.tensor([]).cuda(non_blocking=True)
    all_logits = torch.tensor([]).cuda(non_blocking=True)
    
    pbar = tqdm(enumerate(val_loader), total=val_loader.__len__(), desc='Val')
    for idx, data in pbar:
        pbar.set_description(f"MSE: {format(round(loss.item(), 4), '.4f')}, RMSE: {format(round(rmse.item(), 4), '.4f')}")
        pbar.refresh()
        
        keys = data.keys() if callable(data.keys) else data.keys
        
        for key in keys:
            data[key] = data[key].cuda(non_blocking=True)
            
        target = data['y'].squeeze(-1)
        
        data['x'] = get_features_by_keys(data, cfg.feature_keys)
        data['epoch'] = epoch
        data['iter'] = total_iter
        
        logits = model(data)
        
        if cfg.criterion_args.NAME.lower() == 'weightedmse' or cfg.criterion_args.NAME.lower() == 'reductionloss':
            loss = criterion(logits, target, bins=data['bins'])
        elif 'mask' in cfg.criterion_args.NAME.lower():
            loss = criterion(logits, target, data['mask'])
        else:
            loss = criterion(logits, target)
        
        loss_meter.update(loss.item())
        
        mse_loss = mse_criterion(logits, target)
        
        if cfg.regression:
            rmse = torch.sqrt(mse_loss)
            rmse_meter.update(rmse.item())
    
        all_targets = torch.cat((all_targets, target.view(-1) * ((target.view(-1) + 1) / 2) * 1000))
        all_logits = torch.cat((all_logits, logits.view(-1) * ((logits.view(-1) + 1) / 2) * 1000))
    
    name = f'Confusion matrix validation epoch {epoch}'
    image_dir += 'cm'
    
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)
    
    _, _, _, image_path = binned_cm(all_targets.cpu(), all_logits.cpu(), 0, 1000, 10, name=name, path=image_dir, show=False, save=True)
    wandb.log({f"Validation Confusion matrix": wandb.Image(image_path + '.png')})
    
    return loss_meter.avg, rmse_meter.avg

@torch.no_grad()
def test(cfg, model, root):    
    model.eval()
    
    mse_criterion = torch.nn.MSELoss().cuda()
    data_transform = build_transforms_from_cfg('val', cfg.datatransforms)   
    
    loss_meter = AverageMeter()
    rmse_meter = AverageMeter()
    
    files = traverse_root(root)
    
    all_targets = torch.Tensor().cuda(non_blocking=True)
    all_logits = torch.Tensor().cuda(non_blocking=True)
    
    loss = torch.Tensor([0.0])
    rmse = torch.Tensor([0.0])
    
    pbar = tqdm(enumerate(files), total=len(files), desc='Test')
    
    for idx, test_sample in pbar:
        pbar.set_description(f"Test MSE: {format(round(loss.item(), 4), '.4f')}, Test RMSE: {format(round(rmse.item(), 4), '.4f')} [kWh/m2]")
        pbar.refresh()
        
        data = np.load(test_sample).astype(np.float32)
        
        # Remove the None values (points that should not be included)
        nan_mask = np.isnan(data).any(axis=1)
        data = data[~nan_mask]
                
        # Build sample in format
        pos, normals, targets = data[:,0:3], data[:, 3:-1], data[:, -1] 
        data = {'pos': pos, 'normals': normals, 'x': normals, 'y': targets}
        
        # Transform the data
        data = data_transform(data)
        
        # Make negative 0 positive
        data['normals'] += 0.

        data['normals'] = data['normals'].unsqueeze(0)
        data['pos'] = data['pos'].unsqueeze(0)

        keys = data.keys() if callable(data.keys) else data.keys
        
        for key in keys:
            data[key] = data[key].cuda(non_blocking=True)

        data['x'] = get_features_by_keys(data, cfg.feature_keys)

        logits = model(data)[0, 0, :]    
        targets = data['y']
        
        all_targets = torch.cat((all_targets, targets))
        all_logits = torch.cat((all_logits, logits))

        loss = mse_criterion(logits, targets)
        rmse = torch.sqrt(mse_criterion(((logits + 1) / 2) * 1000, ((targets + 1) / 2) * 1000))

        loss_meter.update(loss.item())
        rmse_meter.update(rmse.item())
            
    all_targets = ((all_targets + 1) / 2) * 1000
    all_logits = ((all_logits + 1) / 2) * 1000
   
    print(f"Test Loss MSE: {loss_meter.avg}")
    print(f"Test Loss RMSE: {loss_meter.avg} [kWh/m2]")
    
    confusion_matrix, _, _, image_path = binned_cm(all_targets.cpu(), all_logits.cpu(), 0, 1000, 10, show=True)
    accuracy, precision, recall, f1_score, micro_avg_accuracy, micro_avg_precision, micro_avg_recall, micro_avg_f1_score, macro_avg_accuracy, macro_avg_precision, macro_avg_recall, macro_avg_f1_score = compute_metrics(confusion_matrix)
    
    if cfg.wandb.use_wandb:
        wandb.log({f"Test Confusion matrix": wandb.Image(image_path + '.png')})
    
    if cfg.wandb.use_wandb:
        wandb.log({"Test Loss MSE": loss_meter.avg})
        wandb.log({"Test Loss RMSE [kWh/m2]": rmse_meter.avg})
        wandb.log({"Precision": precision})
        wandb.log({"Accuracy": accuracy})
        wandb.log({"Recall": recall})
        wandb.log({"F1-score": f1_score})
        wandb.log({"Micro-averaged Accuracy": micro_avg_accuracy})
        wandb.log({"Micro-averaged Precision": micro_avg_precision})
        wandb.log({"Micro-averaged Recall": micro_avg_recall})
        wandb.log({"Micro-averaged F1-score": micro_avg_f1_score})
        wandb.log({"Macro-averaged Accuracy": macro_avg_accuracy})
        wandb.log({"Macro-averaged Precision": macro_avg_precision})
        wandb.log({"Macro-averaged Recall": macro_avg_recall})
        wandb.log({"Macro-averaged F1-score:": macro_avg_f1_score})
        
    print(f"Precision: {precision}")
    print(f"Accuracy: {accuracy}")
    print(f"Recall: {recall}")
    print(f"F1-score: {f1_score}")
    print(f"Micro-averaged Accuracy: {micro_avg_accuracy}")
    print(f"Micro-averaged Precision: {micro_avg_precision}")
    print(f"Micro-averaged Recall: {micro_avg_recall}")
    print(f"Micro-averaged F1-score: {micro_avg_f1_score}")
    print(f"Macro-averaged Accuracy: {macro_avg_accuracy}")
    print(f"Macro-averaged Precision: {macro_avg_precision}")
    print(f"Macro-averaged Recall: {macro_avg_recall}")
    print(f"Macro-averaged F1-score: {macro_avg_f1_score}")

def compute_metrics(confusion_matrix):
    # Total number of classes
    num_classes = confusion_matrix.shape[0]
    
    # Initialize arrays to store metrics
    accuracy = np.zeros(num_classes)
    precision = np.zeros(num_classes)
    recall = np.zeros(num_classes)
    f1_score = np.zeros(num_classes)
    
    for i in range(num_classes):
        # True positives
        TP = confusion_matrix[i, i]
        
        # False positives
        FP = np.sum(confusion_matrix[:, i]) - TP
        
        # False negatives
        FN = np.sum(confusion_matrix[i, :]) - TP
        
        # True negatives
        TN = np.sum(confusion_matrix) - TP - FP - FN
        
        # Accuracy
        accuracy[i] = (TP + TN) / np.sum(confusion_matrix)
        
        # Precision
        precision[i] = TP / (TP + FP) if TP + FP != 0 else 0
        
        # Recall
        recall[i] = TP / (TP + FN) if TP + FN != 0 else 0
        
        # F1-score
        f1_score[i] = 2 * precision[i] * recall[i] / (precision[i] + recall[i]) if precision[i] + recall[i] != 0 else 0
    
    # Compute micro-averaged metrics
    micro_avg_accuracy = np.sum(np.diag(confusion_matrix)) / np.sum(confusion_matrix)
    micro_avg_precision = np.sum(np.diag(confusion_matrix)) / np.sum(confusion_matrix)
    micro_avg_recall = np.sum(np.diag(confusion_matrix)) / np.sum(confusion_matrix)
    micro_avg_f1_score = 2 * micro_avg_precision * micro_avg_recall / (micro_avg_precision + micro_avg_recall) if micro_avg_precision + micro_avg_recall != 0 else 0
    
    # Compute macro-averaged metrics
    macro_avg_accuracy = np.mean(accuracy)
    macro_avg_precision = np.mean(precision)
    macro_avg_recall = np.mean(recall)
    macro_avg_f1_score = np.mean(f1_score)
    
    return accuracy, precision, recall, f1_score, micro_avg_accuracy, micro_avg_precision, micro_avg_recall, micro_avg_f1_score, macro_avg_accuracy, macro_avg_precision, macro_avg_recall, macro_avg_f1_score


@torch.no_grad()
def eval_image(model, sample, idx, name, path):
    model.eval() # set model to eval mode
    
    data = {}
    for key in sample.keys():
        data[key] = torch.clone(sample[key])
    
    data['x'] = get_features_by_keys(data, cfg.feature_keys)
    
    if data['x'].shape[0] > 1:
        data['x'] = data['x'][idx, :, :].unsqueeze(0)
        data['pos'] = data['pos'][idx, :, :].unsqueeze(0)
        data['normals'] = data['normals'][idx, :, :].unsqueeze(0)
        data['y'] = data['y'][idx].unsqueeze(0)
    
    for key in data:
        data[key] = data[key].cuda(non_blocking=True)
    
    logits = model(data)
    
    values = logits.cpu().numpy()[0, 0, :]
    
    for key in data:
        data[key] = data[key].cpu()
        
    from_sample(data, 0, values, False, True, name, path)
    
    return os.path.join(path, name)

@torch.no_grad()
def evaluate_file(model_path, data_path, cfg):
    '''
    Load the model
    '''
    model = build_model_from_cfg(cfg.model)
    load_checkpoint(model, model_path)
    
    model.eval()
    
    '''
    # Load the data
    # Set data in correct format
    # Transform data
    '''
    data = np.load(data_path).astype(np.float32)
    
    # Remove the None values (points that should not be included)
    nan_mask = np.isnan(data).any(axis=1)
    data = data[~nan_mask]
            
    # Build sample in format
    pos, normals, targets = data[:,0:3], data[:, 3:-1], data[:, -1] 
    data = {'pos': pos, 'normals': normals, 'x': normals, 'y': targets}
    
    # Transform the data
    data_transform = build_transforms_from_cfg('val', cfg.datatransforms)
    data = data_transform(data)
    
    # Make negative 0 positive
    data['normals'] += 0.

    data['normals'] = data['normals'].unsqueeze(0)
    data['pos'] = data['pos'].unsqueeze(0)

    data['x'] = get_features_by_keys(data, cfg.feature_keys)

    # ! Send all data to CPU
    for key in data.keys():
        data[key] = data[key].to("cuda")

    model.cuda()

    '''
    Compute the irradiance
    '''
    
    start = time.perf_counter()
    irradiance = model(data).cpu().numpy()[0, 0, :]
    timing = time.perf_counter() - start
    
    irradiance = ((irradiance + 1) / 2) * 1000
    
    for key in data.keys():
        data[key] = data[key].to("cpu")
    
    return data, list(irradiance)

def traverse_root(root):
    res = []
    for (dir_path, _, file_names) in os.walk(root):
        for file in file_names:
            res.append(os.path.join(dir_path, file))

    return res
    
def config_to_cfg(config):
    cfg = EasyConfig()    
    cfg.update(config.cfg)
    
    '''
    Special sweep parameters
    '''
    for key in config.keys():
        if key == 'cfg':
            pass
        elif key == 'crit':
            cfg.criterion_args.NAME = config[key]   
            
            if config[key].lower() == 'weightedmse':
                cfg.criterion_args.bins = 5
                cfg.criterion_args.min = -1
                cfg.criterion_args.max = 1
                cfg.criterion_args.weights = [1,1,1,1,0.25]
            
            if config[key].lower() == 'deltaloss':
                cfg.criterion_args.delta = 0.6
                cfg.criterion_args.power = 2
            
            if config[key].lower() == 'reductionloss':
                cfg.criterion_args.bins = 5
                cfg.criterion_args.min = -1
                cfg.criterion_args.max = 1
                cfg.criterion_args.reduction = 1
            
        elif key == 'optim':
            cfg.optimizer.NAME = config[key]
        elif key == 'voxel_max':
            cfg.dataset.train.voxel_max = config[key]
        else:
            cfg[key] = config[key]
        
    return cfg

def sweep_run(config=None):
    # tell wandb to get started
    with wandb.init(mode="online", project="IrradianceNet_loss_sweep", config=config, allow_val_change=True):        
        # access all HPs through wandb.config, so logging matches execution!
        config = wandb.config
        
        cfg = config_to_cfg(config)
        
        # multi processing.
        if cfg['mp']:
            port = find_free_port()
            cfg['dist_url'] = f"tcp://localhost:{port}"
            print('using mp spawn for distributed training')
            mp.spawn(main, nprocs=cfg['world_size'], args=(cfg,))
        else:
            main(0, cfg)

def sweep(cfg):
    # Random initialization of arguments
    sweep_config = {
        'method': 'grid'
        }
    
    # Metric
    metric = {
        'name': 'mse_loss',
        'goal': 'minimize'   
        }

    sweep_config['metric'] = metric
        
    parameters_dict = {
        'voxel_max': {
            'values': [10000, 20000]
            },
        'crit': {
            'values': ['WeightedMSE', 'DeltaLoss', 'ReductionLoss', 'MSELoss']
        }    
    }
    
    # ['plateau_lr', 'cosine_lr', 'tanh_lr', 'poly_lr']
    parameters_dict.update({
                'cfg': {'value': cfg}
                })
    
    sweep_config['parameters'] = parameters_dict
   
    project = "IrradianceNet_loss_sweep"
    sweep_id = wandb.sweep(sweep_config, project=project)
    
    wandb.agent(sweep_id, sweep_run, count=50, project=project)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser('Scene segmentation training/testing')
    parser.add_argument('--cfg', type=str, required=False, default='cfgs/irradiance/irradiancenet-l.yaml', help='config file')
    parser.add_argument('--profile', action='store_true', default=False, help='set to True to profile speed')
    parser.add_argument('--sweep', required=False, action='store_true', default=False, help='set to True to profile speed')
    args, opts = parser.parse_known_args()       
        
    name = sys.argv[2].split("/")[3][:-5] + "_" + "_".join(sys.argv[3:][1::2])
    cfg = EasyConfig()
    
    cfg.load(args.cfg, recursive=True)
    cfg.update(opts)  # overwrite the default arguments in yml

    if args.sweep:
        cfg.wandb.sweep = True

    if cfg.seed is None:
        cfg.seed = np.random.randint(1, 10000)

    # init distributed env first, since logger depends on the dist info.
    cfg.rank, cfg.world_size, cfg.distributed, cfg.mp = dist_utils.get_dist_info(cfg)
    # cfg.sync_bn = cfg.world_size > 1

    # init log dir
    cfg.task_name = args.cfg.split('.')[-2].split('/')[-2]  # task/dataset name, \eg s3dis, modelnet40_cls
    cfg.cfg_basename = args.cfg.split('.')[-2].split('/')[-1]  # cfg_basename, \eg pointnext-xl
    tags = [
        'IrradianceNet',
        cfg.task_name,  # task name (the folder of name under ./cfgs
        cfg.mode
    ]
    
    opt_list = [] # for checking experiment configs from logging file
    for i, opt in enumerate(opts):
        if 'rank' not in opt and 'dir' not in opt and 'root' not in opt and 'pretrain' not in opt and 'path' not in opt and 'wandb' not in opt and '/' not in opt:
            opt_list.append(opt)
    cfg.root_dir = os.path.join(cfg.root_dir, cfg.task_name)
    cfg.opts = '-'.join(opt_list)

    cfg.is_training = cfg.mode not in ['test', 'testing', 'val', 'eval', 'evaluation']
    
    if cfg.mode in ['resume', 'val', 'test']:
        resume_exp_directory(cfg, pretrained_path=cfg.pretrained_path)
        cfg.wandb.tags = [cfg.mode]
    else:
        generate_exp_directory(cfg, tags, additional_id=os.environ.get('MASTER_PORT', None))
        cfg.wandb.tags = tags
    
    if not os.path.exists('.\log\logs'):
        os.makedirs('.\log\logs')
    
    os.environ["JOB_LOG_DIR"] = cfg.log_dir
    
    cfg_path = os.path.join(cfg.run_dir, "cfg.yaml")
    with open(cfg_path, 'w') as f:
        yaml.dump(cfg, f, indent=2)
        
        # ! check if system is Windows, copy cfg to log
        if os.name == 'nt':
            shutil.copy(args.cfg, cfg.run_dir)
        else:
            os.system('cp %s %s' % (args.cfg, cfg.run_dir))

    cfg.cfg_path = cfg_path
    # wandb config
    cfg.wandb.name = cfg.run_name

    if cfg.wandb.use_wandb:
        wandb.login()
    
    cfg.mp = False
    if cfg.wandb.sweep:
        sweep(cfg)
    else:
        with wandb.init(mode="online", project="Thesis_2", name=name):
            # multi processing
            if cfg.mp:
                port = find_free_port()
                cfg.dist_url = f"tcp://localhost:{port}"
                print('using mp spawn for distributed training')
                mp.spawn(main, nprocs=cfg.world_size, args=(cfg,))
            else:
                main(0, cfg)
               
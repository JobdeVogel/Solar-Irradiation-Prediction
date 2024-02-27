"""
(Distributed) training script for scene segmentation
This file currently supports training and testing on S3DIS
If more than 1 GPU is provided, will launch multi processing distributed training by default
if you only wana use 1 GPU, set `CUDA_VISIBLE_DEVICES` accordingly
"""
import sys
import time
import datetime

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

from visualize import from_sample, plot

warnings.simplefilter(action='ignore', category=FutureWarning)

def main(gpu, cfg):
    if cfg.distributed:
        if cfg.mp:
            cfg.rank = gpu
        dist.init_process_group(backend=cfg.dist_backend,
                                init_method=cfg.dist_url,
                                world_size=cfg.world_size,
                                rank=cfg.rank)
        dist.barrier()

    # logger
    setup_logger_dist(cfg.log_path, cfg.rank, name=cfg.dataset.common.NAME)
    
    # if cfg.rank == 0:
    #     #if not cfg.wandb.sweep:        
    #     #Wandb.launch(cfg, cfg.wandb.use_wandb)
    #     writer = SummaryWriter(log_dir=cfg.run_dir) if cfg.is_training else None
    # else:
    #     writer = None
    set_random_seed(cfg.seed + cfg.rank, deterministic=cfg.deterministic)
    torch.backends.cudnn.enabled = True
    
    # ! Commented
    #logging.info(cfg)

    if cfg.model.get('in_channels', None) is None:
        cfg.model.in_channels = cfg.model.encoder_args.in_channels

    model = build_model_from_cfg(cfg.model).to(cfg.rank)
    model_size = cal_model_parm_nums(model)
    
    # ! Commented
    # logging.info(model)
    # logging.info('Number of params: %.4f M' % (model_size / 1e6))

    if cfg.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        
        # ! commented
        # logging.info('Using Synchronized BatchNorm ...')
    if cfg.distributed:
        torch.cuda.set_device(gpu)
        model = nn.parallel.DistributedDataParallel(model.cuda(), device_ids=[cfg.rank], output_device=cfg.rank)
        
        # ! commented
        # logging.info('Using Distributed Data parallel ...')

    # optimizer & scheduler
    optimizer = build_optimizer_from_cfg(model, lr=cfg.lr, **cfg.optimizer)    
    scheduler = build_scheduler_from_cfg(cfg, optimizer)  

    # build dataset
    val_loader = build_dataloader_from_cfg(cfg.get('val_batch_size', cfg.batch_size),
                                           cfg.dataset,
                                           cfg.dataloader,
                                           datatransforms_cfg=cfg.datatransforms,
                                           split='val',
                                           distributed=cfg.distributed
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
    
    train_loader = build_dataloader_from_cfg(cfg.batch_size,
                                             cfg.dataset,
                                             cfg.dataloader,
                                             datatransforms_cfg=cfg.datatransforms,
                                             split='train',
                                             distributed=cfg.distributed,
                                             )

    logging.info(f"length of training dataset: {len(train_loader.dataset)}")

    if not cfg.regression:
        cfg.criterion_args.weight = None
        if cfg.get('cls_weighed_loss', False):
            if hasattr(train_loader.dataset, 'num_per_class'):
                cfg.criterion_args.weight = get_class_weights(train_loader.dataset.num_per_class, normalize=True)
            else:
                logging.info('`num_per_class` attribute is not founded in dataset')

    criterion = build_criterion_from_cfg(cfg.criterion_args).cuda()

    mse_criterion = torch.nn.MSELoss().cuda()
    
    # ===> start training
    if cfg.use_amp:
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None

    best_val, best_epoch = 0., 0
    
    test_array = iter(val_loader)
    
    evaluation_test_array_0 = next(test_array)
    evaluation_test_array_1 = next(test_array)
    evaluation_test_array_2 = next(test_array)
    evaluation_test_array_3 = next(test_array)
    evaluation_test_array_4 = next(test_array)
    evaluation_train_array = next(iter(train_loader))
    
    total_iter = 0
    
    if cfg.wandb.use_wandb:
        wandb.watch(model, criterion, log="all", log_freq=1)
        
    from_date = "{:%Y_%m_%d_%H_%M_%S}".format(datetime.now())
    image_dir = f'.\\images\\{cfg.cfg_basename}\\{from_date}\\'
    
    if not os.path.exists(image_dir + '\\evaluation'):
        os.makedirs(image_dir)
    
    if not os.path.exists(image_dir + '\\training'):
        os.makedirs(image_dir)
    
    logging.info('Logging initial images...')
    max_images = min([5, cfg.batch_size])
    max_evaluation_images = 5
    
    for idx in range(max_images):
        if idx == 0:
            image_path_0 = eval_image(model, evaluation_test_array_0, idx, f'Epoch base test 0 sample {idx}', image_dir + '\\evaluation')
            image_path_1 = eval_image(model, evaluation_test_array_1, idx, f'Epoch base test 1 sample {idx}', image_dir + '\\evaluation')
            image_path_2 = eval_image(model, evaluation_test_array_2, idx, f'Epoch base test 2 sample {idx}', image_dir + '\\evaluation')
            image_path_3 = eval_image(model, evaluation_test_array_2, idx, f'Epoch base test 3 sample {idx}', image_dir + '\\evaluation')
            image_path_4 = eval_image(model, evaluation_test_array_2, idx, f'Epoch base test 4 sample {idx}', image_dir + '\\evaluation')
            
            if cfg.wandb.use_wandb:
                wandb.log({f"Evaluation Irradiance Predictions 0 {idx}": wandb.Image(image_path_0 + '.png')}, step=0)
                wandb.log({f"Evaluation Irradiance Predictions 1 {idx}": wandb.Image(image_path_1 + '.png')}, step=0)
                wandb.log({f"Evaluation Irradiance Predictions 2 {idx}": wandb.Image(image_path_2 + '.png')}, step=0)
                wandb.log({f"Evaluation Irradiance Predictions 3 {idx}": wandb.Image(image_path_3 + '.png')}, step=0)
                wandb.log({f"Evaluation Irradiance Predictions 4 {idx}": wandb.Image(image_path_4 + '.png')}, step=0)

        image_path = eval_image(model, evaluation_train_array, idx, f'Epoch base train sample {idx}', image_dir + '\\training')
        
        if cfg.wandb.use_wandb:
            wandb.log({f"Train Irradiance Predictions {idx}": wandb.Image(image_path + '.png')}, step=0)
       
    logging.info('Started training...')
    for epoch in range(cfg.start_epoch, cfg.epochs + 1):
        
        # ! Only important for distributed gpu
        if cfg.distributed:
            train_loader.sampler.set_epoch(epoch)
        if hasattr(train_loader.dataset, 'epoch'):  # some dataset sets the dataset length as a fixed steps.
            train_loader.dataset.epoch = epoch - 1
        
        train_loss, train_rmse, total_iter = \
            train_one_epoch(model, train_loader, criterion, mse_criterion, optimizer, scheduler, scaler, epoch, total_iter, cfg)
        
        # ! Log the results from the epoch step
        is_best = False
        
        if epoch % cfg.val_freq == 0:
            eval_loss, eval_rmse = validate_fn(model, val_loader, criterion, cfg, epoch=epoch, total_iter=total_iter)
            
            if eval_loss < best_val:
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
                    wandb.log({f"Evaluation Irradiance Predictions {idx}": wandb.Image(image_path + '.png')})

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
            wandb.log({'Evaluation Loss (mse)': eval_loss}, step=epoch)
            wandb.log({'Evaluation Loss (rmse) [kWh/m2]': eval_rmse}, step=epoch)
        
        # writer.add_scalar('train_loss', train_loss, epoch)
        # writer.add_scalar('RMSE per train step [kWh/m2]', train_rmse, epoch)
        wandb.log({'Learning Rate': lr}, step=epoch)

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
                            is_best=is_best
                            )
            is_best = False
        
    # do not save file to wandb to save wandb space
    # if writer is not None:
    #     Wandb.add_file(os.path.join(cfg.ckpt_dir, f'{cfg.run_name}_ckpt_best.pth'))
    # Wandb.add_file(os.path.join(cfg.ckpt_dir, f'{cfg.logname}_ckpt_latest.pth'))

    # validate
    with np.printoptions(precision=2, suppress=True):
        if epoch % cfg.val_freq == 0:
            logging.info(
                f'Best ckpt @E{best_epoch},  eval loss {eval_loss:.2f}, train loss {train_loss:.2f}')
        else:
            logging.info(
                f'Best ckpt @E{best_epoch},  train loss {train_loss:.2f}')
    
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
    
    for idx, data in pbar:
        try:
            pbar.set_description(f"Average loss: {format(round(loss_meter.avg, 4), '.4f')}, Average RMSE: {format(round(rmse_meter.avg, 4), '.4f')}, Loss: {round(loss.item(), 4)}")
            pbar.refresh()
            time.sleep(0.01)
        except:
            pass
        
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
            
            loss = criterion(logits, target) if 'mask' not in cfg.criterion_args.NAME.lower() \
                else criterion(logits, target, data['mask'])
                
            mse_loss = mse_criterion(logits, target)
            
            wandb.log({'train_loss': loss})
            wandb.log({'mse_loss': mse_loss})
            
            if cfg.regression:
                rmse_scaled = torch.sqrt(mse_loss)
                rmse = rmse_scaled * (1000 - 0) / 2
                rmse_meter.update(rmse.item())
                wandb.log({'RMSE per train step [kWh/m2]': rmse})
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

@torch.no_grad()
def validate(model, val_loader, criterion, cfg, num_votes=1, data_transform=None, epoch=-1, total_iter=-1):
    model.eval()  # set model to eval mode
    
    loss_meter = AverageMeter()
    rmse_meter = AverageMeter()
    logits = []
    
    pbar = tqdm(enumerate(val_loader), total=val_loader.__len__(), desc='Val')
    for idx, data in pbar:
        try:
            pbar.set_description(f"MSE: {format(round(loss, 4), '.4f')}, RMSE: {format(round(rmse, 4), '.4f')}, npoints: {str(logits.shape[1]).zfill(5)}")
            pbar.refresh()
            time.sleep(0.01)
        except:
            pass
        
        keys = data.keys() if callable(data.keys) else data.keys
        
        for key in keys:
            data[key] = data[key].cuda(non_blocking=True)
            
        target = data['y'].squeeze(-1)
        
        data['x'] = get_features_by_keys(data, cfg.feature_keys)
        data['epoch'] = epoch
        data['iter'] = total_iter 
        
        logits = model(data)
        
        loss = criterion(logits, target) if 'mask' not in cfg.criterion_args.NAME.lower() \
                else criterion(logits, target, data['mask'])
            
        loss_meter.update(loss.item())
        
        if cfg.regression:
            rmse = torch.sqrt(loss)
            rmse_meter.update(rmse.item())

    return loss_meter.avg, rmse_meter.avg

@torch.no_grad()
def eval_image(model, sample, idx, name, path):
    data = {}
    for key in sample.keys():
        data[key] = torch.clone(sample[key])
    
    # Evaluate image
    with torch.no_grad():
        model.eval()
        data['x'] = get_features_by_keys(data, cfg.feature_keys)
        
        if data['x'].shape[0] > 1:
            data['x'] = data['x'][0, :, :].unsqueeze(0)
            data['pos'] = data['pos'][0, :, :].unsqueeze(0)
            data['normals'] = data['normals'][0, :, :].unsqueeze(0)
            data['y'] = data['y'][0].unsqueeze(0)
        
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

def test(cfg, root, blank=True):
    model_path = cfg.pretrained_path
    
    for data_path in traverse_root(root):
        data, irradiance = evaluate_file(model_path, data_path, cfg)

        targets = ((data['y'] + 1) / 2) * 1000
        
        if not blank:
            targets = targets.tolist()
        else:
            targets = [0] * len(targets)
        
        plot('0', 
            data['pos'][0, :, :], 
            vectors=[],
            targets = targets,
            values = irradiance,
            show_normals=False, 
            vector_length=1.0,
            save=False,
            show=True,
            name='',
            path = '',
            blank=blank
            )
    
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
        elif key == 'optim':
            cfg.optimizer.NAME = config[key]
        elif key == 'voxel_max':
            cfg.dataset.train.voxel_max = config[key]
        else:
            cfg[key] = config[key]
    
    return cfg

def sweep_run(config=None):
    # tell wandb to get started
    with wandb.init(mode="disabled", project="IrradianceNet", config=config, allow_val_change=True):        
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
    
    early_terminate = {
        'type': 'hyperband',
        'min_iter': 10
    }
    
    sweep_config['early_terminate'] = early_terminate
    
    # parameters_dict = {
    #     'batch_size': {
    #         'values': [4, 8, 16]
    #         },
    #     'sched': {
    #         'values': ['cosine', 'step', 'tanh', 'plateau']
    #         },
    #     'optim': {
    #         'values': ['adamw', 'adamp', 'nadam', 'adam', 'sgdp']
    #         },
    #     'crit': {
    #         'values': ['MSELoss', 'HuberLoss', 'L1Loss']
    #     }
    # }
    
    parameters_dict = {
        'voxel_max': {
            'values': [10000, 15000, 20000, 25000]
            }
        }
    
    # ['plateau_lr', 'cosine_lr', 'tanh_lr', 'poly_lr']
    parameters_dict.update({
                'cfg': {'value': cfg}
                })
    
    sweep_config['parameters'] = parameters_dict
   
    project = "IrradianceNet"
    sweep_id = wandb.sweep(sweep_config, project=project)
    
    wandb.agent(sweep_id, sweep_run, count=50, project=project)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser('Scene segmentation training/testing')
    parser.add_argument('--cfg', type=str, required=True, help='config file')
    parser.add_argument('--profile', action='store_true', default=False, help='set to True to profile speed')
    args, opts = parser.parse_known_args()
    
    cfg = EasyConfig()
    
    cfg.load(args.cfg, recursive=True)
    cfg.update(opts)  # overwrite the default arguments in yml

    if cfg.seed is None:
        cfg.seed = np.random.randint(1, 10000)

    # init distributed env first, since logger depends on the dist info.
    cfg.rank, cfg.world_size, cfg.distributed, cfg.mp = dist_utils.get_dist_info(cfg)
    cfg.sync_bn = cfg.world_size > 1

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

    test(cfg, "D:\\Master Thesis Data\\bag", blank=False)
    sys.exit()
    
    if cfg.wandb.sweep:
        sweep(cfg)
    else:
        with wandb.init(mode="disabled", project="IrradianceNet"):
            # multi processing
            if cfg.mp:
                port = find_free_port()
                cfg.dist_url = f"tcp://localhost:{port}"
                print('using mp spawn for distributed training')
                mp.spawn(main, nprocs=cfg.world_size, args=(cfg,))
            else:
                main(0, cfg)
               
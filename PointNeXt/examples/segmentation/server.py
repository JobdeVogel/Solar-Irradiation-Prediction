print('Booting server... (this can take several seconds)')
print('Please do not quit the server while booting')
import sys
import time

start = time.perf_counter()
import __init__
import argparse, yaml, numpy as np
import torch, torch.nn as nn
from openpoints.utils import load_checkpoint, EasyConfig
from openpoints.dataset import get_features_by_keys
from openpoints.dataset.data_util import voxelize
from openpoints.transforms import build_transforms_from_cfg
from openpoints.models import build_model_from_cfg
import json

import socket
import time

import argparse

end = time.perf_counter()
print(f'Loaded packages in {end-start}s')

parser = argparse.ArgumentParser(prog='name', description='random info', epilog='random bottom info')
parser.add_argument('-p', '--port', type=str, nargs='?', default=50007, help='')
args= parser.parse_args() 

PORT = int(args.port)
BUFFER_SIZE = 196608 * 2 ** 8
CFG = None

CFG_PATH = r'S:\jdevogel\log\irradiance\IrradianceNet-irradiance-train-20240525-143354-WCYpcf8rMMus5LQeUPNfFE\\'

PRETRAINED_PATH_XL = r"S:\jdevogel\log\irradiance\IrradianceNet-irradiance-train-20240525-143354-WCYpcf8rMMus5LQeUPNfFE\\checkpoint\\IrradianceNet-irradiance-train-20240525-143354-WCYpcf8rMMus5LQeUPNfFE_ckpt_epoch_25.pth"

class Time:
    def __init__(self, name):
        self.name = name
    
    def __enter__(self): self.start = time.perf_counter()
    def __exit__(self, *args): print(f"Server finished {self.name} in {round(time.perf_counter() - self.start, 3)}s...")

def build_model(model_name):   
    cfg = EasyConfig()

    model_cfg_path = CFG_PATH + 'cfg' + '.yaml'
    
    with open(model_cfg_path) as f:
        cfg.update(yaml.load(f, Loader=yaml.Loader))
    
    model = build_model_from_cfg(cfg.model).cuda()
    
    model_paths = {
        'irradiancenet-xl': PRETRAINED_PATH_XL
    }
    
    print('Loading model weights and biases...')
    load_checkpoint(model, model_paths.get(model_name, None))

    model.eval()
        
    if not hasattr(cfg.datatransforms, 'test'):
        cfg.datatransforms.test = ['PointsToTensor', 'PointCloudCenterAndNormalize']
        print('WARNING: manual test transform overwrite')
    
    transform = build_transforms_from_cfg('test', cfg.datatransforms)
    
    return model, transform, cfg

def crop_pc(coord, feat, label, split='train',
            voxel_size=0.04, voxel_max=None,
            downsample=True, variable=True, shuffle=False):
    if voxel_size and downsample:
        # Is this shifting a must? I borrow it from Stratified Transformer and Point Transformer. 
        coord -= coord.min(0) 
        uniq_idx = voxelize(coord, voxel_size)
        coord, feat, label = coord[uniq_idx], feat[uniq_idx] if feat is not None else None, label[uniq_idx] if label is not None else None
    if voxel_max is not None:
        crop_idx = None
        N = len(label)  # the number of points
        if N >= voxel_max:
            init_idx = np.random.randint(N) if 'train' in split else N // 2
            crop_idx = np.argsort(
                np.sum(np.square(coord - coord[init_idx]), 1))[:voxel_max]
        elif not variable:
            # fill more points for non-variable case (batched data)
            cur_num_points = N
            query_inds = np.arange(cur_num_points)
            padding_choice = np.random.choice(
                cur_num_points, voxel_max - cur_num_points)
            crop_idx = np.hstack([query_inds, query_inds[padding_choice]])
        crop_idx = np.arange(coord.shape[0]) if crop_idx is None else crop_idx
        if shuffle:
            shuffle_choice = np.random.permutation(np.arange(len(crop_idx)))
            crop_idx = crop_idx[shuffle_choice]
        coord, feat, label = coord[crop_idx], feat[crop_idx] if feat is not None else None, label[crop_idx] if label is not None else None
    coord -= coord.min(0) 
    return coord.astype(np.float32), feat.astype(np.float32) if feat is not None else None , label if label is not None else None
    #return coord.astype(np.float32), feat.astype(np.float32) if feat is not None else None , label.astype(np.long) if label is not None else None

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

def standardize(logits, targets, cfg):
    dmin = cfg.datatransforms.kwargs.irradiance_min
    dmax = 1
    
    if dmin == None: dmin = -1
    
    # Standardize logits
    logits = (logits - dmin) / (dmax - dmin)
    if targets != None:
        targets = (targets - dmin) / (dmax - dmin)

    # Return in range [0, 1]
    return logits, targets

def preprocess(data, transform, cfg, local_client=False, extract_labels=False):
    print(f'Data preprocessing: {sys.getsizeof(data)} bytes')

    if local_client:
        data_path = r'S:\jdevogel\data\dset100_xl_regular\7-368-528-LoD12-3D\irradiance_sample_971_augmentation_0.npy'
        cdata = np.load(data_path).astype(np.float32)
    else:
        cdata = np.array(data).astype(np.float32)

    # Remove the None values (points that should not be included)
    nan_mask = np.isnan(cdata).any(axis=1)
    cdata = cdata[~nan_mask]
    """
    ! Original code uses voxelization here, since I do not want to downsample, skip the voxelization!
    """
    if extract_labels:
        coord, feat, labels = np.split(cdata, [3, 6], axis=1)
    else:
        coord, feat = np.split(cdata, [3], axis=1)
        labels = None
    """
    ! Cropping is not required since I am not downsampling    
    ! However it will change to [0, 100] domain
    """
    
    coord, feat, _ = crop_pc(
                coord, 
                feat,
                None,
                'val', 
                False, 
                None,
                downsample=False, variable=False, shuffle=False)

    data = {'pos': coord, 'x': feat}
    
    if transform != None:
        data = transform(data)
    else:
        print("WARNING: no transform available")
    
    if 'normals' not in data.keys():
        data['normals'] =  torch.from_numpy(feat.astype(np.float32))
    
    # Remove negative zero values in normals
    data['normals'] += 0.
    data['x'] += 0.

    data['pos'] = data['pos'].unsqueeze(0)
    data['normals'] = data['normals'].unsqueeze(0)
    data['x'] = data['x'].unsqueeze(0)

    data['x'] = get_features_by_keys(data, cfg.feature_keys)

    for key in data.keys():
        data[key] = data[key].to("cuda")
    
    if extract_labels:
        data['y'] = labels

    return data

@torch.no_grad()
def forward(model, data, cfg, extract_labels=False):
    with Time('forward pass'):
        irradiance = model(data)
    
    torch.cuda.synchronize()
    with Time('getting data from gpu to cpu'):
        irradiance = irradiance.cpu().numpy()
    
    irradiance = np.squeeze(irradiance)
    irradiance, _ = standardize(irradiance, None, cfg)
    
    irradiance *= 1000
    
    if extract_labels:
        labels = np.array([irr[0] for irr in data['y'].tolist()])
        rmse = np.sqrt(np.mean((labels - (irradiance))**2))
        
        print(irradiance, irradiance.shape)
        print(labels, labels.shape)
        print(f'RMSE: {rmse}')

    return irradiance.tolist()

def server(func, local_client=False):       
    HOST = ''  # Symbolic name meaning all available interfaces
    model = None
    loaded_model = None
    cfg = None
    x = None # Data to forward
    i = 0
    
    serv_start = time.time()
    
    if not local_client:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:

            s.bind((HOST, PORT))
            s.listen(1)
            print('...Server listening on port', PORT)

            available_models = ['irradiancenet-s', 'irradiancenet-b', 'irradiancenet-l', 'irradiancenet-xl']

            while True:            
                try:
                    conn, addr = s.accept()
                    print('Connected by', addr)
                    print('-'*40)

                    with conn:
                        try:
                            req_buffer_size = int(conn.recv(1024).decode())

                            data = conn.recv(req_buffer_size)
                            data = data.decode()
                        except Exception as e:
                            print(e)

                        if data == "exit":
                            print('Closing server (this can take several seconds)')
                            sys.exit()               
                        else:
                            print(f'Processing data {sys.getsizeof(data)} bytes of data')

                        with Time('data extraction'):
                            data = json.loads(data)

                        model_name, data, kwargs = data                   

                        model_change = False
                        
                        if loaded_model != model_name:
                            if loaded_model == None:
                                print("Initial loading of model, processing might take significantly more time than second client call.")
                            
                            if model_name in available_models:
                                with Time('building model'):
                                    model, transform, cfg = build_model(model_name)
                                    loaded_model = model_name

                                print(f'Server built model {model_name}')
                            else:
                                print('WARNING: this model is not available!')

                            model_change = True

                        # TODO: comparison not working for floats
                        if x != data:
                            x = data

                            with Time('preprocess'):
                                try:
                                    x = preprocess(x, transform, cfg)
                                except Exception as e:
                                    print(e)
                                    print('FAILED')

                            if model != None:
                                with Time('forward'):
                                    try:
                                        irradiance = forward(model, x, cfg)
                                    except Exception as e:
                                        print('FAILED')
                                        print(e)
                            else:
                                print('WARNING: network not available on server!')
                                irradiance = []

                        start = time.perf_counter()
                        if model_change:
                            data = str(json.dumps([i, model_name, irradiance]))
                            conn.sendall(data.encode())
                        else:
                            # Do not send model_name so that client knows it has not changed
                            data = str(json.dumps([i, '', irradiance]))
                            conn.sendall(data.encode())

                        print(f'Data packed and send in {round(time.perf_counter() - start, 3)}s')
                        print('-'*40)

                        i += 1

                except Exception as e:
                    print(e)
    else:
        # Run client locally
        print('Running client locally...')
        
        model_name = input('Model name: ')
        model, transform, cfg = build_model(model_name)
        
        data = preprocess(None, transform, cfg, local_client=True, extract_labels=True)
        
        irradiance = forward(model, data, cfg, extract_labels=True)
        
server(build_model, local_client=False)

#model = build_model('irradiancenet-xl')

    
    
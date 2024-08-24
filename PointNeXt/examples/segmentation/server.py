print('Booting server... (this can take several seconds)')
print('Please do not quit the server while booting')
import sys
import time

start = time.perf_counter()
import __init__
import argparse, yaml, numpy as np
from openpoints.utils import load_checkpoint, EasyConfig
from openpoints.dataset import get_features_by_keys
from openpoints.dataset.data_util import voxelize
from openpoints.transforms import build_transforms_from_cfg
from openpoints.models import build_model_from_cfg
from openpoints.utils import set_random_seed, cal_model_parm_nums
import torch.backends.cudnn as cudnn
import torch
import json

import socket
import time
import random

import argparse

def check(vars):
    try:
        import code
        code.interact(local=vars)
    except SystemExit:
        pass

parser = argparse.ArgumentParser(prog='name', description='random info', epilog='random bottom info')
parser.add_argument('-p', '--port', type=str, nargs='?', default=50007, help='')
parser.add_argument('-t', '--time_it', type=bool, nargs='?', default=False, help='')
parser.add_argument('-bs', '--batch_size', type=int, nargs='?', default=1, help='')
parser.add_argument('-sbs', '--sub_batch_size', type=int, nargs='?', default=1, help='')
parser.add_argument('-a', '--autocast', type=bool, nargs='?', default=True, help='')
parser.add_argument('-cd', '--cudnn', type=bool, nargs='?', default=True, help='')
parser.add_argument('-d', '--deterministic', type=bool, nargs='?', default=False, help='')
parser.add_argument('-c', '--compile', type=bool, nargs='?', default=False, help='')
parser.add_argument('-s', '--synchronize', type=str, nargs='?', default=False, help='')
parser.add_argument('-o', '--overwrite_model_path', type=str, nargs='?', default=None, help='')
args= parser.parse_args() 

PORT = int(args.port)
TIME_IT = bool(args.time_it)
BATCH_SIZE = int(args.batch_size)
SUB_BATCH_SIZE = int(args.sub_batch_size)
AUTOCAST = bool(args.autocast)
CUDNN = bool(args.cudnn)
DETERMINISTIC = bool(args.deterministic)
COMPILE = bool(args.compile)
SYNCHRONIZE = bool(args.synchronize)
OVERWRITE_MODEL_PATH = args.overwrite_model_path

BUFFER_SIZE = 196608 * 2 ** 8

if DETERMINISTIC: set_random_seed(12345, deterministic=True)
if CUDNN:
    # Enable cuDNN auto-tuner to find the best algorithm
    cudnn.benchmark = True
else:
    cudnn.benchmark = False   

PRETRAINED_PATH_POINTNET = r"D:\Master Thesis Data\pretrained_irradiancenet\base_pointnet_regular_100\checkpoint\IrradianceNet-irradiance-train-20240527-094731-d4Esm2RFFq5fQrxoQCR5L6_ckpt_best.pth"
PRETRAINED_PATH_L = r"D:\Master Thesis Data\pretrained_irradiancenet\L_super\checkpoint\weights.pth"
PRETRAINED_PATH_XL = r"D:\Master Thesis Data\pretrained_irradiancenet\XL_super\checkpoint\weights.pth"

end = time.perf_counter()
print(f'Loaded packages in {end-start}s')

class Time:
    def __init__(self, name, time_it=True, synchronize=True, vram=False):
        self.name = name
        self.time_it = time_it
        self.synchronize = synchronize
        self.vram = vram
    
    def __enter__(self):
        if self.vram:
            torch.cuda.reset_peak_memory_stats()
        
        if self.time_it:
            self.start = time.perf_counter()
    def __exit__(self, *args):
        if self.synchronize:
            torch.cuda.synchronize()
        
        if self.time_it:
            print(f"Server finished {self.name} in {round(time.perf_counter() - self.start, 3)}s...")

        if self.vram:
            # Get the peak memory usage in bytes
            peak_memory = torch.cuda.max_memory_allocated()
        
            # Convert bytes to megabytes
            peak_memory_MB = peak_memory / (1024 ** 2)
            
            print(f"VRAM used: {round(peak_memory_MB, 2)} MB")

def build_model(model_name):
    model_paths = {
        'pointnet': PRETRAINED_PATH_POINTNET,
        'irradiancenet-l': PRETRAINED_PATH_L,
        'irradiancenet-xl': PRETRAINED_PATH_XL
    }
    
    cfg = EasyConfig()
    
    if OVERWRITE_MODEL_PATH ==  None:
        cfg_path = model_paths[model_name].split("checkpoint")[0]
    else:
        cfg_path = OVERWRITE_MODEL_PATH.split("checkpoint")[0]
    
    model_cfg_path = cfg_path + 'cfg' + '.yaml'

    with open(model_cfg_path) as f:
        cfg.update(yaml.load(f, Loader=yaml.Loader))
    
    # Overwrite loaded parameters
    cfg.test = True
    cfg.pretrained_path = model_paths[model_name]

    
    model = build_model_from_cfg(cfg.model).cuda()
    
    if OVERWRITE_MODEL_PATH == None:
        print('Loading model weights and biases...')
        load_checkpoint(model, model_paths.get(model_name, None))
    else:
        print('no')
        print('Loading model weights and biases...') 
        load_checkpoint(model, OVERWRITE_MODEL_PATH)

    model.eval()
    
    model_size = cal_model_parm_nums(model)
    print(f'Model parameters: {round(model_size / 1000000,2)}M')
    
    # Manually overwrite cfg datatransforms
    cfg.datatransforms.test = ['PointsToTensor', 'PointCloudCenterAndNormalize']
    
    transform = build_transforms_from_cfg('test', cfg.datatransforms)
    
    return model, transform, cfg

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

def preprocess(data, transform, cfg, extract_labels=False, batch_size=8):
    if data == None:
        data_path = r'D:\Master Thesis Data\IrradianceNet\test_dset100_xl_regular\10-226-552-LoD12-3D\irradiance_sample_13_augmentation_0.npy'
        
        cdata = np.load(data_path).astype(np.float64)
        
        # Remove the None values (points that should not be included)
        nan_mask = np.isnan(cdata).any(axis=1)
        cdata = cdata[~nan_mask]
        
        data = []
        data.append(cdata[:, :6])
        
        for _ in range(batch_size - 1):
            size = random.randint(15000, 20000)
            
            random_array = np.random.rand(size, 6)
            data.append(random_array)
    else:
        data = [np.array(d).astype(np.float32) for d in data]
        
        for d in data:
            # Remove the None values (points that should not be included)
            nan_mask = np.isnan(d).any(axis=1)
            d = d[~nan_mask]
    
    original_sizes = [d.shape[0] for d in data] 
    max_size = max([d.shape[0] for d in data])
    
    # Preprocess the data here:
    for idx, sample in enumerate(data):            
        cur_num_points = sample.shape[0]
        query_inds = np.arange(cur_num_points)
        
        padding_choice = np.random.choice(
            cur_num_points, max_size - cur_num_points)
        crop_idx = np.hstack([query_inds, query_inds[padding_choice]])

        crop_idx = np.arange(sample.shape[0]) if crop_idx is None else crop_idx
        
        sample = sample[crop_idx, :].astype(np.float32)
    
        coord, feat = sample[:,0:3], sample[:, 3:]
        
        minima = coord.min(axis=0, keepdims=True)
        coord -= minima 

        sample = {'pos': coord, 'x': feat}
        
        if transform != None:
            sample = transform(sample)
        else:
            print("WARNING: no transform available")
        
        if 'normals' not in sample.keys():
            sample['normals'] =  torch.from_numpy(feat.astype(np.float32))

        # Remove negative zero values in normals
        sample['normals'] += 0.
        sample['x'] += 0.
        
        data[idx] = sample
    
    stacked_data = {'pos': torch.zeros(batch_size, max_size, 3), 'x': torch.zeros(batch_size, max_size, 3), 'normals': torch.zeros(batch_size, max_size, 3)}
    
    for i, sample in enumerate(data):
        stacked_data['pos'][i, :, :] = sample['pos'].unsqueeze(0)
        stacked_data['x'][i, :, :] = sample['x'].unsqueeze(0)
        stacked_data['normals'][i, :, :] = sample['normals'].unsqueeze(0)
    
    for key in stacked_data.keys():
        stacked_data[key] = stacked_data[key].to("cuda")
    
    stacked_data['x'] = get_features_by_keys(stacked_data, cfg.feature_keys)
    
    return stacked_data, original_sizes

@torch.no_grad()
def forward(model, data, cfg, extract_labels=False, autocast=True):
    print("Infering data...")
    
    with torch.cuda.amp.autocast(enabled=autocast):    
        irradiance = model(data)

    irradiance = irradiance.cpu().numpy()
    
    irradiances = [irradiance[i, 0, :] for i in range(irradiance.shape[0])]
    
    for i, irr in enumerate(irradiances):
        irradiances[i], _ = standardize(irr, None, cfg)
        irradiances[i] *= (irradiances[i] * 1000).tolist()

    return irradiances

class Server():
    def __init__(self, port, host='', local=False):
        self.port = port
        self.host = host
        self.local = local
        

def server(local_client=False, time_it=True, synchronize=True):       
    HOST = ''  # Symbolic name meaning all available interfaces
    model = None
    loaded_model = None
    cfg = None
    x = None # Data to forward
    i = 0
    
    if not local_client:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:

            s.bind((HOST, PORT))
            s.listen(1)
            print('...Server listening on port', PORT)

            available_models = ['pointnet', 'irradiancenet-l', 'irradiancenet-xl']

            while True:
                try:
                    conn, addr = s.accept()
                    print('Connected by', addr)
                    print('-'*40)
                    
                    with conn:
                        with Time('End-to-end irradiation prediction', vram=True):
                            with Time('decoding data', time_it=time_it, synchronize=synchronize):
                                try:
                                    req_buffer_size = int(conn.recv(1024).decode())

                                    data = conn.recv(req_buffer_size)
                                    data = data.decode()                              
                                    
                                except Exception as e:
                                    print(e)

                            with Time('checking for exit', time_it=time_it, synchronize=synchronize):
                                if data == "exit":
                                    print('Closing server (this can take several seconds)')
                                    sys.exit()               
                                else:
                                    print(f'Processing data {sys.getsizeof(data)} bytes of data')

                            with Time('data extraction', time_it=time_it, synchronize=synchronize):
                                data = json.loads(data)

                            model_name, data, kwargs = data      
                            model_change = False
                            
                            TIME_IT, BATCH_SIZE, SUB_BATCH_SIZE, AUTOCAST, CUDNN, DETERMINISTIC, COMPILE, SYNCHRONIZE, OVERWRITE_MODEL_PATH = kwargs
                            
                            if len(data) != BATCH_SIZE:
                                print("WARNING: batch size does not correspond with number of samples:")
                                print(f"BATCH_SIZE set to {len(data)}")
                                
                                BATCH_SIZE = len(data)
                            
                            if (len(data) < SUB_BATCH_SIZE) or (len(data) % SUB_BATCH_SIZE != 0):
                                print("WARNING: sub batch size does not correspond with number of samples:")
                                print(f"SUB_BATCH_SIZE set to 1")
                                
                                SUB_BATCH_SIZE = 1
                            
                            time_it = TIME_IT
                            synchronize = SYNCHRONIZE
                            
                            if DETERMINISTIC: set_random_seed(12345, deterministic=True)
                            if CUDNN:
                                # Enable cuDNN auto-tuner to find the best algorithm
                                cudnn.benchmark = True
                            else:
                                cudnn.benchmark = False   
                                
                            
                            if loaded_model != model_name:
                                if loaded_model == None:
                                    print("Initial loading of model, processing might take significantly more time than second client call.")
                                
                                if model_name in available_models:
                                    with Time('building model', time_it=TIME_IT, synchronize=SYNCHRONIZE):
                                        model, transform, cfg = build_model(model_name)
                                        loaded_model = model_name
                                else:
                                    print('WARNING: this model is not available!')

                                model_change = True

                            # TODO: comparison not working for floats
                            if x != data:
                                x = data

                                with Time('preprocess', time_it=TIME_IT, synchronize=SYNCHRONIZE):
                                    try:                                    
                                        x, original_sizes = preprocess(x, transform, cfg, batch_size=BATCH_SIZE)
                                    except Exception as e:
                                        print(e)
                                        print('FAILED PREPROCESSING')

                                if model != None:
                                    x['irr'] = []
                
                                    num_iter = int(BATCH_SIZE / SUB_BATCH_SIZE)
                                    
                                    with Time(f'batch inference ({BATCH_SIZE} samples, {SUB_BATCH_SIZE} sub batch size)', time_it=TIME_IT, synchronize=SYNCHRONIZE):
                                        for i in range(num_iter):
                                            d = {}
                                                            
                                            d['normals'] = x['normals'][i*SUB_BATCH_SIZE:i*SUB_BATCH_SIZE + SUB_BATCH_SIZE, :, :]
                                            d['pos'] = x['pos'][i*SUB_BATCH_SIZE:i*SUB_BATCH_SIZE + SUB_BATCH_SIZE, :, :]
                                            d['x'] = x['x'][i*SUB_BATCH_SIZE:i*SUB_BATCH_SIZE + SUB_BATCH_SIZE, :, :] 
                                    
                                            try:                                   
                                                irradiances = forward(model, d, cfg, autocast=AUTOCAST)
                                                
                                            except Exception as e:
                                                print(e)
                                                print('FAILED FORWARDING')
                                                
                                            for irradiance in irradiances:
                                                x['irr'].append(irradiance)
                                                
                                            del d
                                            del irradiance
                                            torch.cuda.empty_cache()
                                    
                                        # Remove all padded items
                                        x['irr'] = [x['irr'][idx][:original_sizes[idx]].tolist() for idx in range(BATCH_SIZE)]
                                    
                                    irradiance = x['irr']
                                else:
                                    print('WARNING: network not available on server!')
                                    irradiance = []

                                del x; x = None
                                del data
                                torch.cuda.empty_cache()

                            with Time('packing and sending data', time_it=TIME_IT, synchronize=SYNCHRONIZE):
                                if model_change:
                                    with Time('encoding data', time_it=TIME_IT, synchronize=SYNCHRONIZE):
                                        data = str(json.dumps([i, model_name, irradiance]))
                                    conn.sendall(data.encode())
                                else:
                                    with Time('encoding data', time_it=TIME_IT, synchronize=SYNCHRONIZE):
                                        # Do not send model_name so that client knows it has not changed
                                        data = str(json.dumps([i, '', irradiance]))
                                    conn.sendall(data.encode())
                            
                            print('-'*40)
                        print('')
                        
                except Exception as e:
                    print(e)
    else:
        # Run client locally
        print("Running client locally...")
        
        model_name = input("Model name: ")
        model, transform, cfg = build_model(model_name)
               
        data, original_sizes = preprocess(x, transform, cfg, extract_labels=False, batch_size=BATCH_SIZE)        
        """
        with Time('sequential run', synchronize=SYNCHRONIZE):
            for i in range(BATCH_SIZE):
                torch.cuda.reset_peak_memory_stats()
                d = {}
                
                # One sample:
                d['pos'] = data['pos'][i, :, :].unsqueeze(0)
                d['normals'] = data['normals'][i, :, :].unsqueeze(0)
                d['x'] = data['x'][i, :, :].unsqueeze(0)

                with Time('subsequential run', synchronize=SYNCHRONIZE):
                    irradiance = forward(model, d, cfg, extract_labels=False, autocast=autocast)

                # Get the peak memory usage in bytes
                peak_memory = torch.cuda.max_memory_allocated()

                # Convert bytes to megabytes
                peak_memory_MB = peak_memory / (1024 ** 2)
                print(f"VRAM used: {round(peak_memory_MB, 2)} MB")
        
            torch.cuda.synchronize()
        """
        data['irr'] = []
               
        num_iter = int(BATCH_SIZE / SUB_BATCH_SIZE)
        
        with Time(f'batch inference ({BATCH_SIZE} samples, {SUB_BATCH_SIZE} sub batch size)', time_it=TIME_IT, synchronize=SYNCHRONIZE):
            for i in range(num_iter):
                d = {}
                                
                d['normals'] = data['normals'][i*SUB_BATCH_SIZE:i*SUB_BATCH_SIZE + SUB_BATCH_SIZE, :, :]
                d['pos'] = data['pos'][i*SUB_BATCH_SIZE:i*SUB_BATCH_SIZE + SUB_BATCH_SIZE, :, :]
                d['x'] = data['x'][i*SUB_BATCH_SIZE:i*SUB_BATCH_SIZE + SUB_BATCH_SIZE, :, :]              
                
                # One subbatch:
                irradiances = forward(model, d, cfg, extract_labels=False)

                for irradiance in irradiances:
                    data['irr'].append(irradiance)
                    
                del d
                del irradiance

        # Remove all padded items
        data['irr'] = [data['irr'][idx][:original_sizes[idx]].tolist() for idx in range(BATCH_SIZE)]
        
        return data['irr']
        """
        # -------------------        
        # Temp visualization:
        idx = 0
        
        values = data['irr'][idx]               
    
        from visualize import from_sample
        for key in data:
            try:
                data[key] = data[key].cpu()
            except AttributeError:
                pass
    
        data['pos'] = data['pos'][idx, :original_sizes[idx], :].unsqueeze(0)
        
        # Temp targets to 0.0
        data['y'] = torch.zeros(BATCH_SIZE, data['pos'].shape[1])
        data['y'] = data['y'][idx, :].unsqueeze(0)
        
        from_sample(data, 0, values, True, False, 'name', '')
        print('finished')
        sys.exit()
        # -------------------
        """
                        
server(local_client=False, time_it=TIME_IT, synchronize=SYNCHRONIZE)
# print(irradiance)

#model = build_model('irradiancenet-xl')




    
    
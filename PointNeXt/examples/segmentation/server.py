print('Booting server... (this can take several seconds)')
print('Please do not quit the server while booting')
import sys
import time

start = time.perf_counter()
import __init__
import argparse, yaml, os, logging, numpy as np, csv, wandb, glob
from tqdm import tqdm
import torch, torch.nn as nn
from torch import distributed as dist, multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter
# from torch_scatter import scatter
from openpoints.utils import set_random_seed, save_checkpoint, load_checkpoint, resume_checkpoint, setup_logger_dist, \
    cal_model_parm_nums, Wandb, generate_exp_directory, resume_exp_directory, EasyConfig, dist_utils, find_free_port, load_checkpoint_inv
from openpoints.utils import AverageMeter, ConfusionMatrix, get_mious
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

CFG_PATH = 'C:\\Users\\Job de Vogel\\OneDrive\\Documenten\\TU Delft\\Master Thesis\\Code\\IrradianceNet\\PointNeXt\\cfgs\\irradiance\\'

PRETRAINED_PATH_S = r"C:\Users\\Job de Vogel\\OneDrive\\Documenten\\TU Delft\\Master Thesis\\Code\\IrradianceNet\\PointNeXt\\log\\irradiance\\IrradianceNet-irradiance-train-20240219-112609-QPEAG59dYA4CXdapcEsZXQ\\checkpoint\\pretrained.pth"
PRETRAINED_PATH_B = ''
PRETRAINED_PATH_L = ''
PRETRAINED_PATH_XL = ''

class Time:
    def __init__(self, name):
        self.name = name
    
    def __enter__(self): self.start = time.perf_counter()
    def __exit__(self, *args): print(f"Server finished {self.name} in {round(time.perf_counter() - self.start, 3)}s...")

def build_model(model):   
    cfg = EasyConfig()

    cfg_name = CFG_PATH + model + '.yaml'
    cfg.load(cfg_name, recursive=True)
    
    network = build_model_from_cfg(cfg.model).to("cuda")
    
    model_paths = {
        'irradiancenet-s': PRETRAINED_PATH_S,
        'irradiancenet-b': PRETRAINED_PATH_B,
        'irradiancenet-l': PRETRAINED_PATH_L,
        'irradiancenet-xl': PRETRAINED_PATH_XL
    }

    model_path = model_paths.get(model, None)

    try:
        load_checkpoint(network, model_path)
    except Exception as e:
        print(e)
        print('Loading checkpoint failed...')

    network.eval()
    
    return cfg, model, network

def preprocess(data, cfg):
    print(f'Data preprocessing: {sys.getsizeof(data)} bytes')
       
    data = np.array(data).astype(np.float32)
    points = data[:, :3]
    normals = data[:, 3:]
    
    # Build sample in format
    # Targets are random data
    pos, normals = points, normals
    
    targets = np.array([0]  * points.shape[0]).astype(np.float32)
    
    data = {'pos': pos, 'normals': normals, 'x': normals, 'y': targets}
    
        # Transform the data
    data_transform = build_transforms_from_cfg('val', cfg.datatransforms)
    data = data_transform(data)
    
    # Make negative 0 positive
    data['normals'] += 0.

    data['normals'] = data['normals'].unsqueeze(0)
    data['pos'] = data['pos'].unsqueeze(0)

    data['x'] = get_features_by_keys(data, cfg.feature_keys)

    for key in data.keys():
        data[key] = data[key].to("cuda")
    
    return data

@torch.no_grad()
def forward(model, data):
    with Time('forward pass'):
        irradiance = model(data)
    
    with Time('getting data from gpu to cpu'):
        irradiance = irradiance.cpu().numpy()
    
    irradiance = np.squeeze(irradiance) 
    irradiance = ((irradiance + 1) / 2) * 1000
       
    irradiance = irradiance.tolist()
    
    return irradiance

def server(func):       
    HOST = ''  # Symbolic name meaning all available interfaces
    model = None
    cfg = None
    network = None
    x = None # Data to forward
    i = 0
    
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        
        s.bind((HOST, PORT))
        s.listen(1)
        print('...Server listening on port', PORT)
        
        available_models = ['irradiancenet-s', 'irradiancenet-b', 'irradiancenet-l', 'irradiancenet-xl']
        
        while True:
            try:
                conn, addr = s.accept()
                print('Connected by', addr)
                print('-'*80)

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
                    if model != model_name:
                        if model_name in available_models:
                            with Time('building model'):
                                cfg, model, network = build_model(model_name)

                            print(f'Server built model {model_name}')
                        else:
                            print('WARNING: this model is not available!')

                        model_change = True

                    # TODO: comparison not working for floats
                    if x != data:
                        x = data

                        with Time('preprocess'):
                            x = preprocess(data, cfg)

                        if network != None:
                            with Time('forward'):
                                irradiance = forward(network, x)
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
                    print('-'*80)
                    
                    i += 1

            except Exception as e:
                print(e)
                        
server(build_model)
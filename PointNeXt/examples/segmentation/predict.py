try:
    import time
    start = time.perf_counter()
    import __init__

    import torch
    import torch.nn as nn
    import numpy as np

    from openpoints.utils import load_checkpoint, EasyConfig
    from openpoints.transforms import build_transforms_from_cfg
    from openpoints.dataset import get_features_by_keys
    from openpoints.models import build_model_from_cfg
    end = time.perf_counter()
    print(f'Import timing: {end-start}s')
except ImportError as importerror:
    print(importerror)
    print('Please make sure you are using a valid installation for the irradiancenet package.')
    print('Calling the script from subprocess only works from a Python 3.9 installation, not 3.7.')
except Exception as e:
    print(e)

import argparse
import sys

parser = argparse.ArgumentParser(prog='name', description='random info', epilog='random bottom info')
parser.add_argument('-s', '--sensors', type=str, nargs='?', default='', help='')
parser.add_argument('-n', '--normals', type=str, nargs='?', default='', help='')
parser.add_argument('-m', '--model', type=str, nargs='?', default='', help='')
# parser.add_argument('-m', '--model', type=str, nargs='?', default='', help='')

args= parser.parse_args() 
    
@torch.no_grad()
def evaluate_opt(model, points, normals, cfg, plot=False):
    '''
    Load the model
    '''
    cfg = EasyConfig()

    cfg_name = 'C:\\Users\\Job de Vogel\\OneDrive\\Documenten\\TU Delft\\Master Thesis\\Code\\IrradianceNet\\PointNeXt\\cfgs\\irradiance\\' + model + '.yaml'
    cfg.load(cfg_name, recursive=True)
    
    model = build_model_from_cfg(cfg.model).to("cuda")
    model_path ="C:\\Users\\Job de Vogel\\OneDrive\\Documenten\\TU Delft\\Master Thesis\\Code\\IrradianceNet\\PointNeXt\\log\\irradiance\\IrradianceNet-irradiance-train-20240213-154852-bxo6cZn9ZWaFBrHScmtuDj\\checkpoint\\IrradianceNet-irradiance-train-20240213-154852-bxo6cZn9ZWaFBrHScmtuDj_ckpt_best.pth"
    load_checkpoint(model, model_path)

    model.eval()
    
    points = np.array(eval(points)).astype(np.float32)
    normals = np.array(eval(normals)).astype(np.float32)
    
    # Build sample in format
    # ! WRONGGGG:
    
    pos, normals, targets = points, normals, np.array([0]).astype(np.float32) * len(points)
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
    
    irradiance = model(data).cpu().numpy()[0, 0, :]  
    irradiance = ((irradiance + 1) / 2) * 1000
    print(irradiance)

def main(args):
    evaluate_opt(args.model, args.sensors, args.normals, '', plot=False)

if __name__ == '__main__':
    main(args)
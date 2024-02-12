
import sys
import json

import argparse
import time
import random

def forward(num_sensors):
    return [random.randrange(0,1000) for _ in range(num_sensors)]  

def main():
    parser = argparse.ArgumentParser(prog='name', description='random info', epilog='random bottom info')
    parser.add_argument('-s', '--sensors', type=str, nargs='?', default='', help='')
    parser.add_argument('-n', '--normals', type=str, nargs='?', default='', help='')
    args= parser.parse_args()
    
    num_sensors = eval(args.sensors)
    
    start = time.perf_counter()
    irradiance = forward(num_sensors)   
    # print(f'\nFinished computing {num_sensors} irradiance values in {round(time.perf_counter() - start, 2)}s')
    irradiance.append(time.perf_counter() - start)
    print(irradiance)
    
if __name__ == '__main__':
    main()
import os
import time

path = "C:\\Users\\Job de Vogel\\OneDrive\\Documenten\\TU Delft\\Master Thesis\\Dataset_pipeline\\dataset\\data\\geometry"

def get_latest_file(dir):
    files = os.listdir(dir)
    paths = [os.path.join(dir, basename) for basename in files]
    
    latest_file = max(paths, key=os.path.getctime)
    
    return latest_file, paths

files = os.listdir(path)
subfolders = []
for file in os.listdir(path):
    d = os.path.join(path, file)
    if os.path.isdir(d):
        subfolders.append(d)

all_files = []

for folder in subfolders:
    latest_file, files = get_latest_file(folder)
    all_files.extend(files)

initial_count = len(all_files)

counts = []

start = time.perf_counter()
print('initializing...')
time.sleep(60)
while True:
    files = os.listdir(path)
    subfolders = []
    for file in os.listdir(path):
        d = os.path.join(path, file)
        if os.path.isdir(d):
            subfolders.append(d)
    
    all_files = []
    
    for folder in subfolders:
        latest_file, files = get_latest_file(folder)
        all_files.extend(files)
    
    number_of_samples = len(all_files)
    number_of_samples -= initial_count
    
    end = time.perf_counter()
    
    try:
        past_time = (end-start)/60
        per_min = number_of_samples / past_time
        counts.append(per_min)
    except ZeroDivisionError:
        per_min = 0
        counts.append(per_min)
    
    print(f'{sum(counts) / len(counts)} it/minute')
    time.sleep(5)

import requests
import os
import shutil
import sys
import time
from zipfile import ZipFile 

VERSION = 'v20230809'

# Maximum number of files to download within the map
max_urls=100

# Minimum and maximum coordinates from the 3D BAG map
x_min=272
x_max=332
y_min=488
y_max=532

def generate_url(id):
    items = id.split("-")

    url = f'https://data.3dbag.nl/obj/{VERSION}/tiles/{items[0]}/{items[1]}/{items[2]}/{id}-obj.zip'
    
    return url

def validate_url(url):
    response = requests.get(url)

    if response.status_code == 200:
        return True
    else:
        return False

def generate_id(i, x, y):
    return f'{i}-{x}-{y}'

def compute_skip_coordinates(x, y, size):
    coordinates = set()

    for u in range(size):
        for v in range(size):
            coordinates.add((x+u,y+v))
    
    return coordinates

def download_file(id, directory):
    print(f'Downloading {id}...')

    start = time.perf_counter()
    url = generate_url(id)

    r = requests.get(url, allow_redirects=True)

    filename = f'{id}.zip'

    path = os.path.join(directory, filename)

    open(path, 'wb').write(r.content)

    with ZipFile(path, 'r') as file: 
        file.extractall(path=path[:-4]) 

    os.remove(path)

    for file in os.listdir(path[:-4]):
        filepath = os.path.join(path[:-4], file)

        if 'LoD12-3D.obj' not in filepath or '.mtl' in filepath:
            os.remove(filepath)

        else:
            new_path = os.path.join(directory, path[:-4]) + str('.obj')

            try:
                shutil.move(filepath, directory)
            except shutil.Error:
                print(f'{new_path} already exists, will not be overwritten')
                os.remove(filepath)
            
    os.rmdir(path[:-4])
    print(f'Saved file with id {id} in {round(time.perf_counter() - start, 2)}s...')

def download(directory, max_urls=100, x_min=272, x_max=332, y_min=488, y_max=532):
    urls = []
    skip_coordinates = set()

    sizes = {5:64, 6:32, 7:16, 8:8, 9:4, 10:2, 11:1}

    size_likelihood = [9, 8, 10, 11, 7, 6, 5]

    for x in range(x_min, x_max+1):
        for y in range(y_min, y_max+1):
            
            if (x,y) not in skip_coordinates: # check if this (x,y) coordinate should be checked
                for i in size_likelihood:                   
                    id = generate_id(i, x, y)
                    url = generate_url(id)
                    
                    if validate_url(url): # The url exists
                        print(f'{id} is a valid id')

                        # The url does exist
                        size = sizes[i]
                        skip_coordinates.update(compute_skip_coordinates(x,y, size))
                        
                        urls.append(url)
                        
                        download_file(id, directory)

                        # Break the i iterations
                        break
                    else:
                        # The url does not exist
                        print(f'{id} is not a valid id')
                        pass
            else:
                # The coordinates should be skipped
                pass
        
            if len(urls) == max_urls:
                print(f'Downloaded {max_urls} files!')
                sys.exit()

directory = str(input('Path to save bag data: '))

directory = r'D:\\graduation_jobdevogel\\Graduation-Building-Technology\\dataset\\data\\bag'

download(directory, max_urls=max_urls, x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max)

print(f'Downloaded all requested files!')
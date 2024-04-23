import argparse
import os
import shutil
import random
import tqdm

random.seed(12345)

def traverse_root(root):
    res = []
    for (dir_path, _, file_names) in os.walk(root):
        for file in file_names:
            res.append(os.path.join(dir_path, file))

    return res

parser = argparse.ArgumentParser(prog='name', description='random info', epilog='random bottom info')
parser.add_argument('-p', '--path', type=str, nargs='?', help='')
parser.add_argument('-r', '--reverse', type=bool, nargs='?', help='')

args= parser.parse_args()

files = traverse_root(args.path)
num_files = len(files)

factor = 0.1
test_idxs = random.sample(range(0, num_files), int(factor * num_files))

for idx in tqdm.tqdm(test_idxs):
    test_sample = files[idx]
    paths = test_sample.split("\\")
    root = "\\".join(paths[:-3]) + "\\" + "test_" + paths[-3]
    
    destination = root + "\\" + paths[-2]

    #######
    if not os.path.exists(destination):
        os.makedirs(destination)

    shutil.move(test_sample, destination)
    # shutil.move(test_sample, )

for folder in list(os.walk(args.path))[1:]:
    if not folder[2]:
        try:
            os.rmdir(folder[0])
        except:
            pass

eval_size = int(factor * num_files)
eval_ratio = (factor * num_files) / (num_files - int(factor * num_files))

path = '\\'.join(paths[:-3])

with open(f"{path}\\eval_{eval_size}_{paths[-3]}.txt", 'w') as file:
    file.write(f"eval_size {eval_size}\n")
    file.write(f"eval_ratio {eval_ratio}")

print(f"Recommended validation size: {eval_size}")
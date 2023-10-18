import multiprocessing
from main import main
from parameters.params import BAG_FILE_PATH, BAG_PATH, GEOMETRY_PATH, IRRADIANCE_PATH, OUTLINES_PATH, RAW_PATH
import os
from log.logger import generate_logger
import subprocess

def task(file, logger):
    bag_file = BAG_PATH + file
    
    bag_file = BAG_PATH + file
        
    geometry_path = f'{GEOMETRY_PATH}{file[:-4]}/'
    irradiance_path = f'{IRRADIANCE_PATH}{file[:-4]}/'
    raw_path = f'{RAW_PATH}{file[:-4]}/'
    
    if not os.path.exists(geometry_path):
        os.makedirs(geometry_path)
    
    if not os.path.exists(irradiance_path):
        os.makedirs(irradiance_path)
    
    if not os.path.exists(raw_path):
        os.makedirs(raw_path)
        
    # arguments = [
    #         '--BAG_FILE_PATH', bag_file,
    #         '--GEOMETRY_PATH', geometry_path,
    #         '--IRRADIANCE_PATH', irradiance_path,
    #         '--RAW_PATH', raw_path
    #     ]

    # environment_name = "graduation"
    # script = './main.py'

    # activate_command = f"conda activate {environment_name}"
    # script_command = ["python", script] + arguments

    # # Construct the platform-specific command to open a new terminal window
    # window_command = ["cmd", "/c", "start"]

    # # Combine activation command and script command
    # full_command = [activate_command, "&&"] + script_command

    # # Construct the final command to run in a new terminal window
    # window_command += full_command
    # process  = subprocess.Popen(window_command)

    # print('Running process...')
    # return_code = process.wait()
    # # print('Finished process...')
    
    main(bag_file, 0, logger, geometry_path=geometry_path, irradiance_path=irradiance_path, raw_path=raw_path)

def process(idx, file):
    # Initialize a logger
    identifier = file.split("/")[-1][:-4]
    
    print(f"(stdout:) Initialized process {idx} for file {file}")
    
    if idx == 0:
        logger = generate_logger(identifier=identifier, stdout=True)
        logger.info(f"Sending log output to stdout for process 0, file: {file}")
    else:
        logger = generate_logger(identifier=identifier, stdout=False)
    
    task(file, logger)

if __name__ =='__main__':
    MAIN_LOGGER = generate_logger('main', stdout=True)  
    folder_paths = [GEOMETRY_PATH, IRRADIANCE_PATH, OUTLINES_PATH, RAW_PATH]
    # delete_dataset(folder_paths, secure=True)
    
    if not os.path.exists(GEOMETRY_PATH):
        os.makedirs(GEOMETRY_PATH)
    
    if not os.path.exists(IRRADIANCE_PATH):
        os.makedirs(IRRADIANCE_PATH)
    
    if not os.path.exists(OUTLINES_PATH):
        os.makedirs(OUTLINES_PATH)
    
    if not os.path.exists(RAW_PATH):
        os.makedirs(RAW_PATH)
    
    args = os.listdir(BAG_PATH)
    cpus = 7
    
    # for file in args[:cpus]:
    #     task(file, None)
    
    MAIN_LOGGER.info(f'Initializing pool with {cpus} cpus based on dataset {BAG_PATH}')
    with multiprocessing.Pool(processes=cpus) as pool:
        # Map the main function to the arguments using the pool
        data = pool.starmap_async(process, enumerate(args))
        pool.close()
        
        pool.join()

# arguments = [
#         '--BAG_FILE_PATH', bag_file,
#         '--GEOMETRY_PATH', geometry_path,
#         '--IRRADIANCE_PATH', irradiance_path,
#         '--RAW_PATH', raw_path
#     ]

# environment_name = "graduation"
# script = './main.py'

# activate_command = f"conda activate {environment_name}"
# script_command = ["python", script] + arguments

# # Construct the platform-specific command to open a new terminal window
# window_command = ["cmd", "/c", "start"]

# # Combine activation command and script command
# full_command = [activate_command, "&&"] + script_command

# # Construct the final command to run in a new terminal window
# window_command += full_command
# process  = subprocess.Popen(window_command)

# # print('Running process...')
# # return_code = process.wait()
# # print('Finished initializing process...')
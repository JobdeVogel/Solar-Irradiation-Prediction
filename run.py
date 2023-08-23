import multiprocessing
from main import delete_dataset
from parameters.params import BAG_FILE_PATH, BAG_PATH, GEOMETRY_PATH, IRRADIANCE_PATH, OUTLINES_PATH, RAW_PATH
import os
import subprocess

if __name__ =='__main__':
    folder_paths = [GEOMETRY_PATH, IRRADIANCE_PATH, OUTLINES_PATH, RAW_PATH]
    # delete_dataset(folder_paths, secure=True)
    
    
    environment_name = "graduation"
    script = './main.py'
    
    bag_files = os.listdir(BAG_PATH)    
    
    for i, file in enumerate(bag_files):
        bag_file = BAG_PATH + file
        
        geometry_path = f'./data/geometry/{file[:-4]}/'
        irradiance_path = f'./data/irradiance/{file[:-4]}/'
        raw_path = f'./data/raw/{file[:-4]}/'
        
        if not os.path.exists(geometry_path):
            os.makedirs(geometry_path)
        
        if not os.path.exists(irradiance_path):
            os.makedirs(irradiance_path)
        
        if not os.path.exists(raw_path):
            os.makedirs(raw_path)
        
        arguments = [
            '--BAG_FILE_PATH', bag_file,
            '--GEOMETRY_PATH', geometry_path,
            '--IRRADIANCE_PATH', irradiance_path,
            '--RAW_PATH', raw_path
        ]
        
        activate_command = f"conda activate {environment_name}"
        script_command = ["python", script] + arguments
        
        # Construct the platform-specific command to open a new terminal window
        window_command = ["cmd", "/c", "start"]
        
        # Combine activation command and script command
        full_command = [activate_command, "&&"] + script_command
        
        # Construct the final command to run in a new terminal window
        window_command += full_command
        
        # Run the command in a new terminal window using subprocess
        subprocess.Popen(window_command, shell=True)